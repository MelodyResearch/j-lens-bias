import json
import re

import pytest

from rq4 import config as C

NB = C.ROOT / "notebooks" / "rq4_colab.ipynb"


@pytest.fixture(scope="module")
def nb():
    return json.loads(NB.read_text())


def code_cells(nb):
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def all_text(nb):
    return "\n".join("".join(c["source"]) for c in nb["cells"])


def test_notebook_is_valid_nbformat_with_empty_outputs(nb):
    assert nb["nbformat"] == 4
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None
            compile("".join(c["source"]), "cell", "exec")          # every code cell is valid Python (no magics)
    assert len({c["id"] for c in nb["cells"]}) == len(nb["cells"])


def test_hosted_gpu_verification_comes_first(nb):
    first = code_cells(nb)[0]
    for needle in ("COLAB_RELEASE_TAG", "nvidia-smi", "linux", "free", "SystemExit"):
        assert needle in first
    assert "local runtime" in first.lower()


def test_exact_commit_pin_and_clone(nb):
    t = all_text(nb)
    assert t.count('CODE_COMMIT = "28fdda141ce139018e9d5a290fa784378b760aa3"') == 1
    assert "__RQ4_CODE_COMMIT__" not in t
    assert "https://github.com/MelodyResearch/j-lens-bias.git" in t
    assert "git\", \"checkout\", \"--detach\", CODE_COMMIT" in t and "rev-parse" in t
    assert 'os.environ["RQ4_CODE_COMMIT"] = CODE_COMMIT' in t
    assert re.search(r"fullmatch\(r\"\[0-9a-f\]\{40\}\"", t)              # unreplaced placeholder cannot proceed


def test_pinned_dependencies_installed_remotely(nb):
    t = all_text(nb)
    assert "requirements-colab.txt" in t and "pip" in t
    req = (C.ROOT / "requirements-colab.txt").read_text()
    pins = [l for l in req.splitlines() if l and not l.startswith("#")]
    assert all("==" in l or "@ git+" in l for l in pins)
    assert any(l.startswith("torch==") for l in pins) and "581d398613e5602a5af361e1c34d3a92ea82ba8e" in req


def test_drive_mount_and_persistent_results_env(nb):
    t = all_text(nb)
    assert "drive.mount(\"/content/drive\")" in t and 'os.environ["RQ4_RESULTS_DIR"] = RESULTS_DIR' in t
    assert 'RESULTS_DIR = f"/content/drive/MyDrive/' in t and "/proc/mounts" in t


def test_hf_home_is_ephemeral_vm_path_not_drive_and_public_download(nb):
    t = all_text(nb)
    assert 'HF_HOME = "/content/hf_home"' in t and 'assert not HF_HOME.startswith("/content/drive")' in t
    assert "snapshot_download" in t and "hf_hub_download" in t
    assert re.search(r"2fc06364715b967f1860aea9cf38778875588b17", t) and "16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a" in t
    assert "HF_TOKEN" not in t and "login(" not in t


def test_kernel_never_loads_models_or_uses_mps(nb):
    for src in code_cells(nb):
        assert not re.search(r"^\s*(import|from)\s+(torch|transformers|jlens)\b", src, re.M), src[:80]
        assert "from_pretrained" not in src and not re.search(r"\bmps\b", src.lower())


def test_no_secrets_or_credentials(nb):
    t = all_text(nb)
    assert not re.search(r"hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{20,}", t)
    assert "os.environ)" not in t and "printenv" not in t and "!env" not in t


def test_pilot_then_validation_then_review_then_stage2_order_and_gates(nb):
    cells = code_cells(nb)
    idx = lambda needle: next(i for i, c in enumerate(cells) if needle in c)
    pilot, valid, show = idx('"--stage", "1"'), idx("--validate-stage1"), idx("--show-stage1-generations")
    ack, stage2 = idx("ACK = False"), idx('"--stage", "2"')
    assert pilot < valid <= show < ack < stage2
    assert "--approve-stage1" in cells[ack] and "--reviewer" in cells[ack] and "ACK = False" in cells[ack]
    assert 'STAGE1_APPROVED' in cells[stage2] and "LOCKED" in cells[stage2]
    # Stage 2 command appears only in the gated branch
    assert cells[stage2].index("LOCKED") < cells[stage2].index('"--stage", "2"')


def test_analysis_and_result_locations_exposed(nb):
    t = all_text(nb)
    assert "--analyze" in t and "figures" in t and "tables" in t and "Results folder (Drive)" in t
    assert "nothing is simulated" in t


def test_timeouts_and_no_background_execution_are_stated(nb):
    t = all_text(nb).lower()
    for needle in ("not always-on", "does **not guarantee** a gpu", "nothing runs in the background", "disconnect", "resum"):
        assert needle in t
    assert "zero measured rows" in t and "pilot_only" in t


def test_download_cell_pins_exact_revisions_one_shard_and_hashes(nb):
    cell = next(s for s in code_cells(nb) if "snapshot_download" in s)
    assert "snapshot_download(C.MODEL_NAME, revision=C.MODEL_REVISION)" in cell
    assert "hf_hub_download(C.LENS_REPO, C.LENS_FILE, revision=C.LENS_REVISION)" in cell
    assert 'got["n_model_shards"] == 1' in cell and "!= C.MODEL_REVISION" not in cell
    for h in (C.MODEL_SHA256, C.LENS_SHA256, C.MODEL_REVISION, C.LENS_REVISION):
        assert h in cell
    assert "in (v, None)" not in cell                                      # model hash may not be skipped


def test_notebook_pins_match_config_and_are_full_commits():
    assert re.fullmatch(r"[0-9a-f]{40}", C.MODEL_REVISION) and re.fullmatch(r"[0-9a-f]{40}", C.LENS_REVISION)
    assert (C.MODEL_REVISION, C.LENS_REVISION) == ("2fc06364715b967f1860aea9cf38778875588b17", "16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a")
    assert C.LENS_TAG == "qwen-n1000" and C.MODEL_SHA256.startswith("04b1c301") and C.LENS_SHA256.startswith("aa26b68e")


def test_revision_pins_change_fingerprint_and_are_in_provenance(monkeypatch):
    d = C.config_dict()
    assert d["model_revision"] == C.MODEL_REVISION and d["lens"][1] == C.LENS_REVISION
    assert d["model_sha256"] == C.MODEL_SHA256 and d["lens_sha256"] == C.LENS_SHA256
    base = C.config_fingerprint()
    monkeypatch.setattr(C, "MODEL_REVISION", "0" * 40)
    assert C.config_fingerprint() != base
    monkeypatch.undo()
    monkeypatch.setattr(C, "LENS_REVISION", "qwen-n1000")
    assert C.config_fingerprint() != base
    from rq4 import remote
    assert {"model_revision", "lens_revision"} <= set(remote.PINNED_KEYS)


def test_every_runtime_from_pretrained_uses_pinned_revision():
    import ast
    tree = ast.parse((C.ROOT / "rq4" / "runtime.py").read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "from_pretrained"]
    assert len(calls) == 3                                                   # model, tokenizer, lens
    for c in calls:
        rev = next((k.value for k in c.keywords if k.arg == "revision"), None)
        assert rev is not None and ast.unparse(rev) in ("C.MODEL_REVISION", "C.LENS_REVISION")
    lens = [c for c in calls if "JacobianLens" in ast.unparse(c.func)]
    assert len(lens) == 1 and "C.LENS_REVISION" in ast.unparse(lens[0])
    assert sum("C.MODEL_REVISION" in ast.unparse(c) for c in calls) == 2
    assert "model_revision" in (C.ROOT / "rq4" / "runtime.py").read_text()          # recorded in provenance


def test_gpu_and_pip_cells_never_imply_paid_compute(nb):
    cells = nb["cells"]
    idx = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and "nvidia-smi" in "".join(c["source"])][0]
    gpu_text = ("".join(cells[idx - 1]["source"]) + "".join(cells[idx]["source"])).lower()
    for needle in ("free", "premium", "stop"):
        assert needle in gpu_text
    pip_i = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and "pip" in "".join(c["source"]) and "install" in "".join(c["source"])][0]
    pip_text = ("".join(cells[pip_i - 1]["source"]) + "".join(cells[pip_i]["source"])).lower()
    assert "paid" in pip_text and "decline" in pip_text
