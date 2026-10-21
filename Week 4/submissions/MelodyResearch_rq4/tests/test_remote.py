import json
import subprocess
import sys
from pathlib import Path

import pytest

from rq4 import config as C
from rq4 import remote

SHA = "0123456789abcdef0123456789abcdef01234567"
DRIVE_MOUNTS = "overlay / overlay rw 0 0\ndrive /content/drive fuse.drive rw 0 0\ntmpfs /dev/shm tmpfs rw 0 0\n"
ident = lambda p: p


def vr(raw, mounts=DRIVE_MOUNTS):
    return remote.validate_results_dir(raw, mounts_text=mounts, realpath=ident, code_root=Path("/content/j-lens-bias/x"))


# ---- macOS / platform refusal -------------------------------------------------------------------------
def test_inference_refused_on_darwin_and_non_linux():
    with pytest.raises(SystemExit, match="remote-only"):
        remote.refuse_inference_on_this_platform("darwin")
    with pytest.raises(SystemExit):
        remote.refuse_inference_on_this_platform("win32")
    remote.refuse_inference_on_this_platform("linux")


def test_require_remote_config_refuses_darwin_even_with_perfect_config():
    env = {C.ENV_RESULTS_DIR: "/content/drive/MyDrive/rq4", C.ENV_CODE_COMMIT: SHA}
    with pytest.raises(SystemExit):
        remote.require_remote_config(env, platform="darwin", mounts_text=DRIVE_MOUNTS, head=SHA)


def test_backend_constructor_refuses_off_linux_before_importing_torch():
    src = (C.ROOT / "rq4" / "runtime.py").read_text()
    ctor = src[src.index("def __init__(self):"):]
    assert ctor.index('sys.platform != "linux"') < ctor.index("import torch")
    assert "mps" not in src.replace("CPU/MPS fallback", "").lower()          # no MPS path for inference anywhere in the backend


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
def test_runner_cli_refuses_inference_on_this_mac_with_any_flags(tmp_path):
    for flags in (["--run-inference"], ["--run-inference", "--i-confirm-no-heavy-experiment", "--stage", "1"]):
        r = subprocess.run([sys.executable, str(C.ROOT / "run_rq4.py"), *flags], capture_output=True, text=True, cwd=tmp_path,
                           env={"PATH": "/usr/bin", C.ENV_RESULTS_DIR: "/content/drive/MyDrive/x", C.ENV_CODE_COMMIT: SHA})
        assert r.returncode != 0 and "remote-only" in r.stderr + r.stdout


# ---- results dir guard --------------------------------------------------------------------------------
def test_results_dir_unset_relative_dotdot_refused():
    for bad in (None, "", "   ", "results", "./results", "/content/drive/MyDrive/../../tmp/x"):
        with pytest.raises(remote.RemoteGuardError):
            vr(bad)


def test_results_dir_under_content_requires_persistent_mount():
    with pytest.raises(remote.RemoteGuardError):
        vr("/content/rq4_results")                                               # ephemeral
    with pytest.raises(remote.RemoteGuardError):
        vr("/content/drive/MyDrive/rq4", mounts="overlay / overlay rw 0 0\n")  # Drive not mounted: a plain dir
    with pytest.raises(remote.RemoteGuardError):
        vr("/content/drive/MyDrive/rq4", mounts="tmpfs /content/drive tmpfs rw 0 0\n")   # mount, but not FUSE
    with pytest.raises(remote.RemoteGuardError):
        vr("/content/drive")                                                     # the mount root itself
    assert vr("/content/drive/MyDrive/rq4") == Path("/content/drive/MyDrive/rq4")


def test_results_dir_outside_content_ok_but_ephemeral_dirs_and_checkout_refused():
    assert vr("/mnt/persist/rq4") == Path("/mnt/persist/rq4")
    for bad in ("/tmp/rq4", "/var/tmp/rq4", "/dev/shm/rq4"):
        with pytest.raises(remote.RemoteGuardError):
            vr(bad)
    with pytest.raises(remote.RemoteGuardError):
        vr("/content/j-lens-bias/x/results")                                     # inside the checkout


def test_symlink_into_content_is_resolved_and_refused():
    real = lambda p: "/content/scratch" if p == "/mnt/link" else p
    with pytest.raises(remote.RemoteGuardError):
        remote.validate_results_dir("/mnt/link", mounts_text=DRIVE_MOUNTS, realpath=real, code_root=Path("/x"))


def test_parse_proc_mounts():
    assert ("/content/drive", "fuse.drive") in remote.parse_proc_mounts(DRIVE_MOUNTS)
    assert remote.parse_proc_mounts("") == []


def test_ensure_writable_probe(tmp_path):
    d = tmp_path / "new" / "results"
    remote.ensure_writable(d)
    assert d.is_dir() and list(d.iterdir()) == []                                # probe files cleaned up
    f = tmp_path / "file"
    f.write_text("x")
    with pytest.raises(remote.RemoteGuardError):
        remote.ensure_writable(f / "sub")


def test_require_remote_config_checks_env_matches_config(tmp_path, monkeypatch):
    env = {C.ENV_RESULTS_DIR: "/mnt/persist/rq4", C.ENV_CODE_COMMIT: SHA}
    with pytest.raises(remote.RemoteGuardError, match="RQ4_RESULTS_DIR"):
        remote.require_remote_config(env, platform="linux", mounts_text=DRIVE_MOUNTS, head=SHA)   # config imported without the var
    with pytest.raises(remote.RemoteGuardError):
        remote.require_remote_config({C.ENV_CODE_COMMIT: SHA}, platform="linux", mounts_text="", head=SHA)


def test_config_results_dir_follows_env_in_a_fresh_interpreter(tmp_path):
    code = "import sys; sys.path.insert(0, %r); from rq4 import config as C; print(C.RESULTS_DIR, C.EVAL_DIR, C.PID_FILE)" % str(C.ROOT)
    env = {"PATH": "/usr/bin:/bin", C.ENV_RESULTS_DIR: str(tmp_path / "persist")}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=tmp_path).stdout.split()
    assert out[0] == str(tmp_path / "persist") and out[1] == str(tmp_path / "persist" / "evals")
    assert not out[2].startswith(str(tmp_path / "persist"))                      # PID lock stays VM-local


# ---- code commit & fingerprint ------------------------------------------------------------------------
def test_code_commit_validation():
    assert remote.validate_code_commit(SHA, SHA) == SHA
    for env, head in ((None, SHA), ("", SHA), (remote.PLACEHOLDER, SHA), ("abc123", SHA), (SHA.upper(), SHA), (SHA, None), (SHA, "f" * 40)):
        with pytest.raises(remote.RemoteGuardError):
            remote.validate_code_commit(env, head)


def test_fingerprint_includes_code_commit(monkeypatch):
    monkeypatch.delenv(C.ENV_CODE_COMMIT, raising=False)
    unset = C.config_fingerprint()
    monkeypatch.setenv(C.ENV_CODE_COMMIT, SHA)
    a = C.config_fingerprint()
    monkeypatch.setenv(C.ENV_CODE_COMMIT, "f" * 40)
    b = C.config_fingerprint()
    assert len({unset, a, b}) == 3 and C.config_dict()["code_commit"] == "f" * 40
    assert C.config_fingerprint(with_commit=False) == C.config_fingerprint(with_commit=False)
    assert "code_commit" not in C.config_dict(False)                              # tracked manifests stay commit-independent


def test_resume_across_a_different_commit_is_rejected(tmp_path, monkeypatch):
    from rq4 import store
    monkeypatch.setenv(C.ENV_CODE_COMMIT, SHA)
    row = {"schema_version": C.SCHEMA_VERSION, "key": store.eval_key("p1", "clean", None), "config_fingerprint": C.config_fingerprint(),
           "stage": "1", "base_item_id": "b", "pair_id": "q", "prompt_id": "p1", "context_condition": "ambig",
           "question_polarity": "neg", "prompt_format": "multiple_choice", "prompt_hash": "h", "arm": "clean", "layer": None, "status": "ok"}
    store.save_row(tmp_path, row)
    assert set(store.load_valid_rows(tmp_path, C.config_fingerprint(), tmp_path / "q")) == {row["key"]}
    monkeypatch.setenv(C.ENV_CODE_COMMIT, "f" * 40)
    with pytest.raises(store.FingerprintMismatch):
        store.load_valid_rows(tmp_path, C.config_fingerprint(), tmp_path / "q")


def test_manifests_do_not_depend_on_the_commit_env(monkeypatch):
    from rq4 import manifest
    monkeypatch.setenv(C.ENV_CODE_COMMIT, SHA)
    a = json.dumps(manifest.config_manifest() if hasattr(manifest, "config_manifest") else C.config_dict(False), sort_keys=True)
    monkeypatch.setenv(C.ENV_CODE_COMMIT, "f" * 40)
    b = json.dumps(manifest.config_manifest() if hasattr(manifest, "config_manifest") else C.config_dict(False), sort_keys=True)
    assert a == b


# ---- runtime provenance -------------------------------------------------------------------------------
def prov(**kw):
    p = {"code_commit": SHA, "config_fingerprint": "fp", "dtype": "float16", "torch": "2.14.1", "transformers": "5.20.2",
         "jlens": "0.1.0@abc", "gpu_name": "Tesla T4"}
    p.update(kw)
    return p


def test_provenance_pins_first_session_and_rejects_mixing(tmp_path):
    path = tmp_path / "prov.json"
    doc = remote.record_provenance(prov(), path)
    assert doc["pinned"]["dtype"] == "float16" and len(doc["sessions"]) == 1
    doc = remote.record_provenance(prov(gpu_name="Tesla L4"), path)               # different GPU, same dtype/libs: fine
    assert len(doc["sessions"]) == 2
    for bad in (prov(dtype="bfloat16"), prov(torch="2.15.0"), prov(code_commit="f" * 40), prov(config_fingerprint="other")):
        with pytest.raises(remote.RemoteGuardError):
            remote.record_provenance(bad, path)
    assert len(json.loads(path.read_text())["sessions"]) == 2


def test_dtype_choice_bf16_else_fp16():
    from rq4.runtime import choose_dtype

    class Cuda:
        def __init__(self, ok): self.ok = ok
        def is_bf16_supported(self, including_emulation=True): return self.ok if not including_emulation else True

    class T:
        def __init__(self, ok): self.cuda = Cuda(ok)

    assert choose_dtype(T(True)) == "bfloat16" and choose_dtype(T(False)) == "float16"   # T4: emulated bf16 is not used
