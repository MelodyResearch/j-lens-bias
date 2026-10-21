import json

import pytest

import run_rq4
from rq4 import config as C
from synth_pilot import FP

NOTE = "read all 128 raw generations and the memory log"


@pytest.fixture
def gate(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "STAGE1_APPROVAL", tmp_path / "STAGE1_APPROVED")
    monkeypatch.setattr(C, "config_fingerprint", lambda with_commit=True: FP)
    state = {"report": {"passed": True, "rows_digest": "d1", "n_rows": 128, "status_counts": {}}}
    monkeypatch.setattr(run_rq4, "_validation_report", lambda: state["report"])
    return state


def test_approval_requires_validator_pass_reviewer_and_note(gate):
    with pytest.raises(SystemExit, match="reviewer"):
        run_rq4.approve_stage1(NOTE, None)
    with pytest.raises(SystemExit, match="reviewer"):
        run_rq4.approve_stage1("ok", "melody")
    gate["report"] = {**gate["report"], "passed": False}
    with pytest.raises(SystemExit, match="validator"):
        run_rq4.approve_stage1(NOTE, "melody")
    assert not C.STAGE1_APPROVAL.exists()
    gate["report"] = {**gate["report"], "passed": True}
    run_rq4.approve_stage1(NOTE, "melody")
    a = json.loads(C.STAGE1_APPROVAL.read_text())
    assert a["fingerprint"] == FP and a["reviewer"] == "melody" and a["rows_digest"] == "d1"


def test_stage2_gate_needs_approval_matching_fingerprint_passing_validator_and_unchanged_rows(gate):
    with pytest.raises(SystemExit, match="STAGE1_APPROVED missing"):
        run_rq4._require_stage2_gate()                       # 128 files existing is irrelevant without review
    run_rq4.approve_stage1(NOTE, "melody")
    run_rq4._require_stage2_gate()                           # passes
    gate["report"] = {**gate["report"], "rows_digest": "d2"}
    with pytest.raises(SystemExit, match="changed after"):
        run_rq4._require_stage2_gate()
    gate["report"] = {**gate["report"], "rows_digest": "d1", "passed": False}
    with pytest.raises(SystemExit, match="no longer passes"):
        run_rq4._require_stage2_gate()
    gate["report"] = {**gate["report"], "passed": True}
    C.STAGE1_APPROVAL.write_text(json.dumps({"fingerprint": "other-commit", "rows_digest": "d1"}))
    with pytest.raises(SystemExit, match="different config fingerprint"):
        run_rq4._require_stage2_gate()


def test_validate_stage1_cli_writes_report_and_exit_code(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(C, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run_rq4, "_validation_report", lambda: {"passed": False, "rows_digest": "x", "n_rows": 0, "status_counts": {},
                                                                 "checks": [{"check": "K", "passed": False, "detail": "missing"}], "note": "n"})
    assert run_rq4.validate_stage1() is False
    assert json.loads((tmp_path / "stage1_validation.json").read_text())["passed"] is False
    assert "FAIL" in capsys.readouterr().out


def test_remote_run_stops_at_preload_gate_before_any_model_code(tmp_path, monkeypatch):
    """Simulated Linux VM with unreadable telemetry: the run must stop before torch/runtime is imported."""
    import sys
    from rq4 import remote, resources
    for name, val in (("RESULTS_DIR", tmp_path), ("EVAL_DIR", tmp_path / "evals"), ("RUN_STATE", tmp_path / "state.json"),
                      ("RAW_JSONL", tmp_path / "raw.jsonl"), ("PID_FILE", tmp_path / "pid"), ("MEMORY_LOG", tmp_path / "mem.jsonl")):
        monkeypatch.setattr(C, name, val)
    monkeypatch.setattr(remote, "require_remote_config", lambda *a, **k: {})
    monkeypatch.setattr(remote, "refuse_inference_on_this_platform", lambda *a, **k: None)
    monkeypatch.setattr(resources, "refuse_if_duplicate", lambda: None)
    monkeypatch.setattr(resources, "snapshot", lambda *a, **k: {"free_pct": None, "swap_used_mb": None, "gpu_free_mb": None})
    monkeypatch.setattr(resources, "log_snapshot", lambda *a, **k: None)
    from rq4 import data
    monkeypatch.setattr(run_rq4, "prepare", lambda items=None: data.plan_prompts(data.load_items(), C.STAGE1_BASE_ITEMS))
    for m in ("rq4.runtime", "torch"):
        sys.modules.pop(m, None)
    with pytest.raises(SystemExit, match="MEMORY SAFETY STOP before model load"):
        run_rq4.run_inference(1)
    assert "rq4.runtime" not in sys.modules
    assert json.loads((tmp_path / "state.json").read_text())["stopped_reason"].startswith("pre-load gate")
    assert not list((tmp_path / "evals").glob("*.json")) if (tmp_path / "evals").exists() else True   # no fabricated rows
