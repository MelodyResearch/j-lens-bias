import json

import pytest

from rq4 import config as C
from rq4 import store
from rq4 import stage1_validator as V
from synth_pilot import FP, make_rows, memory_records, write_pilot


def validate(root):
    return V.validate_stage1(root, fingerprint=FP)


def failed(rep):
    return {c["check"] for c in rep["checks"] if not c["passed"]}


def test_complete_synthetic_pilot_passes(tmp_path):
    rep = validate(write_pilot(tmp_path))
    assert rep["passed"], failed(rep)
    assert rep["n_rows"] == 128 and rep["outcome_blind"]


def test_zero_rows_never_passes(tmp_path):
    rep = validate(write_pilot(tmp_path, rows=[]))
    assert not rep["passed"] and rep["n_rows"] == 0                      # 128 absent files are not a pass; nothing is invented
    assert not (tmp_path / "evals").exists() or not list((tmp_path / "evals").glob("*.json"))


def test_128_files_alone_are_not_enough(tmp_path):
    rows = make_rows()
    for r in rows:                                                       # 128 valid files, but nothing ever fired
        for c in r.get("hook_calls", []):
            if r["arm"] != C.CLEAN:
                c["fired"] = 0
    rep = validate(write_pilot(tmp_path, rows=rows))
    assert len(list((tmp_path / "evals").glob("*.json"))) == 128 and not rep["passed"]
    assert "H hook firing/site/position" in failed(rep)


@pytest.mark.parametrize("name,mut,check", [
    ("empty generation", lambda r: r.update(generation_raw="  ") if r["arm"] == "sham" and r["layer"] == 8 else None, "C raw nonempty generations"),
    ("wrong layer", lambda r: r["hook_calls"][-1].update(layer=r["layer"] + 1) if r["arm"] == "target" and r["layer"] == 16 else None, "H hook firing/site/position"),
    ("wrong position", lambda r: r["hook_calls"][-1].update(position=3) if r["arm"] == "target" and r["layer"] == 22 else None, "H hook firing/site/position"),
    ("fired twice", lambda r: r["hook_calls"][-1].update(fired=2) if r["arm"] == "random_s11" else None, "H hook firing/site/position"),
    ("over cap", lambda r: r["hook_calls"][-1].update(measured=10.5) if r["arm"] == "target" and r["layer"] == 8 else None, "D measured displacement <= cap"),
    ("control not norm matched", lambda r: r.update(measured_displacement=9.0) if r["arm"] == "random_s23" else None, "N norm-matched controls"),
    ("control intended differs", lambda r: r.update(intended_displacement=6.0) if r["arm"] == "sham" else None, "N norm-matched controls"),
    ("control collinear", lambda r: r.update(cos_with_target=0.995) if r["arm"] == "random_s47" else None, "N norm-matched controls"),
    ("control wrong kind", lambda r: r.update(direction_kind="token_set_contrast_J_lens_ablation") if r["arm"] == "sham" else None, "N norm-matched controls"),
])
def test_row_level_failures_fail_the_gate(tmp_path, name, mut, check):
    rep = validate(write_pilot(tmp_path, rows=make_rows(mutate=mut)))
    assert not rep["passed"] and check in failed(rep), (name, failed(rep))


def test_bad_statuses_fail(tmp_path):
    for st in ("error", "control_degenerate", "ineligible_pair"):
        d = tmp_path / st
        rep = validate(write_pilot(d, rows=make_rows(mutate=lambda r, st=st: r.update(status=st) if r["arm"] == "sham" and r["layer"] == 8 and r["prompt_id"].endswith("open_ended") else None)))
        assert not rep["passed"], st


def test_whole_safety_excluded_cell_is_recorded_faithfully_and_allowed(tmp_path):
    pid = make_rows()[0]["prompt_id"]
    rep = validate(write_pilot(tmp_path, rows=make_rows(excluded={(pid, 16)})))
    assert rep["passed"], failed(rep)
    assert rep["status_counts"]["safety_excluded"] == 5
    assert any("safety-excluded cells" in c["detail"] and pid in c["detail"] for c in rep["checks"])


def _measured_cap_cell(pid, layer, marked=True):
    def m(r):
        if r["prompt_id"] == pid and r["layer"] == layer and r["arm"] != C.CLEAN:
            r.update(status="safety_excluded", status_detail="measured displacement cap breached by arm sham: displacement ratio 0.104 exceeds cap 0.1",
                     target_ratio=0.08, intended_ratio=0.08, intended_displacement=8.0, measured_cap_ratio=0.104)
            if marked:
                r.update(cell_exclusion="measured_cap", cell_exclusion_arm="sham")
            for k in ("measured_displacement", "hook_calls", "generation_raw", "direction_kind", "cos_with_target", "category"):
                r.pop(k, None)
    return m


def test_whole_cell_exclusion_from_documented_measured_cap_is_allowed(tmp_path):
    pid = make_rows()[0]["prompt_id"]
    rep = validate(write_pilot(tmp_path / "a", rows=make_rows(mutate=_measured_cap_cell(pid, 16))))
    assert rep["passed"], failed(rep)
    assert any("measured cap" in c["detail"] for c in rep["checks"])
    rep = validate(write_pilot(tmp_path / "b", rows=make_rows(mutate=_measured_cap_cell(pid, 16, marked=False))))
    assert not rep["passed"] and "S cell statuses & coverage" in failed(rep)      # undocumented: ratio <= cap and no marker


def test_norm_match_tolerance_is_ten_percent(tmp_path):
    from rq4 import stage1_validator as V
    assert V.NORM_MATCH_RTOL == 0.10
    for dev, expect in ((1.08, True), (1.12, False)):
        rows = make_rows(mutate=lambda r, dev=dev: r.update(measured_displacement=5.0 * dev) if r["arm"] == "sham" and r["layer"] == 8 and r["prompt_id"].endswith("open_ended") else None)
        assert validate(write_pilot(tmp_path / str(dev), rows=rows))["passed"] is expect, dev


def test_provenance_must_pin_model_and_lens_revisions(tmp_path):
    root = write_pilot(tmp_path)
    pv = json.loads((root / "runtime_provenance.json").read_text())
    pv["pinned"]["lens_revision"] = "qwen-n1000"                          # floating tag instead of the commit
    (root / "runtime_provenance.json").write_text(json.dumps(pv))
    assert not validate(root)["passed"]


def test_partial_or_all_excluded_is_not_a_pass(tmp_path):
    rows = make_rows()
    pid = rows[0]["prompt_id"]
    rep = validate(write_pilot(tmp_path / "a", rows=make_rows(excluded={(pid, l) for l in C.LAYERS})))
    assert not rep["passed"]                                             # prompt with no layer cell that ran
    rep = validate(write_pilot(tmp_path / "b", rows=make_rows(mutate=lambda r: r.update(status="safety_excluded", target_ratio=0.3) if r["arm"] == "sham" and r["layer"] == 8 else None)))
    assert not rep["passed"]                                             # mixed cell


def test_missing_duplicate_wrong_fingerprint_and_tmp(tmp_path):
    rows = make_rows()
    rep = validate(write_pilot(tmp_path / "m", rows=rows[:-1]))
    assert not rep["passed"] and "K 128 unique valid checkpoints" in failed(rep)
    root = write_pilot(tmp_path / "f", rows=make_rows(fp="other"))
    assert not validate(root)["passed"]
    root = write_pilot(tmp_path / "t")
    (root / "evals" / "x.json.tmp").write_text("{}")
    assert not validate(root)["passed"]
    root = write_pilot(tmp_path / "h", rows=make_rows(mutate=lambda r: r.update(prompt_hash="0" * 64)))
    assert not validate(root)["passed"]


def test_nonfinite_checkpoint_is_rejected(tmp_path):
    root = write_pilot(tmp_path)
    path = store.row_path(root / "evals", make_rows()[0]["key"])
    path.write_text(path.read_text().replace('"seq_len": 120', '"seq_len": 120, "bad": NaN'))
    assert not validate(root)["passed"]


def test_missing_preflight_or_provenance_fails(tmp_path):
    assert "R runtime preflight" in failed(validate(write_pilot(tmp_path / "a", preflight=False)))
    assert "P runtime provenance" in failed(validate(write_pilot(tmp_path / "b", provenance=False)))


def test_verdict_is_outcome_blind(tmp_path):
    a = validate(write_pilot(tmp_path / "a"))
    mutate = lambda r: r.update(category="stereotype", tf_margin_stereo_minus_counter=9e9, follows_evidence=False)
    b = validate(write_pilot(tmp_path / "b", rows=make_rows(mutate=mutate)))
    assert a["passed"] and b["passed"]
    assert [c["passed"] for c in a["checks"]] == [c["passed"] for c in b["checks"]]


# ---- memory log ----------------------------------------------------------------------------------------
def run_mem(tmp_path, recs, n=128):
    p = tmp_path / "m.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    return V.check_memory_log(p, n)


def test_memory_log_pass_and_failure_modes(tmp_path):
    assert run_mem(tmp_path, memory_records())[0]
    assert not V.check_memory_log(tmp_path / "absent.jsonl", 128)[0]
    r = memory_records(); r[1]["ts"] = r[0]["ts"] + 5                       # readings not 25 s apart
    assert not run_mem(tmp_path, r)[0]
    r = memory_records(); r[2]["free_pct"] = 59                              # one pre-load reading under 60 %
    assert not run_mem(tmp_path, r)[0]
    r = memory_records(); r[-1]["swap_used_mb"] = 5000.0                     # swap growth during the run
    assert not run_mem(tmp_path, r)[0]
    r = memory_records(); r[-1]["gpu_alloc_mb"] = 1700 + C.GPU_DRIFT_CEIL_MB + 5
    assert not run_mem(tmp_path, r)[0]
    r = memory_records(); r[-1]["gpu_free_mb"] = None
    assert not run_mem(tmp_path, r)[0]
    assert not run_mem(tmp_path, memory_records(n_rows=10))[0]               # too few after_eval readings
    assert not run_mem(tmp_path, [x for x in memory_records() if x["tag"] != "post_model_load"])[0]
    assert not run_mem(tmp_path, [x for x in memory_records() if x["tag"] != "preload_reading_2"])[0]


def test_session_refused_by_preload_gate_is_reported_not_failed(tmp_path):
    refused = [{"tag": "preload_reading_1", "ts": 1.0, "free_pct": 48.0, "available_mb": 1.0, "swap_used_mb": 1.0, "gpu_free_mb": 1.0}]
    ok, detail = run_mem(tmp_path, refused + memory_records(t0=5000.0))
    assert ok and "1 refused" in detail
    assert not run_mem(tmp_path, refused)[0]                                 # nothing ever loaded
