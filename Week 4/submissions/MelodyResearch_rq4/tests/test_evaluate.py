import json

import pytest
import torch

from rq4 import config as C
from rq4 import data, store
from rq4 import evaluate as E
from rq4 import parsing
from rq4 import preflight as P
from rq4.directions import unit
from rq4.tokens import resolve_target_ids
from toy import ToyBackend

FP = "testfp"


def make_ctx(backend, prompt_id_suffix="multiple_choice", item=0):
    plan = [p for p in data.plan_prompts(data.load_items()) if p["prompt_format"] == prompt_id_suffix][item]
    ids, pos = E.source_position(backend, plan)
    targets = resolve_target_ids(P._Shim(backend), plan["stereotype_target_text"], plan["counter_target_text"], backend.mask)
    return E.PromptContext(plan, ids, pos, targets, backend.capture(ids, list(C.LAYERS), pos)), plan


class SmallCosBank(E.DirectionBank):
    """Target direction with |cos| ~ 0.05 to the clean activation so ablation stays under the 10% cap."""

    def __init__(self, backend, sham_ids, ctx):
        super().__init__(backend, sham_ids)
        self.ctx = ctx

    def target(self, ids, layer):
        h = self.ctx.clean_h[layer]
        hn = h / h.norm()
        v = torch.randn(h.shape[0], generator=torch.Generator().manual_seed(layer))
        v = v - (v @ hn) * hn
        return unit(v / v.norm() + 0.05 * hn)


def sham_ids(backend):
    return resolve_target_ids(P._Shim(backend), "banana", "table", backend.mask)


def run(backend, ctx, bank, todo=None, guard=lambda: None):
    saved = []
    plan = ctx.plan
    todo = todo or {store.eval_key(plan["prompt_id"], a, l) for _, a, l in [(plan, C.CLEAN, None)] + [(plan, a, l) for l in C.LAYERS for a in C.ARMS_INTERVENTION]}
    n = E.evaluate_prompt(backend, ctx, bank, todo, "1", FP, saved.append, guard)
    return saved, n


def test_full_prompt_evaluation_schema_controls_and_displacement():
    b = ToyBackend()
    ctx, plan = make_ctx(b)
    rows, n = run(b, ctx, SmallCosBank(b, sham_ids(b), ctx))
    assert n == len(rows) == 16
    assert len({r["key"] for r in rows}) == 16 and all(store.validate_row(r) == [] for r in rows)
    json.dumps(rows, allow_nan=False)                                   # fully serialisable and finite
    clean = next(r for r in rows if r["arm"] == "clean")
    assert "measured_displacement" not in clean and clean["layer"] is None
    iv = [r for r in rows if r["arm"] != "clean"]
    assert {r["status"] for r in iv} == {"ok"} and {r["arm"] for r in iv} == set(C.ARMS_INTERVENTION)
    for r in iv:
        assert r["measured_ratio"] <= C.DISPLACEMENT_CAP * 1.001, r["key"]
        assert [h["fired"] for h in r["hook_calls"]] == [1] * 5            # 3 teacher-forced + readouts + generate
        assert r["generation_raw"] != "" and r["category"] in parsing.CATEGORIES
        assert set(r["readouts"]) == {"8", "16", "22", "model_output"}
        assert r["target_ids"]["stereotype_ids"] and r["target_ids"]["counter_ids"]
        assert r["calibration_ratio"] == pytest.approx(1.0, rel=1e-3)
        if r["arm"] != "target":                                          # norm-matched to the target's displacement
            assert r["calibration_ratio"] == pytest.approx(1.0, rel=1e-3)
            assert r["measured_displacement"] == pytest.approx(r["intended_displacement"], rel=1e-3)
            assert r["direction_kind"] == "norm_matched_displacement" and abs(r["cos_with_target"]) < 0.99
        else:
            assert r["direction_kind"].endswith("ablation")
    for r in rows:
        m = r["tf_margin_stereo_minus_counter"]
        assert m is not None and abs(m) < 1e6
        assert set(r["teacher_forced"]) == {"stereotype", "counter", "unknown"}
        assert r["teacher_forced"]["stereotype"]["text"] in (" (a)", " (b)", " (c)")
    assert all(len(m.layers[0]._forward_hooks) == 0 for m in [b.model]) and all(len(l._forward_hooks) == 0 for l in b.model.layers)


def test_open_ended_scores_actual_answer_phrases():
    b = ToyBackend()
    ctx, plan = make_ctx(b, "open_ended")
    rows, _ = run(b, ctx, SmallCosBank(b, sham_ids(b), ctx))
    clean = next(r for r in rows if r["arm"] == "clean")
    assert clean["teacher_forced"]["stereotype"]["text"] == " " + plan["stereotype_answer_text"]
    assert len(clean["teacher_forced"]["stereotype"]["token_ids"]) == len(plan["stereotype_answer_text"].split())     # full phrase


def test_seed_determinism_end_to_end():
    outs = []
    for _ in range(2):
        b = ToyBackend()
        ctx, _ = make_ctx(b)
        rows, _ = run(b, ctx, SmallCosBank(b, sham_ids(b), ctx))
        outs.append({r["key"]: (r["tf_margin_stereo_minus_counter"], r.get("measured_displacement"), r.get("cos_with_target")) for r in rows})
    assert outs[0] == outs[1]
    assert len({outs[0][k][0] for k in outs[0] if "random_s" in k and k.endswith("L8")}) == 3     # seeds give different controls


def test_over_cap_target_is_safety_excluded_without_inference():
    b = ToyBackend()
    ctx, _ = make_ctx(b)

    class Aligned(E.DirectionBank):
        def target(self, ids, layer):
            return ctx.clean_h[layer] / ctx.clean_h[layer].norm()          # removes 100% of the activation

    calls = {"n": 0}
    orig = b.logits
    b.logits = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), orig(*a, **k))[1]
    rows, _ = run(b, ctx, Aligned(b, sham_ids(b)), todo={k for k in {store.eval_key(ctx.plan["prompt_id"], a, l) for l in C.LAYERS for a in C.ARMS_INTERVENTION}})
    assert len(rows) == 15 and {r["status"] for r in rows} == {"safety_excluded"}
    assert calls["n"] == 0 and all("generation_raw" not in r for r in rows)
    assert all(r["target_ratio"] > C.DISPLACEMENT_CAP for r in rows)


def test_degenerate_control_recorded_not_replaced():
    b = ToyBackend()
    ctx, _ = make_ctx(b)

    class BadSham(SmallCosBank):
        def sham(self, layer):
            return self.target(None, layer)                                 # collinear with target

    rows, _ = run(b, ctx, BadSham(b, sham_ids(b), ctx))
    sham = [r for r in rows if r["arm"] == "sham"]
    assert len(sham) == 3 and {r["status"] for r in sham} == {"control_degenerate"} and "collinear" in sham[0]["status_detail"]
    assert {r["status"] for r in rows if r["arm"] in ("target", "random_s11")} == {"ok"}


def test_resume_only_runs_missing_keys_and_exceptions_leave_no_partial_row():
    b = ToyBackend()
    ctx, plan = make_ctx(b)
    bank = SmallCosBank(b, sham_ids(b), ctx)
    one = {store.eval_key(plan["prompt_id"], "target", 16)}
    rows, n = run(b, ctx, bank, todo=one)
    assert n == 1 and rows[0]["key"] in one
    orig = b.generate

    def boom(ids, edit, max_new):
        if edit is not None and edit.layer == 22:
            raise RuntimeError("generation failed")
        return orig(ids, edit, max_new)

    b.generate = boom
    saved = []
    with pytest.raises(RuntimeError):
        E.evaluate_prompt(b, ctx, bank, {store.eval_key(plan["prompt_id"], "target", 22)}, "1", FP, saved.append)
    assert saved == [] and all(len(l._forward_hooks) == 0 for l in b.model.layers)


def test_nonfinite_model_output_stops():
    b = ToyBackend()
    ctx, plan = make_ctx(b)
    orig = b.readouts

    def bad(prompt, ids, edit):
        r, ml, rec = orig(prompt, ids, edit)
        ml = ml.clone(); ml[3] = float("nan")
        return r, ml, rec

    b.readouts = bad
    with pytest.raises(FloatingPointError):
        E.evaluate_prompt(b, ctx, SmallCosBank(b, sham_ids(b), ctx), {store.eval_key(plan["prompt_id"], "clean", None)}, "1", FP, lambda r: None)


def test_hook_that_never_fires_is_an_error():
    b = ToyBackend()
    ctx, plan = make_ctx(b)
    orig = b.logits
    b.logits = lambda ids, edit: (orig(ids, None)[0], None)                 # silently drops the intervention
    with pytest.raises(RuntimeError, match="hook alignment"):
        E.evaluate_prompt(b, ctx, SmallCosBank(b, sham_ids(b), ctx), {store.eval_key(plan["prompt_id"], "target", 8)}, "1", FP, lambda r: None)


def test_source_position_requires_colon():
    b = ToyBackend()
    plan = dict(data.plan_prompts(data.load_items())[0])
    plan["prompt"] = plan["prompt"] + " extra"
    with pytest.raises(AssertionError):
        E.source_position(b, plan)


def test_ineligible_rows_cover_all_keys():
    plan = data.plan_prompts(data.load_items())[0]
    rows = list(E.ineligible_rows(plan, "no meaningful token", "1", FP))
    assert len(rows) == 16 and all(store.validate_row(r) == [] for r in rows) and {r["status"] for r in rows} == {"ineligible_pair"}


def test_position_check_against_week3_reference_reports_mismatch():
    b = ToyBackend()
    plan = data.plan_prompts(data.load_items(), C.STAGE1_BASE_ITEMS)
    ref = data.week3_reference_positions()
    with pytest.raises(AssertionError):                                     # toy tokenizer lengths != Qwen's saved seq_len
        P.check_positions(b, plan, ref)


def _breach_at(layer, arm_index):
    """run_condition wrapper: the arm_index-th intervention evaluated at `layer` breaches the cap when MEASURED."""
    orig, seen = E.run_condition, {"n": 0}

    def wrapped(backend, ctx, edit):
        if edit is not None and edit.layer == layer:
            seen["n"] += 1
            if seen["n"] == arm_index:
                raise E.DisplacementCapError("displacement ratio 0.10400 exceeds cap 0.1", 0.104)
        return orig(backend, ctx, edit)
    return wrapped


def test_measured_cap_breach_excludes_whole_cell_not_one_arm(monkeypatch):
    b = ToyBackend()
    ctx, plan = make_ctx(b)
    bank = SmallCosBank(b, sham_ids(b), ctx)
    monkeypatch.setattr(E, "run_condition", _breach_at(16, 3))          # third arm (random_s23) breaches after target/s11 ran
    rows, _ = run(b, ctx, bank)
    cell = [r for r in rows if r["layer"] == 16]
    assert len(cell) == 5 and {r["status"] for r in cell} == {"safety_excluded"}
    assert {r["arm"] for r in cell} == set(C.ARMS_INTERVENTION)
    for r in cell:
        assert "generation_raw" not in r and store.validate_row(r) == []
        assert r["cell_exclusion"] == "measured_cap" and r["cell_exclusion_arm"] == "random_s23"
        assert "0.10400" in r["status_detail"] and r["measured_cap_ratio"] == pytest.approx(0.104)
        assert r["target_ratio"] <= C.DISPLACEMENT_CAP and r["intended_ratio"] == r["target_ratio"]
        assert r["intended_displacement"] == r["target_displacement"]
    others = [r for r in rows if r["arm"] != "clean" and r["layer"] != 16]
    assert len(others) == 10 and {r["status"] for r in others} == {"ok"}   # other layers unaffected
    json.dumps(rows, allow_nan=False)


def test_measured_cap_breach_on_resume_excludes_only_remaining_arms(monkeypatch):
    b = ToyBackend()
    ctx, plan = make_ctx(b)
    pid = plan["prompt_id"]
    todo = {store.eval_key(pid, a, 8) for a in C.ARMS_INTERVENTION if a != C.ARM_TARGET}   # target already saved earlier
    monkeypatch.setattr(E, "run_condition", _breach_at(8, 2))
    rows, n = run(b, ctx, SmallCosBank(b, sham_ids(b), ctx), todo=todo)
    assert n == len(rows) == 4 and {r["status"] for r in rows} == {"safety_excluded"}
    assert {r["key"] for r in rows} == todo                              # never re-emits (or overwrites) the saved target row


class _StopGuard(Exception):
    pass


def test_guard_runs_after_each_arm_and_stops_before_next_arm():
    b = ToyBackend()
    ctx, plan = make_ctx(b)
    calls = {"guard": 0, "run": 0}
    orig = E.run_condition

    def counting(*a, **k):
        calls["run"] += 1
        return orig(*a, **k)

    def guard():
        # the guard must see each run_condition immediately: guard calls == run_condition calls
        assert calls["guard"] < calls["run"]
        calls["guard"] += 1
        if calls["guard"] == 2:
            raise _StopGuard()

    todo = {store.eval_key(plan["prompt_id"], a, C.LAYERS[0]) for a in C.ARMS_INTERVENTION}
    saved = []
    E.run_condition = counting
    try:
        with pytest.raises(_StopGuard):
            E.evaluate_prompt(b, ctx, SmallCosBank(b, sham_ids(b), ctx), todo, "1", FP, saved.append, guard)
    finally:
        E.run_condition = orig
    assert calls["run"] == 2 and calls["guard"] == 2      # no third arm ran
    assert saved == []                                     # buffered cell is discarded, nothing half-saved


def test_guard_not_duplicated_for_buffered_rows():
    b = ToyBackend()
    ctx, plan = make_ctx(b)
    n_guard = [0]
    rows, n = run(b, ctx, SmallCosBank(b, sham_ids(b), ctx), guard=lambda: n_guard.__setitem__(0, n_guard[0] + 1))
    assert n == 16 and n_guard[0] == 16                    # exactly one reading per row/forward pass
