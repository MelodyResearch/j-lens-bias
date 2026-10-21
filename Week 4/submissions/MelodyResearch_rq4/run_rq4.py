#!/usr/bin/env python3
"""RQ4 runner. DEFAULT = preparation only: no torch/transformers/jlens import, no model load.

REMOTE-ONLY INFERENCE: --run-inference is refused on macOS (any flags) and needs a Linux CUDA VM with
RQ4_RESULTS_DIR (absolute, persistent) and RQ4_CODE_COMMIT (exact 40-hex SHA of the checkout) set.
See REMOTE_EXECUTION_ADDENDUM.md and notebooks/rq4_colab.ipynb.

  python run_rq4.py                      # prepare: validate inputs, write manifests + plan, print counts
  python run_rq4.py --analyze            # tables + figures from saved results only (no model)
  python run_rq4.py --combine            # rebuild results/rq4_raw_results.jsonl from checkpoints
  python run_rq4.py --run-inference --i-confirm-no-heavy-experiment --stage 1
  python run_rq4.py --validate-stage1    # outcome-blind objective pilot gate (read-only; writes stage1_validation.json)
  python run_rq4.py --approve-stage1 "reviewed raw generations: ..." --reviewer NAME   # needs validator PASS
  python run_rq4.py --run-inference --i-confirm-no-heavy-experiment --stage 2

Inference needs BOTH flags. Stage 2 additionally needs $RQ4_RESULTS_DIR/STAGE1_APPROVED and a fresh validator PASS.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rq4 import config as C  # noqa: E402
from rq4 import data, manifest, remote, store  # noqa: E402


def prepare(stage_items=None) -> list[dict]:
    rows = data.load_items()
    plan = data.plan_prompts(rows, stage_items)
    em = manifest.eligibility_manifest(rows)
    if em["static_failures"]:
        raise SystemExit(f"STATIC VALIDATION FAILED for: {em['static_failures']}")
    paths = manifest.write_manifests()
    full = data.plan_prompts(rows)
    keys = store.planned_keys(full)
    plan_path = C.MANIFEST_DIR / "planned_evaluations.jsonl"
    with open(plan_path, "w") as f:
        for p, arm, layer in keys:
            f.write(json.dumps({"key": store.eval_key(p["prompt_id"], arm, layer), "prompt_hash": p["prompt_hash"],
                                "arm": arm, "layer": layer}) + "\n")
    if C.protocol_sha256() != C.FROZEN_PROTOCOL_SHA256:
        print("WARNING: FROZEN_PROTOCOL.md hash differs from the implemented protocol hash.", file=sys.stderr)
    print(f"PREPARED: {len(full)} prompts ({len(full)//2} matched pairs, {len(C.BASE_ITEM_IDS)} base items), "
          f"{len(keys)} planned evaluations; config fingerprint {C.config_fingerprint()}.")
    for n, p in paths.items():
        print(f"  wrote {p.relative_to(C.ROOT)}")
    print(f"  wrote {plan_path.relative_to(C.ROOT)}")
    return plan


class ResourceStop(RuntimeError):
    pass


def run_inference(stage: int) -> None:
    remote.refuse_inference_on_this_platform()            # first: nothing below runs on macOS
    try:
        remote.require_remote_config()                    # Linux, RQ4_RESULTS_DIR persistent+writable, RQ4_CODE_COMMIT == HEAD
    except remote.RemoteGuardError as exc:
        raise SystemExit(f"REFUSING remote inference: {exc}")
    from rq4 import resources
    if C.protocol_sha256() != C.FROZEN_PROTOCOL_SHA256:
        raise SystemExit("REFUSING: FROZEN_PROTOCOL.md changed since implementation (needs dated amendment).")
    items = C.STAGE1_BASE_ITEMS if stage == 1 else C.STAGE2_BASE_ITEMS
    if stage == 2:
        _require_stage2_gate()
    plan = prepare(items)
    fp = C.config_fingerprint()
    resources.refuse_if_duplicate()
    resources.acquire_lock()
    state = {"stage": stage, "fingerprint": fp, "label": "PILOT_ONLY" if stage == 1 else "FULL_RUN_IN_PROGRESS",
             "started": time.time(), "stopped_reason": None}
    try:
        _run(plan, stage, fp, state)
    finally:
        C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        C.RUN_STATE.write_text(json.dumps(state, indent=2))
        store.combine(C.EVAL_DIR, C.RAW_JSONL, fp)
        resources.release_lock()


def _run(plan, stage, fp, state) -> None:
    from rq4 import resources
    done = store.load_valid_rows(C.EVAL_DIR, fp)
    wanted = {store.eval_key(p["prompt_id"], a, l) for p, a, l in store.planned_keys(plan)}
    todo = wanted - set(done)
    print(f"Stage {stage}: {len(wanted)} planned evaluations, {len(set(done) & wanted)} done, {len(todo)} pending.")
    if not todo:
        print("Nothing to do.")
        return
    ok, why = resources.preload_gate(log_fn=resources.log_snapshot)
    if not ok:
        state["stopped_reason"] = f"pre-load gate: {why}"
        raise SystemExit(f"MEMORY SAFETY STOP before model load: {why}")
    pre = resources.snapshot()
    resources.log_snapshot("pre_model_load", pre)
    code_commit = remote.validate_code_commit(os.environ.get(C.ENV_CODE_COMMIT), remote.git_head())

    from rq4 import evaluate as E, preflight as P, runtime
    from rq4.hooks import HookAlignmentError  # noqa: F401
    import torch

    backend = runtime.QwenBackend()
    post = resources.snapshot(torch)
    resources.log_snapshot("post_model_load", post)
    baseline = resources.post_load_baseline(pre, post)
    ok, why = resources.is_safe(post, baseline)
    if not ok:
        state["stopped_reason"] = f"post-load: {why}"
        raise SystemExit(f"MEMORY SAFETY STOP after load: {why}")
    prov = {**backend.provenance(), "code_commit": code_commit, "config_fingerprint": fp}
    try:
        remote.record_provenance(prov)                     # refuses a dtype/library/commit change between sessions
    except remote.RemoteGuardError as exc:
        state["stopped_reason"] = f"provenance: {exc}"
        raise SystemExit(f"REFUSING to resume: {exc}")
    print(f"runtime: {prov['gpu_name']} dtype={prov['dtype']} torch={prov['torch']} transformers={prov['transformers']} "
          f"jlens={prov['jlens']} commit={code_commit[:12]}")
    stamp = {k: prov[k] for k in ("code_commit", "dtype", "gpu_name", "torch", "transformers", "jlens", "cuda")}

    # ---- outcome-blind preflight (tokenization + one clean forward) ----
    P.check_model_and_lens(backend.lens_info())
    w3ref = data.week3_reference_positions()
    pos_report = P.check_positions(backend, plan, w3ref)
    pairs, sham = P.resolve_pairs(backend, plan, backend.meaningful_mask())
    boundaries = P.candidate_boundary_report(backend, plan)
    align = P.check_alignment(backend, plan[0])
    pre_json = {"alignment": align, "positions": pos_report, "candidate_boundaries": boundaries,
                "sham_ids": sham.as_log(),
                "pairs": {k: (v.as_log() if hasattr(v, "as_log") else {"ineligible": v}) for k, v in pairs.items()}}
    (C.RESULTS_DIR / "runtime_preflight.json").write_text(json.dumps(pre_json, indent=2))
    P.assert_boundaries(boundaries)         # report is already on disk; fail before the first outcome
    print(f"PREFLIGHT OK: alignment {align['formula']} rel_err={align['rel_err']:.4g}; "
          f"ineligible pairs: {[k for k, v in pairs.items() if isinstance(v, str)]}")

    def save(row):
        store.save_row(C.EVAL_DIR, {**row, "runtime": stamp})

    def guard():
        snap = resources.snapshot(torch)
        resources.log_snapshot("after_eval", snap)
        ok, why = resources.is_safe(snap, baseline)
        if not ok:
            raise ResourceStop(why)

    dirs = E.DirectionBank(backend, sham)
    try:
        for p in plan:
            mine = {k for k in todo if k.startswith(p["prompt_id"] + "__")}
            if not mine:
                continue
            tgt = pairs[p["pair_id"]]
            if isinstance(tgt, str):
                for row in E.ineligible_rows(p, tgt, str(stage), fp):
                    if row["key"] in mine:
                        save(row)
                continue
            ids, pos = E.source_position(backend, p)
            clean_h = backend.capture(ids, list(C.LAYERS), pos)
            ctx = E.PromptContext(p, ids, pos, tgt, clean_h)
            t0 = time.time()
            n = E.evaluate_prompt(backend, ctx, dirs, mine, str(stage), fp, save, guard)
            print(f"{p['prompt_id']}: {n} evaluations in {time.time() - t0:.1f}s", file=sys.stderr)
            backend.release()
    except ResourceStop as exc:
        state["stopped_reason"] = f"resource guard: {exc}"
        print(f"MEMORY SAFETY STOP: {exc}")
    except Exception as exc:  # noqa: BLE001  (non-finite values, hook misalignment, ... -> stop, keep checkpoints)
        state["stopped_reason"] = f"{type(exc).__name__}: {exc}"
        print(f"STOPPING on error: {state['stopped_reason']}", file=sys.stderr)
        raise


def _validation_report() -> dict:
    from rq4 import stage1_validator as V
    return V.validate_stage1()


def validate_stage1() -> bool:
    from rq4 import stage1_validator as V
    rep = _validation_report()
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    V.write_report(rep, C.RESULTS_DIR / "stage1_validation.json")
    print(f"STAGE 1 VALIDATION: {'PASS' if rep['passed'] else 'FAIL'} ({rep['n_rows']} rows, statuses {rep['status_counts']})")
    for c in rep["checks"]:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['check']}: {c['detail']}")
    print(rep["note"])
    return rep["passed"]


def approve_stage1(note: str, reviewer: str | None) -> None:
    """Records a HUMAN/Codex review acknowledgement. Requires the objective validator to PASS first."""
    if len((note or "").strip()) < 20 or not (reviewer or "").strip():
        raise SystemExit("Approval needs --reviewer NAME and a >=20-character note stating which raw generations were read.")
    rep = _validation_report()
    if not rep["passed"]:
        raise SystemExit("Not approving: the outcome-blind Stage 1 validator did not PASS (run --validate-stage1).")
    C.STAGE1_APPROVAL.write_text(json.dumps({"fingerprint": C.config_fingerprint(), "note": note, "reviewer": reviewer,
                                             "rows_digest": rep["rows_digest"], "ts": time.time()}, indent=2))
    print("Recorded Stage 1 approval (validator PASS + review acknowledgement).")


def _require_stage2_gate() -> None:
    if not C.STAGE1_APPROVAL.exists():
        raise SystemExit("REFUSING Stage 2: STAGE1_APPROVED missing (validator PASS + review, then --approve-stage1).")
    appr = json.loads(C.STAGE1_APPROVAL.read_text())
    if appr.get("fingerprint") != C.config_fingerprint():
        raise SystemExit("REFUSING Stage 2: Stage 1 approval was recorded under a different config fingerprint/code commit.")
    rep = _validation_report()
    if not rep["passed"]:
        raise SystemExit("REFUSING Stage 2: the Stage 1 validator no longer passes.")
    if rep["rows_digest"] != appr.get("rows_digest"):
        raise SystemExit("REFUSING Stage 2: Stage 1 checkpoints changed after the recorded approval.")


def analyze() -> None:
    from rq4 import analysis as A, figures
    n = store.combine(C.EVAL_DIR, C.RAW_JSONL, C.config_fingerprint())
    df = A.load_results(C.RAW_JSONL)
    print(f"{n} rows; gate: {A.completion_status(df)}")
    if df.empty:
        return
    pairs = A.per_pair_effects(df)
    items = A.per_item(pairs)
    out = C.RESULTS_DIR / "tables"
    out.mkdir(parents=True, exist_ok=True)
    pairs.drop(columns=[c for c in pairs if c.endswith("_seeds")]).to_csv(out / "per_pair.csv", index=False)
    items.to_csv(out / "per_item.csv", index=False)
    A.summarize(items).to_csv(out / "summary_by_layer.csv", index=False)
    A.category_rates(df).to_csv(out / "category_rates.csv", index=False)
    A.displacement_summary(df).to_csv(out / "displacement.csv", index=False)
    print("figures:", [str(p) for p in figures.make_figures()])
    print(f"tables: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepare-only", action="store_true", help="default mode; accepted for explicitness")
    ap.add_argument("--run-inference", action="store_true")
    ap.add_argument("--i-confirm-no-heavy-experiment", action="store_true")
    ap.add_argument("--stage", type=int, choices=(1, 2), default=1)
    ap.add_argument("--approve-stage1", metavar="NOTE")
    ap.add_argument("--reviewer", metavar="NAME")
    ap.add_argument("--validate-stage1", action="store_true")
    ap.add_argument("--show-stage1-generations", action="store_true", help="print saved Stage 1 raw generations (read-only)")
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.run_inference:                         # before anything else, regardless of other flags
        remote.refuse_inference_on_this_platform()
    if a.show_stage1_generations:
        from rq4 import stage1_validator as V
        return print(V.format_generations())
    if a.validate_stage1:
        raise SystemExit(0 if validate_stage1() else 1)
    if a.approve_stage1:
        return approve_stage1(a.approve_stage1, a.reviewer)
    if a.combine:
        print(store.combine(C.EVAL_DIR, C.RAW_JSONL, C.config_fingerprint()), "rows combined")
        return
    if a.analyze:
        return analyze()
    if a.run_inference and a.i_confirm_no_heavy_experiment:
        return run_inference(a.stage)
    if a.run_inference or a.i_confirm_no_heavy_experiment:
        print("NOTE: BOTH --run-inference and --i-confirm-no-heavy-experiment are required; staying in prepare-only mode.", file=sys.stderr)
    prepare()
    print("PREPARE-ONLY: no model import or load.")


if __name__ == "__main__":
    main()
