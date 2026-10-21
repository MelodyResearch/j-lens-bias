"""Synthetic Stage 1 pilot directory (structure only; NOT data, no model). Used by validator/gate tests."""

import json
from pathlib import Path

from rq4 import config as C
from rq4 import data, store

FP = "synthfp0000000000"[:16]
SEQ = 120


def _hooks(layer, pos, measured, norm, fired=1):
    ops = ["tf:stereotype", "tf:counter", "tf:unknown", "readouts", "generate"]
    return [{"op": o, "fired": fired, "layer": layer, "position": pos, "dtype": "torch.bfloat16",
             "seq_len": SEQ if o in ("readouts", "generate") else SEQ + 3, "activation_norm": norm,
             "intended": measured, "measured": measured} for o in ops]


def make_rows(fp=FP, excluded=(), mutate=None):
    plan = data.plan_prompts(data.load_items(), C.STAGE1_BASE_ITEMS)
    rows = []
    for p, arm, layer in store.planned_keys(plan):
        base = {"schema_version": C.SCHEMA_VERSION, "key": store.eval_key(p["prompt_id"], arm, layer),
                "config_fingerprint": fp, "stage": "1", "base_item_id": p["base_item_id"], "pair_id": p["pair_id"],
                "prompt_id": p["prompt_id"], "context_condition": p["context_condition"],
                "question_polarity": p["question_polarity"], "prompt_format": p["prompt_format"],
                "prompt_hash": p["prompt_hash"], "arm": arm, "layer": layer, "status": "ok", "status_detail": None,
                "seq_len": SEQ, "position": SEQ - 1, "category": "counter", "generation_raw": " (b) The answer."}
        norm, d_t = 100.0, 5.0
        if arm == C.CLEAN:
            base["hook_calls"] = [{"op": "readouts", "fired": 0}]
        elif (p["prompt_id"], layer) in excluded:
            base.update({"status": "safety_excluded", "status_detail": "ratio > cap", "activation_norm": norm,
                         "target_ratio": 0.2, "target_displacement": 20.0})
            base.pop("generation_raw")
        else:
            base.update({"activation_norm": norm, "intended_displacement": d_t, "measured_displacement": d_t,
                         "hook_calls": _hooks(layer, SEQ - 1, d_t, norm), "target_ratio": 0.05,
                         "direction_kind": "token_set_contrast_J_lens_ablation" if arm == C.ARM_TARGET else "norm_matched_displacement"})
            if arm != C.ARM_TARGET:
                base["cos_with_target"] = 0.02
        rows.append(base)
    if mutate:
        for r in rows:
            mutate(r)
    return rows


def memory_records(n_rows=128, *, t0=1000.0, gpu_alloc=1700.0):
    ok = {"free_pct": 80.0, "available_mb": 10000.0, "swap_used_mb": 10.0, "psi_some_avg10": 0.5, "gpu_free_mb": 14000.0,
          "gpu_alloc_mb": None}
    recs = [{"tag": f"preload_reading_{i}", "ts": t0 + 25 * (i - 1), **ok} for i in (1, 2, 3)]
    recs.append({"tag": "pre_model_load", "ts": t0 + 50, **ok})
    recs.append({"tag": "post_model_load", "ts": t0 + 90, **{**ok, "gpu_alloc_mb": gpu_alloc}})
    recs += [{"tag": "after_eval", "ts": t0 + 100 + i, **{**ok, "gpu_alloc_mb": gpu_alloc}} for i in range(n_rows)]
    return recs


def write_pilot(root: Path, rows=None, mem=None, fp=FP, provenance=True, preflight=True):
    root.mkdir(parents=True, exist_ok=True)
    for r in (rows if rows is not None else make_rows(fp)):
        store.write_atomic(store.row_path(root / "evals", r["key"]), r)
    recs = mem if mem is not None else memory_records()
    (root / "rq4_memory_log.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    if preflight:
        (root / "runtime_preflight.json").write_text(json.dumps({"alignment": {"formula": "h@J @ U.T", "rel_err": 0.01}}))
    if provenance:
        (root / "runtime_provenance.json").write_text(json.dumps(
            {"pinned": {"config_fingerprint": fp, "dtype": "bfloat16", "model_revision": C.MODEL_REVISION, "lens_revision": C.LENS_REVISION, "torch": "2.14.1", "transformers": "5.20.2",
                        "jlens": "0.1.0@abc", "code_commit": "a" * 40}, "sessions": []}))
    return root
