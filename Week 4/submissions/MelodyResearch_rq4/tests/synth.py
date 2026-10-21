"""Synthetic saved-results generator with KNOWN effects (for analysis tests only; not data)."""

import json

from rq4 import config as C
from rq4 import data, store


def synth_rows(n_items=10, mc_effect=2.0, oe_effect=0.5, rand_effect=0.1, flip_target=True, drop=None):
    plan = [p for p in data.plan_prompts(data.load_items()) if p["base_item_id"] in C.BASE_ITEM_IDS[:n_items]]
    rows = []
    for i, p in enumerate(plan):
        eff = mc_effect if p["prompt_format"] == "multiple_choice" else oe_effect
        base_m = 0.3 * (i % 5)
        for _, arm, layer in [(p, C.CLEAN, None)] + [(p, a, l) for l in C.LAYERS for a in C.ARMS_INTERVENTION]:
            if drop and (p["prompt_id"], arm, layer) in drop:
                continue
            if arm == C.CLEAN:
                d = 0.0
            elif arm == C.ARM_TARGET:
                d = eff
            elif arm == C.ARM_SHAM:
                d = 0.2 * rand_effect
            else:
                d = rand_effect * (1 + int(arm[-2:]) % 3)
            cat = "stereotype"
            if arm == "target" and flip_target and p["prompt_format"] == "multiple_choice":
                cat = "counter"
            r = {"schema_version": C.SCHEMA_VERSION, "key": store.eval_key(p["prompt_id"], arm, layer), "config_fingerprint": "f",
                 "stage": "2", "base_item_id": p["base_item_id"], "pair_id": p["pair_id"], "prompt_id": p["prompt_id"],
                 "context_condition": p["context_condition"], "question_polarity": p["question_polarity"],
                 "prompt_format": p["prompt_format"], "prompt_hash": p["prompt_hash"], "arm": arm, "layer": layer,
                 "seed": int(arm.split("_s")[1]) if arm.startswith("random_s") else None, "status": "ok", "status_detail": None,
                 "correct_role": p["correct_role"], "category": cat, "parse_rule": "letter", "generation_raw": " (a)",
                 "follows_evidence": cat == p["correct_role"], "evidence_applicable": p["context_condition"] == "disambig",
                 "tf_margin_stereo_minus_counter": base_m + d, "activation_norm": 10.0,
                 "teacher_forced": {"stereotype": {"logprob_sum": -1.0}, "counter": {"logprob_sum": -1.0 - base_m - d}},
                 "readouts": {"8": {"rank_gap": 5, "set_logit_margin": 0.1}}}
            if arm != C.CLEAN:
                r.update({"measured_displacement": 0.5, "measured_ratio": 0.05, "calibration_ratio": 1.0, "intended_displacement": 0.5})
            rows.append(r)
    return rows


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
