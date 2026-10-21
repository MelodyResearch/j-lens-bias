"""Static manifests (no model, no tokenizer): prompt eligibility and config/seed manifests.

Contents are deterministic (no timestamps) so `--prepare-only` is idempotent and diffable.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import config as C
from . import data

# Outcome-blind intelligibility review of every open-ended question, done 2026-10-20 by reading the
# prompts only (no model output existed). One line per base item; no item was excluded.
INTELLIGIBILITY_NOTES = {
    "age_01_qi1": "open-ended questions answerable from context (grandson/grandfather); intelligible",
    "age_02_qi2": "intelligible; ambiguous-context nonneg question presupposes both walk together, still answerable",
    "age_03_qi3": "intelligible; target 'elderly' vs 'younger' answers are 'The elderly man'/'The younger man'",
    "age_04_qi4": "intelligible; answers 'my grandpa'/'my son' (first-person context)",
    "age_05_qi5": "intelligible; ambiguous context names two professors, question answerable with 'Millennial'/'Boomer'",
    "age_06_qi9": "intelligible; 'valued equal rights' questions answerable from context",
    "age_07_qi13": "intelligible; disambiguating text contains BBQ's original typo 'load it to their son' (kept verbatim)",
    "age_08_qi16": "intelligible; 'middle-aged guy' answer contains a hyphen (parser alias handled)",
    "age_09_qi19": "intelligible; long context, question answerable",
    "age_10_qi22": "intelligible; answers '15 year old'/'40-year-old' tokenize into digits (checked at preflight)",
}


def eligibility_manifest(rows: list[dict] | None = None) -> dict:
    rows = rows or data.load_items()
    plan = data.plan_prompts(rows)
    w3 = {}
    for line in C.WEEK3_PLANNED.read_text().splitlines():
        r = json.loads(line)
        w3[(r["pair_id"], r["prompt_format"])] = r
    w3pos = data.week3_reference_positions()
    prompts, problems = [], []
    for p in plan:
        ref = w3.get((p["pair_id"], p["prompt_format"]))
        matches = ref is not None and ref["prompt"] == p["prompt"]
        pos = w3pos.get((p["pair_id"], p["prompt_format"]))
        ends = p["prompt"].endswith("Answer:")
        row = next(r for r in rows if data.pair_id_for(r) == p["pair_id"])
        roles_ok = sorted(data.option_roles(row).values()) == ["counter", "stereotype", "unknown"]
        txt_ok = (p["stereotype_target_text"].lower() in p["stereotype_answer_text"].lower()
                  and p["counter_target_text"].lower() in p["counter_answer_text"].lower())
        gold = data.correct_role(row)
        gold_ok = (gold == "unknown") if p["context_condition"] == "ambig" else (gold in ("stereotype", "counter"))
        ok = all([matches, pos is not None, ends, roles_ok, txt_ok, gold_ok])
        if not ok:
            problems.append(p["prompt_id"])
        prompts.append({
            "prompt_id": p["prompt_id"], "prompt_hash": p["prompt_hash"], "matches_week3_planned_prompt": matches,
            "week3_seq_len": pos and pos["seq_len"], "week3_analysis_position": pos and pos["analysis_position"],
            "week3_source_token": pos and pos["source_token"], "ends_with_Answer_colon": ends,
            "roles_one_each": roles_ok, "target_text_inside_answer_text": txt_ok, "gold_role_consistent": gold_ok,
            "correct_role": gold, "stereotype_target_text": p["stereotype_target_text"],
            "counter_target_text": p["counter_target_text"], "static_eligible": ok,
        })
    return {
        "protocol": "FROZEN_PROTOCOL.md v1 (2026-10-20)", "n_base_items": len(C.BASE_ITEM_IDS),
        "n_prompts": len(prompts), "n_pairs": len(prompts) // 2, "base_item_ids": list(C.BASE_ITEM_IDS),
        "intelligibility_review": {"method": "outcome-blind reading of open-ended prompts (no model output existed)",
                                   "notes": INTELLIGIBILITY_NOTES},
        "pre_outcome_exclusions": [],
        "static_failures": problems,
        "pending_at_stage1_preflight": [
            "tokenizer-level eligibility (no meaningful constituent token / identical token sets) for every pair",
            "banana/table sham token ids", "boundary stability of prompt+candidate tokenization",
            "final-token ':' and seq_len equality with Week 3 for every prompt",
        ],
        "prompts": prompts,
    }


def config_manifest() -> dict:
    n_prompts, per_prompt = 80, 1 + len(C.LAYERS) * len(C.ARMS_INTERVENTION)
    return {
        "config": C.config_dict(False), "config_fingerprint": C.config_fingerprint(False),
        "arms": list(C.ARMS_ALL), "layers": list(C.LAYERS), "random_seeds": list(C.RANDOM_SEEDS),
        "random_seed_note": "unit Gaussian from torch.Generator(cpu).manual_seed(seed), same vector for every prompt and layer",
        "bootstrap": {"n": C.N_BOOT, "seed": C.BOOT_SEED, "resampling_unit": "base item"},
        "counts": {"prompts": n_prompts, "evaluations_per_prompt": per_prompt, "evaluations_total": n_prompts * per_prompt,
                   "stage1_prompts": 8, "stage1_evaluations": 8 * per_prompt},
        "control_construction": {
            "target": "clean projection ablation along unit(J_l^T (mean U[stereo ids] - mean U[counter ids]))",
            "random_and_sham": "norm-matched displacement: delta = -sign(h.u_c) * ||delta_target|| * u_c (NOT clean ablation)",
            "cap": "||delta|| / ||h|| <= 0.10 at the edited position; target over cap => safety_excluded (all arms)",
            "degenerate_control": "zero/non-finite direction or |cos| > 0.99 with target => control_degenerate, no replacement",
        },
        "stage_scope": {"stage1": list(C.STAGE1_BASE_ITEMS), "stage2": list(C.STAGE2_BASE_ITEMS)},
        "protocol_sha256": C.FROZEN_PROTOCOL_SHA256,
    }


def write_manifests(out_dir: Path = C.MANIFEST_DIR) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, obj in (("prompt_eligibility_manifest.json", eligibility_manifest()), ("config_seed_manifest.json", config_manifest())):
        p = out_dir / name
        p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
        paths[name] = p
    return paths
