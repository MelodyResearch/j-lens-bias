"""Item loading, prompt building and the static prompt plan. Pure Python; no ML imports.

The prompt builder is a verbatim copy of Week 3's (`tests/test_data.py` asserts equality
with the Week 3 module and with every row of `week3_planned_prompts.jsonl`).
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from . import config as C

PROMPT_TEMPLATE_MC = "{context}\n{question}\n(a) {ans0}\n(b) {ans1}\n(c) {ans2}\nAnswer:"
PROMPT_TEMPLATE_OPEN = "{context}\n{question}\nAnswer:"
_COND_ORDER = {("ambig", "neg"): 0, ("ambig", "nonneg"): 1, ("disambig", "neg"): 2, ("disambig", "nonneg"): 3}


def build_prompt(fmt: str, row: dict) -> str:
    if fmt == "multiple_choice":
        return PROMPT_TEMPLATE_MC.format(
            context=row["context"], question=row["question"], ans0=row["ans0"], ans1=row["ans1"], ans2=row["ans2"]
        )
    if fmt == "open_ended":
        return PROMPT_TEMPLATE_OPEN.format(context=row["context"], question=row["question"])
    raise ValueError(f"unknown prompt_format {fmt!r}")


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def pair_id_for(row: dict) -> str:
    return f"{row['base_item_id']}__{row['context_condition']}__{row['question_polarity']}"


def load_items(path: Path = C.WEEK2_ITEMS) -> list[dict]:
    """Week 2 rows, restricted to the frozen ten base items, in Week 3's order."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    first_seen: dict[str, int] = {}
    for r in rows:
        first_seen.setdefault(r["base_item_id"], len(first_seen))
    if tuple(first_seen) != C.BASE_ITEM_IDS:
        raise ValueError(f"Week 2 base items {tuple(first_seen)} != frozen {C.BASE_ITEM_IDS}")
    rows.sort(key=lambda r: (first_seen[r["base_item_id"]], _COND_ORDER[(r["context_condition"], r["question_polarity"])]))
    if len(rows) != 40:
        raise ValueError(f"expected 40 rows (10 items x 4 conditions), got {len(rows)}")
    return rows


def role_of_key(row: dict, key: str) -> str:
    """BBQ role (stereotype/counter/unknown) of an answer key, from the saved Week 2 mapping."""
    if key == row["role_stereotype_key"]:
        return "stereotype"
    if key == row["role_counter_key"]:
        return "counter"
    if key == row["role_unknown_key"]:
        return "unknown"
    raise ValueError(f"key {key!r} has no role in {row.get('base_item_id')}")


def correct_role(row: dict) -> str:
    """Role of the BBQ gold answer (`label` indexes ans0..ans2)."""
    return role_of_key(row, f"ans{row['label']}")


def option_roles(row: dict) -> dict[str, str]:
    """{'a': role, 'b': role, 'c': role} for the multiple-choice letters."""
    return {C.LETTERS[i]: role_of_key(row, f"ans{i}") for i in range(3)}


def plan_prompts(rows: list[dict], base_items: tuple[str, ...] | None = None) -> list[dict]:
    """One record per (item, condition, format). `base_items` restricts scope (stages)."""
    out = []
    for row in rows:
        if base_items is not None and row["base_item_id"] not in base_items:
            continue
        for fmt in C.FORMATS:
            prompt = build_prompt(fmt, row)
            out.append({
                "base_item_id": row["base_item_id"], "example_id": row["example_id"],
                "pair_id": pair_id_for(row), "context_condition": row["context_condition"],
                "question_polarity": row["question_polarity"], "prompt_format": fmt,
                "prompt": prompt, "prompt_hash": prompt_hash(prompt),
                "prompt_id": f"{pair_id_for(row)}__{fmt}",
                "stereotype_target_text": row["role_stereotype_target_text"],
                "counter_target_text": row["role_counter_target_text"],
                "stereotype_answer_text": row[row["role_stereotype_key"]],
                "counter_answer_text": row[row["role_counter_key"]],
                "unknown_answer_text": row[row["role_unknown_key"]],
                "option_roles": option_roles(row), "correct_role": correct_role(row),
                "option_texts": {C.LETTERS[i]: row[f"ans{i}"] for i in range(3)},
                "label": row["label"],
            })
    return out


def week3_reference_positions(path: Path = C.WEEK3_RAW) -> dict[tuple[str, str], dict]:
    """Saved Week 3 seq_len / analysis_position / source_token per (pair_id, format)."""
    csv.field_size_limit(10**9)
    ref: dict[tuple[str, str], dict] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["readout"] == "jacobian_lens" and r["layer"] == "0":
                ref[(r["pair_id"], r["prompt_format"])] = {
                    "seq_len": int(r["seq_len"]), "analysis_position": int(r["analysis_position"]),
                    "source_token": r["source_token"],
                }
    return ref
