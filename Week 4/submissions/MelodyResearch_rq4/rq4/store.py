"""Atomic per-evaluation checkpoints, unique-key schema, and safe resume.

One JSON file per evaluation, named by its key. Writes go to a temp file, are fsynced, then
renamed; nothing is ever overwritten silently. On resume every file is schema-validated and
checked against the run's config fingerprint; invalid files are moved to quarantine/ (not
deleted) and re-run; a valid file with a *different* fingerprint aborts the run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from . import config as C

REQUIRED_FIELDS = {
    "schema_version": int, "key": str, "config_fingerprint": str, "stage": str,
    "base_item_id": str, "pair_id": str, "prompt_id": str, "context_condition": str,
    "question_polarity": str, "prompt_format": str, "prompt_hash": str,
    "arm": str, "layer": (int, type(None)), "status": str,
}
STATUSES = ("ok", "safety_excluded", "ineligible_pair", "control_degenerate", "error")


class FingerprintMismatch(RuntimeError):
    pass


def eval_key(prompt_id: str, arm: str, layer: int | None) -> str:
    """Unique evaluation key. The clean baseline is layer-independent (layer None -> 'LNA')."""
    return f"{prompt_id}__{arm}__L{'NA' if layer is None else layer}"


def planned_keys(plan: list[dict]) -> list[tuple[dict, str, int | None]]:
    out = []
    for p in plan:
        out.append((p, C.CLEAN, None))
        for layer in C.LAYERS:
            for arm in C.ARMS_INTERVENTION:
                out.append((p, arm, layer))
    return out


def validate_row(row: dict) -> list[str]:
    errs = []
    for k, typ in REQUIRED_FIELDS.items():
        if k not in row:
            errs.append(f"missing {k}")
        elif not isinstance(row[k], typ):
            errs.append(f"bad type for {k}: {type(row[k]).__name__}")
    if not errs:
        if row["schema_version"] != C.SCHEMA_VERSION:
            errs.append(f"schema_version {row['schema_version']} != {C.SCHEMA_VERSION}")
        if row["arm"] not in C.ARMS_ALL:
            errs.append(f"unknown arm {row['arm']}")
        if row["status"] not in STATUSES:
            errs.append(f"unknown status {row['status']}")
        if row["key"] != eval_key(row["prompt_id"], row["arm"], row["layer"]):
            errs.append("key does not match prompt_id/arm/layer")
        if (row["arm"] == C.CLEAN) != (row["layer"] is None):
            errs.append("clean must have layer None and only clean")
    return errs


def row_path(eval_dir: Path, key: str) -> Path:
    return Path(eval_dir) / f"{key}.json"


def write_atomic(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, allow_nan=False)    # NaN/inf must never reach disk
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_row(eval_dir: Path, row: dict, *, overwrite: bool = False) -> Path:
    errs = validate_row(row)
    if errs:
        raise ValueError(f"refusing to save invalid row {row.get('key')}: {errs}")
    path = row_path(eval_dir, row["key"])
    if path.exists() and not overwrite:
        raise FileExistsError(f"duplicate evaluation key {row['key']}")
    write_atomic(path, row)
    return path


def _quarantine(path: Path, qdir: Path, reason: str) -> None:
    qdir.mkdir(parents=True, exist_ok=True)
    dest = qdir / f"{path.name}.{reason}"
    n = 0
    while dest.exists():
        n += 1
        dest = qdir / f"{path.name}.{reason}.{n}"
    os.replace(path, dest)


def load_valid_rows(eval_dir: Path, fingerprint: str, qdir: Path = C.QUARANTINE_DIR) -> dict[str, dict]:
    """Return {key: row} for valid checkpoints. Stray .tmp files and corrupt/invalid files are
    quarantined. A valid row from a different config fingerprint raises FingerprintMismatch."""
    eval_dir = Path(eval_dir)
    rows: dict[str, dict] = {}
    if not eval_dir.exists():
        return rows
    for tmp in eval_dir.glob("*.tmp"):
        _quarantine(tmp, qdir, "stale_tmp")
    for path in sorted(eval_dir.glob("*.json")):
        try:
            row = json.loads(path.read_text())
            errs = validate_row(row) if isinstance(row, dict) else ["not an object"]
        except Exception as exc:  # noqa: BLE001
            errs = [f"unreadable: {exc!r}"]
            row = None
        if errs or path.stem != row["key"]:
            _quarantine(path, qdir, "invalid")
            continue
        if row["config_fingerprint"] != fingerprint:
            raise FingerprintMismatch(
                f"{path.name} was written under config {row['config_fingerprint']}, current is {fingerprint}; "
                f"refusing to mix. Use a new results directory or restore the matching config."
            )
        rows[row["key"]] = row
    return rows


def combine(eval_dir: Path, out_jsonl: Path, fingerprint: str) -> int:
    """Rebuild the combined JSONL from checkpoints (sorted, unique keys)."""
    rows = load_valid_rows(eval_dir, fingerprint)
    tmp = Path(out_jsonl).with_name(Path(out_jsonl).name + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w") as f:
        for k in sorted(rows):
            f.write(json.dumps(rows[k], allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out_jsonl)
    return len(rows)
