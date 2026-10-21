"""Outcome-blind Stage 1 (pilot) validator. Read-only; never writes result rows, never loads a model.

PRESPECIFIED PASS RULE (see REMOTE_EXECUTION_ADDENDUM.md). The pilot PASSES iff ALL of the following hold; any
single failure is FAIL. It inspects only structure, hook firing and safety fields. It never reads the
classified category, margins, rank gaps, evidence-following, or any effect, so it cannot be tuned to outcomes.

 K  exactly the 8 prompts x 16 = 128 planned keys exist, once each, as valid checkpoints under the current
    config fingerprint (which includes the code commit); no stray .tmp files; prompt hashes match the plan.
 S  every row is `ok` or `safety_excluded`. `error`, `control_degenerate`, `ineligible_pair` => FAIL.
    A safety-excluded cell must be the whole cell (all 5 intervention arms), triggered either by the intended
    float32 target ratio > cap or by a documented measured bf16/fp16 cap breach (every arm carries
    cell_exclusion="measured_cap" + the triggering arm); it is kept as is (never bypassed, never replaced). Each prompt needs >= 1 `ok` layer cell.
 C  all 8 clean rows are `ok`; every `ok` row has a non-empty raw generation and no hook fired on clean rows.
 H  every `ok` intervention row: >=1 `readouts` and >=1 `generate` hook call; every call fired exactly once,
    at the row's layer and the row's source position (= prompt length - 1), on the first (prefill) pass.
 D  every hook's measured displacement / activation norm <= the 10 % cap (1e-6 relative slack).
 N  every `ok` cell: random_s11/23/47 and sham present, direction kind `norm_matched_displacement`, intended
    displacement equal to the target's (1e-6), measured displacement within NORM_MATCH_RTOL (10 %) of the target's,
    |cos(control, target)| <= 0.99 recorded; target direction kind is the token-set J-Lens ablation.
 F  all stored numbers are finite (strict JSON parse).
 R  results/runtime_preflight.json exists with a finite lens/block alignment error.
 P  runtime_provenance.json pins the current config fingerprint, model/lens repo commits and a bf16/fp16 CUDA dtype.
 M  memory log: every session that reached model load has 3 pre-load readings >= 25 s apart that each passed the
    60 % / VRAM / pressure gate, a pre_model_load and a post_model_load record, and every after_eval reading
    passes `resources.is_safe` against the post-load baseline; >= one after_eval per row. Sessions that
    were refused by the pre-load gate (no model load) are reported, not failed.

PASS makes the pilot *eligible* for human/Codex review; it is not approval and does not start Stage 2.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from . import config as C
from . import data, resources, store

NORM_MATCH_RTOL = 0.10            # |measured_control - measured_target| / measured_target; technical calibration gate (tightened from 0.25 before any outcome existed)
CAP_RTOL = 1e-6
INTENDED_RTOL = 1e-6
MIN_GAP_SEC = C.PRELOAD_READING_GAP_SEC - 1.0
EXPECTED_PROMPTS, EXPECTED_KEYS = 8, 128
BAD_STATUSES = ("error", "control_degenerate", "ineligible_pair")


def _no_constants(name):
    raise ValueError(f"non-finite JSON constant {name}")


def _load_json(path: Path):
    return json.loads(Path(path).read_text(), parse_constant=_no_constants)


class _Report:
    def __init__(self):
        self.checks: list[dict] = []

    def add(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append({"check": name, "passed": bool(ok), "detail": detail})
        return bool(ok)


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _hook_ok(row: dict, errs: list[str], *, intervention: bool) -> None:
    calls = row.get("hook_calls")
    if not isinstance(calls, list) or not calls:
        errs.append("no hook_calls")
        return
    if not intervention:
        if any(c.get("fired") != 0 for c in calls):
            errs.append("hook fired on clean row")
        return
    ops = {c.get("op") for c in calls}
    for need in ("readouts", "generate"):
        if need not in ops:
            errs.append(f"no {need} hook call")
    pos, n = row.get("position"), row.get("seq_len")
    if not isinstance(n, int) or pos != n - 1:
        errs.append(f"position {pos} != seq_len-1 ({n})")
    for c in calls:
        tag = c.get("op")
        if c.get("fired") != 1:
            errs.append(f"{tag}: fired={c.get('fired')}")
        if c.get("layer") != row["layer"]:
            errs.append(f"{tag}: hook layer {c.get('layer')} != row layer {row['layer']}")
        if c.get("position") != pos:
            errs.append(f"{tag}: hook position {c.get('position')} != {pos}")
        if tag in ("readouts", "generate") and c.get("seq_len") != n:
            errs.append(f"{tag}: hook seq_len {c.get('seq_len')} != prompt length {n}")
        an, ms = c.get("activation_norm"), c.get("measured")
        if not (_finite(an) and _finite(ms) and an > 0):
            errs.append(f"{tag}: non-numeric displacement")
        elif ms / an > C.DISPLACEMENT_CAP * (1 + CAP_RTOL):
            errs.append(f"{tag}: measured ratio {ms / an:.4f} > cap {C.DISPLACEMENT_CAP}")


def _check_cells(rep: _Report, rows: dict, plan: list[dict]) -> None:
    probs: list[str] = []
    excluded: list[str] = []
    for p in plan:
        pid, ok_cells = p["prompt_id"], 0
        clean = rows[store.eval_key(pid, C.CLEAN, None)]
        if clean["status"] != "ok":
            probs.append(f"{pid}: clean status {clean['status']}")
        for layer in C.LAYERS:
            cell = {a: rows[store.eval_key(pid, a, layer)] for a in C.ARMS_INTERVENTION}
            statuses = {r["status"] for r in cell.values()}
            if statuses == {"ok"}:
                ok_cells += 1
            elif statuses == {"safety_excluded"}:
                t = cell[C.ARM_TARGET]
                intended = _finite(t.get("target_ratio")) and t["target_ratio"] > C.DISPLACEMENT_CAP
                # documented measured (bf16/fp16) cap failure: intended float32 ratio is within the cap, but every arm of the
                # cell carries the same measured_cap marker, the triggering arm, and the intended ratio.
                measured = (_finite(t.get("target_ratio")) and all(
                    r.get("cell_exclusion") == "measured_cap" and r.get("cell_exclusion_arm") in C.ARMS_INTERVENTION
                    and r.get("cell_exclusion_arm") == t.get("cell_exclusion_arm") and _finite(r.get("intended_ratio"))
                    and isinstance(r.get("status_detail"), str) and r["status_detail"] for r in cell.values()))
                if not (intended or measured):
                    probs.append(f"{pid} L{layer}: safety_excluded without target_ratio > cap or documented measured-cap failure")
                excluded.append(f"{pid} L{layer} ({'intended ratio > cap' if intended else 'measured cap'})")
            else:
                probs.append(f"{pid} L{layer}: mixed/bad cell statuses {sorted(statuses)}")
        if ok_cells == 0:
            probs.append(f"{pid}: no layer cell ran (all safety-excluded or bad)")
    rep.add("S cell statuses & coverage", not probs, "; ".join(probs) or f"safety-excluded cells (kept, reported): {excluded or 'none'}")


def _check_row_content(rep: _Report, rows: dict, plan: list[dict]) -> None:
    gen, hook, disp, ctrl = [], [], [], []
    for p in plan:
        pid = p["prompt_id"]
        clean = rows[store.eval_key(pid, C.CLEAN, None)]
        if clean["status"] == "ok":
            g = clean.get("generation_raw")
            if not (isinstance(g, str) and g.strip()):
                gen.append(clean["key"])
            errs: list[str] = []
            _hook_ok(clean, errs, intervention=False)
            hook += [f"{clean['key']}: {e}" for e in errs]
        for layer in C.LAYERS:
            cell = {a: rows[store.eval_key(pid, a, layer)] for a in C.ARMS_INTERVENTION}
            if {r["status"] for r in cell.values()} != {"ok"}:
                continue
            for r in cell.values():
                g = r.get("generation_raw")
                if not (isinstance(g, str) and g.strip()):
                    gen.append(r["key"])
                errs = []
                _hook_ok(r, errs, intervention=True)
                for e in errs:
                    (disp if "> cap" in e else hook).append(f"{r['key']}: {e}")
            t = cell[C.ARM_TARGET]
            if t.get("direction_kind") != "token_set_contrast_J_lens_ablation":
                ctrl.append(f"{t['key']}: target direction_kind {t.get('direction_kind')}")
            d_t = t.get("intended_displacement")
            m_t = t.get("measured_displacement")
            for arm in C.ARMS_INTERVENTION[1:]:
                r = cell[arm]
                if r.get("direction_kind") != "norm_matched_displacement":
                    ctrl.append(f"{r['key']}: direction_kind {r.get('direction_kind')}")
                di, dm, cos = r.get("intended_displacement"), r.get("measured_displacement"), r.get("cos_with_target")
                if not all(_finite(v) for v in (d_t, m_t, di, dm, cos)):
                    ctrl.append(f"{r['key']}: missing/non-finite control calibration fields")
                    continue
                if abs(di - d_t) > INTENDED_RTOL * max(abs(d_t), 1e-12):
                    ctrl.append(f"{r['key']}: intended {di} != target intended {d_t}")
                if m_t > 0 and abs(dm - m_t) / m_t > NORM_MATCH_RTOL:
                    ctrl.append(f"{r['key']}: measured {dm:.4g} not within {NORM_MATCH_RTOL:.0%} of target {m_t:.4g}")
                if abs(cos) > 0.99:
                    ctrl.append(f"{r['key']}: |cos|={abs(cos):.3f} > 0.99")
    rep.add("C raw nonempty generations", not gen, f"empty/missing: {gen}" if gen else "all ok rows have non-empty raw generations")
    rep.add("H hook firing/site/position", not hook, "; ".join(hook[:10]) or "every hook fired once at the row's layer and the source position")
    rep.add("D measured displacement <= cap", not disp, "; ".join(disp[:10]) or f"all measured ratios <= {C.DISPLACEMENT_CAP}")
    rep.add("N norm-matched controls", not ctrl, "; ".join(ctrl[:10]) or "random seeds and sham norm-matched to the target within tolerance")


def check_memory_log(path: Path, n_rows: int) -> tuple[bool, str]:
    path = Path(path)
    if not path.exists():
        return False, f"memory log {path} missing"
    try:
        recs = [json.loads(l, parse_constant=_no_constants) for l in path.read_text().splitlines() if l.strip()]
    except ValueError as exc:
        return False, f"memory log unreadable: {exc}"
    sessions: list[list[dict]] = []
    for r in recs:
        if r.get("tag") == "preload_reading_1":
            sessions.append([])
        if sessions:
            sessions[-1].append(r)
    problems, loaded, refused, n_eval = [], 0, 0, 0
    for i, sess in enumerate(sessions):
        by = {}
        for r in sess:
            by.setdefault(r["tag"], []).append(r)
        pre, post = by.get("pre_model_load"), by.get("post_model_load")
        if not pre:
            refused += 1
            continue
        loaded += 1
        reads = [by.get(f"preload_reading_{k}", [None])[0] for k in range(1, C.PRELOAD_READINGS + 1)]
        if any(r is None for r in reads):
            problems.append(f"session {i}: fewer than {C.PRELOAD_READINGS} pre-load readings")
            continue
        ok, why = resources.preload_gate(iter(reads).__next__, lambda s: None)
        if not ok:
            problems.append(f"session {i}: pre-load readings fail the gate ({why})")
        for a, b in zip(reads, reads[1:]):
            if not (_finite(a.get("ts")) and _finite(b.get("ts")) and b["ts"] - a["ts"] >= MIN_GAP_SEC):
                problems.append(f"session {i}: pre-load readings not spaced >= {C.PRELOAD_READING_GAP_SEC}s")
                break
        if not post:
            problems.append(f"session {i}: no post_model_load reading")
            continue
        baseline = resources.post_load_baseline(pre[0], post[0])
        okp, whyp = resources.is_safe(post[0], baseline)
        if not okp:
            problems.append(f"session {i}: post-load unsafe ({whyp})")
        for r in by.get("after_eval", []):
            n_eval += 1
            oke, whye = resources.is_safe(r, baseline)
            if not oke:
                problems.append(f"session {i}: after_eval unsafe ({whye})")
                break
    if loaded == 0:
        problems.append("no session reached model load")
    if n_eval < n_rows:
        problems.append(f"only {n_eval} after_eval readings for {n_rows} rows")
    detail = "; ".join(problems) if problems else f"{loaded} loaded session(s), {refused} refused by pre-load gate, {n_eval} after_eval readings, all safe"
    return not problems, detail


def validate_stage1(results_dir: Path | None = None, *, fingerprint: str | None = None,
                    eval_dir: Path | None = None, memory_log: Path | None = None,
                    preflight: Path | None = None, provenance: Path | None = None) -> dict:
    """Return a JSON-serialisable report. `report['passed']` is the objective gate."""
    results_dir = Path(results_dir) if results_dir is not None else C.RESULTS_DIR
    eval_dir = Path(eval_dir) if eval_dir is not None else results_dir / "evals"
    memory_log = Path(memory_log) if memory_log is not None else results_dir / "rq4_memory_log.jsonl"
    preflight = Path(preflight) if preflight is not None else results_dir / "runtime_preflight.json"
    provenance = Path(provenance) if provenance is not None else results_dir / "runtime_provenance.json"
    fingerprint = fingerprint or C.config_fingerprint()
    rep = _Report()
    plan = data.plan_prompts(data.load_items(), C.STAGE1_BASE_ITEMS)
    want = [store.eval_key(p["prompt_id"], a, l) for p, a, l in store.planned_keys(plan)]
    rep.add("K plan size", len(plan) == EXPECTED_PROMPTS and len(want) == EXPECTED_KEYS == len(set(want)),
            f"{len(plan)} prompts, {len(want)} keys")

    rows: dict[str, dict] = {}
    problems: list[str] = []
    digest = hashlib.sha256()
    for k in sorted(want):
        path = store.row_path(eval_dir, k)
        if not path.exists():
            problems.append(f"missing {k}")
            continue
        try:
            raw = path.read_text()
            row = json.loads(raw, parse_constant=_no_constants)
        except ValueError as exc:
            problems.append(f"{k}: unreadable/non-finite ({exc})")
            continue
        errs = store.validate_row(row) if isinstance(row, dict) else ["not an object"]
        if errs or row.get("key") != k:
            problems.append(f"{k}: invalid {errs}")
        elif row["config_fingerprint"] != fingerprint:
            problems.append(f"{k}: fingerprint {row['config_fingerprint']} != {fingerprint}")
        else:
            rows[k] = row
            digest.update(k.encode() + hashlib.sha256(raw.encode()).digest())
    prompt_hash = {p["prompt_id"]: p["prompt_hash"] for p in plan}
    for k, r in rows.items():
        if r["prompt_hash"] != prompt_hash.get(r["prompt_id"]):
            problems.append(f"{k}: prompt hash mismatch")
    stray = sorted(x.name for x in eval_dir.glob("*.tmp")) if eval_dir.exists() else []
    if stray:
        problems.append(f"stray temp files: {stray[:5]}")
    keys_ok = rep.add("K 128 unique valid checkpoints", not problems and len(rows) == EXPECTED_KEYS,
                      "; ".join(problems[:10]) or f"{len(rows)} valid checkpoints for {EXPECTED_PROMPTS} prompts")
    statuses: dict[str, int] = {}
    for r in rows.values():
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    bad = [k for k, r in rows.items() if r["status"] in BAD_STATUSES]
    rep.add("S no error/degenerate/ineligible rows", not bad, f"{bad[:10]}" if bad else f"status counts {statuses}")
    if keys_ok and not bad:
        _check_cells(rep, rows, plan)
        _check_row_content(rep, rows, plan)

    # R: runtime preflight
    try:
        pf = _load_json(preflight)
        err = pf["alignment"]["rel_err"]
        rep.add("R runtime preflight", _finite(err), f"alignment {pf['alignment'].get('formula')} rel_err={err}")
    except Exception as exc:  # noqa: BLE001
        rep.add("R runtime preflight", False, f"{type(exc).__name__}: {exc}")
    # P: provenance
    try:
        pv = _load_json(provenance)["pinned"]
        ok = (pv.get("config_fingerprint") == fingerprint and pv.get("dtype") in ("bfloat16", "float16")
              and pv.get("model_revision") == C.MODEL_REVISION and pv.get("lens_revision") == C.LENS_REVISION)
        rep.add("P runtime provenance", ok, f"dtype={pv.get('dtype')} torch={pv.get('torch')} transformers={pv.get('transformers')} "
                f"jlens={pv.get('jlens')} model_rev={pv.get('model_revision')} lens_rev={pv.get('lens_revision')} code_commit={pv.get('code_commit')}")
    except Exception as exc:  # noqa: BLE001
        rep.add("P runtime provenance", False, f"{type(exc).__name__}: {exc}")
    ok, detail = check_memory_log(memory_log, len(rows))
    rep.add("M memory log safety", ok, detail)
    rep.add("F finite values (strict JSON)", keys_ok, "checkpoints parsed with strict JSON" if keys_ok else "see K")

    return {"passed": all(c["passed"] for c in rep.checks), "outcome_blind": True, "fingerprint": fingerprint,
            "rows_digest": digest.hexdigest(), "n_rows": len(rows), "status_counts": statuses, "checks": rep.checks,
            "note": "PASS only makes the pilot eligible for human/Codex raw-generation review; it is not approval."}


def format_generations(results_dir: Path | None = None, eval_dir: Path | None = None, max_chars: int = 400) -> str:
    """Raw generations of the Stage 1 rows for human review (read-only; prints what is on disk, nothing else)."""
    eval_dir = Path(eval_dir) if eval_dir is not None else (Path(results_dir) if results_dir else C.RESULTS_DIR) / "evals"
    plan = data.plan_prompts(data.load_items(), C.STAGE1_BASE_ITEMS)
    out = []
    for p, arm, layer in store.planned_keys(plan):
        path = store.row_path(eval_dir, store.eval_key(p["prompt_id"], arm, layer))
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        out.append(f"{r['key']}  [{r['status']}]  parsed={r.get('category')}\n    raw: {str(r.get('generation_raw'))[:max_chars]!r}")
    return "\n".join(out) if out else "(no Stage 1 checkpoints found)"


def write_report(report: dict, path: Path) -> None:
    store.write_atomic(path, report)
