"""One-evaluation orchestration against an abstract Backend (real: rq4.runtime; tests: fakes).

Protocol Backend (duck-typed):
  d_model, n_layers
  tokenize(text) -> list[int]                  (prompt path: identical to model.encode / Week 3 / lens.apply)
  phrase_tokenize(text) -> list[int]           (optional; target/sham phrases, add_special_tokens=False)
  decode_token(id) -> str
  capture(ids, layers, position) -> {layer: Tensor[d]}      clean block outputs, read-only
  logits(ids, edit|None) -> (Tensor[T,V], HookRecord|None)
  readouts(prompt, ids, edit|None) -> ({readout_key: Tensor[V]}, Tensor[V], HookRecord|None)
  generate(ids, edit|None, max_new) -> (raw_text, HookRecord|None)
  unembedding() -> Tensor[V,d];  jacobian(layer) -> Tensor[d,d];  rank_fn(logits, ids) -> list[int]
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from . import config as C
from . import directions as D
from . import parsing, scoring
from .hooks import DisplacementCapError, Edit, HookRecord, compute_delta
from .tokens import IneligiblePair, TargetIds


@dataclass
class PromptContext:
    plan: dict
    ids: list[int]
    position: int
    targets: TargetIds
    clean_h: dict[int, torch.Tensor]


def source_position(backend, plan: dict) -> tuple[list[int], int]:
    """Token ids and the 'Answer:' source position (last token), asserting it decodes to ':'."""
    ids = [int(i) for i in backend.tokenize(plan["prompt"])]
    pos = len(ids) - 1
    tok = backend.decode_token(ids[pos])
    if tok.strip() != C.EXPECTED_SOURCE_TOKEN:
        raise AssertionError(f"{plan['prompt_id']}: final token {tok!r} is not ':'")
    return ids, pos


def _base_row(ctx: PromptContext, arm: str, layer: int | None, stage: str, fingerprint: str) -> dict:
    from .store import eval_key
    p = ctx.plan
    return {
        "schema_version": C.SCHEMA_VERSION, "key": eval_key(p["prompt_id"], arm, layer),
        "config_fingerprint": fingerprint, "stage": stage, "parser_version": C.PARSER_VERSION,
        "base_item_id": p["base_item_id"], "pair_id": p["pair_id"], "prompt_id": p["prompt_id"],
        "context_condition": p["context_condition"], "question_polarity": p["question_polarity"],
        "prompt_format": p["prompt_format"], "prompt_hash": p["prompt_hash"],
        "arm": arm, "layer": layer, "seed": int(arm.split("_s")[1]) if arm.startswith("random_s") else None,
        "status": "ok", "status_detail": None, "seq_len": len(ctx.ids), "position": ctx.position,
        "correct_role": p["correct_role"],
        "target_ids": ctx.targets.as_log(),
        "alpha": C.ALPHA, "cap": C.DISPLACEMENT_CAP,
    }


def _hook_summary(op: str, rec: HookRecord | None, edit: Edit | None) -> dict:
    if edit is None:
        return {"op": op, "fired": 0}
    if rec is None or rec.fired != 1 or rec.layer != edit.layer or rec.position != edit.position:
        raise RuntimeError(f"hook alignment failure in {op}: {rec}")
    return {"op": op, "fired": rec.fired, "layer": rec.layer, "position": rec.position, "dtype": rec.dtype,
            "seq_len": rec.seq_len, "activation_norm": rec.activation_norm,
            "intended": rec.intended_displacement, "measured": rec.measured_displacement}


def run_condition(backend, ctx: PromptContext, edit: Edit | None) -> dict:
    """Every outcome for one (arm, layer) cell. Raises on non-finite values or hook misalignment."""
    p, hooks = ctx.plan, []
    # --- teacher-forced full-continuation scores ---------------------------------------------------
    cands = {}
    for role in scoring.ROLES:
        text = scoring.role_candidate_text(p, role)
        ct = scoring.candidate_tokens(backend.tokenize, p["prompt"], text)
        if ct.boundary_ok:
            if ct.start - 1 != ctx.position:
                raise AssertionError("candidate start does not follow the ':' source position")
            logits, rec = backend.logits(list(ct.full_ids), edit)
            hooks.append(_hook_summary(f"tf:{role}", rec, edit))
            res = scoring.continuation_logprob(logits.cpu(), ct)
        else:
            res = scoring.continuation_logprob(torch.zeros(len(ct.full_ids), 1), ct)
        cands[role] = {"text": text, "token_ids": list(ct.candidate_ids), **res}
    lp = {r: cands[r]["logprob_sum"] for r in cands}
    margin = (lp["stereotype"] - lp["counter"]) if lp["stereotype"] is not None and lp["counter"] is not None else None
    # --- internal readouts + next-token logits -----------------------------------------------------
    lens_logits, model_logits, rec = backend.readouts(p["prompt"], ctx.ids, edit)
    hooks.append(_hook_summary("readouts", rec, edit))
    s_ids, c_ids = ctx.targets.stereotype, ctx.targets.counter
    readouts = {str(k): scoring.readout_metrics(v.cpu(), s_ids, c_ids, backend.rank_fn) for k, v in lens_logits.items()}
    ml = model_logits.cpu().float()
    next_tok = scoring.top_logits(ml)
    next_tok["stereotype_id_logits"] = {str(i): float(ml[i]) for i in s_ids}
    next_tok["counter_id_logits"] = {str(i): float(ml[i]) for i in c_ids}
    next_tok["candidate_first_token_logits"] = {r: (float(ml[cands[r]["token_ids"][0]]) if cands[r]["token_ids"] else None) for r in cands}
    if not bool(torch.isfinite(ml).all()):
        raise FloatingPointError("non-finite model logits")
    # --- greedy generation + parsing ---------------------------------------------------------------
    raw, rec = backend.generate(ctx.ids, edit, C.MAX_NEW_TOKENS)
    hooks.append(_hook_summary("generate", rec, edit))
    if p["prompt_format"] == "multiple_choice":
        parsed = parsing.parse_multiple_choice(raw, p["option_roles"], p["option_texts"])
    else:
        parsed = parsing.parse_open_ended(raw, p["stereotype_answer_text"], p["counter_answer_text"], p["unknown_answer_text"])
    out = {
        "teacher_forced": cands, "tf_margin_stereo_minus_counter": margin,
        "readouts": readouts, "next_token": next_tok,
        "generation_raw": raw, **parsed.as_dict(),
        "follows_evidence": parsing.follows_evidence(parsed.category, p["correct_role"]),
        "evidence_applicable": p["context_condition"] == "disambig",
        "hook_calls": hooks,
    }
    if edit is not None:
        fired = [h for h in hooks if h["fired"]]
        out["measured_displacement"] = max(h["measured"] for h in fired)
        out["measured_displacement_min"] = min(h["measured"] for h in fired)
        out["measured_ratio"] = out["measured_displacement"] / fired[0]["activation_norm"]
    return out


def evaluate_prompt(backend, ctx: PromptContext, dirs: "DirectionBank", todo: set[str], stage: str,
                    fingerprint: str, save, guard=lambda: None) -> int:
    """Run every not-yet-done evaluation for one prompt; `save(row)` persists atomically.

    `guard()` runs after every actual run_condition (clean, target, each control) and raises to stop at once; rows
    buffered for the current prompt/layer cell are then discarded (a partly buffered cell is recomputed on resume,
    while cells already saved stay checkpointed). Status-only rows (no forward pass) call the guard when emitted."""
    from .store import eval_key
    p, n = ctx.plan, 0
    k = lambda arm, layer: eval_key(p["prompt_id"], arm, layer)

    def emit(row, guarded=False):
        """Persist one row. `guarded=True` means the guard already ran right after the run_condition that produced it."""
        nonlocal n
        save(row)
        n += 1
        if not guarded:
            guard()

    if k(C.CLEAN, None) in todo:
        row = _base_row(ctx, C.CLEAN, None, stage, fingerprint)
        row.update(run_condition(backend, ctx, None))
        guard()                                # a failing guard stops before this row is saved
        row["activation_norm_by_layer"] = {str(l): float(h.double().norm()) for l, h in ctx.clean_h.items()}
        emit(row, guarded=True)
    for layer in C.LAYERS:
        cell_keys = {arm: k(arm, layer) for arm in C.ARMS_INTERVENTION}
        if not any(v in todo for v in cell_keys.values()):
            continue
        h = ctx.clean_h[layer]
        hn = float(h.double().norm())

        def non_ok(arm, status, detail, **extra):
            if cell_keys[arm] in todo:
                r = _base_row(ctx, arm, layer, stage, fingerprint)
                r.update({"status": status, "status_detail": detail, "activation_norm": hn, **extra})
                emit(r)

        try:
            u_t = dirs.target(ctx.targets, layer)
        except D.DegenerateDirection as exc:
            for arm in C.ARMS_INTERVENTION:
                non_ok(arm, "ineligible_pair", f"target direction degenerate: {exc}")
            continue
        target_edit = Edit(layer, ctx.position, u_t, "ablate")
        d_t = float(compute_delta(h, target_edit).double().norm())
        ratio_t = d_t / hn
        calib = {"target_displacement": d_t, "target_ratio": ratio_t, "target_projection": float(h @ u_t)}
        if not math.isfinite(ratio_t) or ratio_t > C.DISPLACEMENT_CAP:
            for arm in C.ARMS_INTERVENTION:
                non_ok(arm, "safety_excluded", f"target ablation ratio {ratio_t:.4f} > cap {C.DISPLACEMENT_CAP}", **calib)
            continue
        # Compute the whole cell before saving anything: a measured (bf16/fp16) cap breach in ANY arm excludes the
        # entire prompt/layer cell, never a single arm (a mixed cell would be neither a clean comparison nor a clean exclusion).
        pending, cap_hit = [], None
        for arm in C.ARMS_INTERVENTION:
            if cell_keys[arm] not in todo:
                continue
            extra = dict(calib)
            if arm == C.ARM_TARGET:
                edit, extra["direction_kind"] = target_edit, "token_set_contrast_J_lens_ablation"
            else:
                try:
                    u_c = dirs.sham(layer) if arm == C.ARM_SHAM else dirs.random(int(arm.split("_s")[1]))
                    extra["cos_with_target"] = D.check_control(u_c, u_t)
                    if d_t <= 0.0:
                        raise D.DegenerateDirection("target displacement is zero; norm-matched control undefined")
                except (D.DegenerateDirection, IneligiblePair) as exc:
                    pending.append(("non_ok", arm, "control_degenerate", str(exc), dict(calib)))
                    continue
                edit = Edit(layer, ctx.position, u_c, "norm_matched", magnitude=d_t)
                extra["direction_kind"] = "norm_matched_displacement"
            try:
                body = run_condition(backend, ctx, edit)
            except DisplacementCapError as exc:
                guard()                        # the forward pass ran; check resources before any further work
                cap_hit = (arm, exc)
                break
            guard()                            # after EVERY actual run_condition, before the next arm
            extra["calibration_ratio"] = body["measured_displacement"] / d_t   # measured / target ablation norm
            row = _base_row(ctx, arm, layer, stage, fingerprint)
            row.update({"activation_norm": hn, **extra, **body})
            row["intended_displacement"] = d_t
            row["intended_ratio"] = ratio_t
            pending.append(("row", row))
        if cap_hit is not None:
            trig_arm, exc = cap_hit
            for arm in C.ARMS_INTERVENTION:
                non_ok(arm, "safety_excluded", f"measured displacement cap breached by arm {trig_arm}: {exc}",
                       intended_displacement=d_t, intended_ratio=ratio_t, cell_exclusion="measured_cap",
                       cell_exclusion_arm=trig_arm, measured_cap_ratio=exc.ratio if exc.ratio is not None and math.isfinite(exc.ratio) else None, **calib)
            continue
        for item in pending:
            if item[0] == "row":
                emit(item[1], guarded=True)        # guard already ran after its run_condition
            else:
                non_ok(item[1], item[2], item[3], **item[4])
    return n


class DirectionBank:
    """Target / sham / random directions, cached per layer (target per target-id set)."""

    def __init__(self, backend, sham_ids: TargetIds):
        self.backend, self.sham_ids = backend, sham_ids
        self._t, self._s, self._r = {}, {}, {}

    def _contrast(self, ids: TargetIds, layer: int) -> torch.Tensor:
        return D.contrast_direction(self.backend.unembedding(), self.backend.jacobian(layer), ids.stereotype, ids.counter)

    def target(self, ids: TargetIds, layer: int) -> torch.Tensor:
        key = (ids.stereotype, ids.counter, layer)
        if key not in self._t:
            self._t[key] = self._contrast(ids, layer).cpu()
        return self._t[key]

    def sham(self, layer: int) -> torch.Tensor:
        if layer not in self._s:
            self._s[layer] = self._contrast(self.sham_ids, layer).cpu()
        return self._s[layer]

    def random(self, seed: int) -> torch.Tensor:
        if seed not in self._r:
            self._r[seed] = D.random_direction(seed, self.backend.d_model)
        return self._r[seed]


def ineligible_rows(plan: dict, detail: str, stage: str, fingerprint: str, position: int | None = None, seq_len: int | None = None):
    """Rows (all arms/layers) for a pair excluded before any outcome existed (tokenizer-level reasons)."""
    from .store import eval_key
    for arm, layer in [(C.CLEAN, None)] + [(a, l) for l in C.LAYERS for a in C.ARMS_INTERVENTION]:
        yield {
            "schema_version": C.SCHEMA_VERSION, "key": eval_key(plan["prompt_id"], arm, layer),
            "config_fingerprint": fingerprint, "stage": stage, "parser_version": C.PARSER_VERSION,
            "base_item_id": plan["base_item_id"], "pair_id": plan["pair_id"], "prompt_id": plan["prompt_id"],
            "context_condition": plan["context_condition"], "question_polarity": plan["question_polarity"],
            "prompt_format": plan["prompt_format"], "prompt_hash": plan["prompt_hash"], "arm": arm, "layer": layer,
            "seed": int(arm.split("_s")[1]) if arm.startswith("random_s") else None,
            "status": "ineligible_pair", "status_detail": detail, "seq_len": seq_len, "position": position,
            "correct_role": plan["correct_role"],
        }
