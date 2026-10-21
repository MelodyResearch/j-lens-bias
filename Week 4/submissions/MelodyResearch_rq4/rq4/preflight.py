"""Stage-1 preflight against a live backend: lens/model dimensions, ':' position, token eligibility,
sham ids, candidate boundaries and the lens-source-layer <-> block-output alignment check.
All of it is outcome-blind (tokenization and one clean forward pass)."""

from __future__ import annotations

from types import SimpleNamespace

from . import config as C
from . import scoring, tokens
from .alignment import check_lens_block_alignment


class _Shim:
    """Makes the backend's phrase tokenizer look like a HF tokenizer call for rq4.tokens. Phrases must be
    tokenized WITHOUT special tokens, so the real backend's `phrase_tokenize` is used; `tokenize` (the
    prompt path, = model.encode) is only a fallback for fake backends that have no special tokens."""

    def __init__(self, backend):
        self.fn = getattr(backend, "phrase_tokenize", None) or backend.tokenize

    def __call__(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise ValueError("target phrases are tokenized without special tokens")
        return SimpleNamespace(input_ids=self.fn(text))


def check_model_and_lens(info: dict) -> None:
    if tuple(info["source_layers"]) != C.EXPECTED_LENS_SOURCE_LAYERS:
        raise AssertionError(f"lens source layers {info['source_layers']} != exactly 0..22")
    if info["d_model"] != info["model_d_model"]:
        raise AssertionError(f"lens d_model {info['d_model']} != model d_model {info['model_d_model']}")
    if info["model_n_layers"] <= max(C.LAYERS) or max(info["source_layers"]) >= info["model_n_layers"]:
        raise AssertionError("model has too few blocks for the lens / intervention layers")


def resolve_pairs(backend, plan: list[dict], mask) -> tuple[dict, dict]:
    """{pair_id: TargetIds or reason-string}; plus the sham TargetIds."""
    shim = _Shim(backend)
    sham = tokens.resolve_target_ids(shim, C.SHAM_STEREO_TEXT, C.SHAM_COUNTER_TEXT, mask)
    out = {}
    for p in plan:
        if p["pair_id"] in out:
            continue
        try:
            out[p["pair_id"]] = tokens.resolve_target_ids(shim, p["stereotype_target_text"], p["counter_target_text"], mask)
        except tokens.IneligiblePair as exc:
            out[p["pair_id"]] = str(exc)
    return out, sham


def check_positions(backend, plan: list[dict], week3_ref: dict) -> list[dict]:
    rows = []
    for p in plan:
        ids = [int(i) for i in backend.tokenize(p["prompt"])]
        ref = week3_ref[(p["pair_id"], p["prompt_format"])]
        tok = backend.decode_token(ids[-1])
        ok = tok.strip() == C.EXPECTED_SOURCE_TOKEN and len(ids) == ref["seq_len"] and len(ids) - 1 == ref["analysis_position"]
        rows.append({"prompt_id": p["prompt_id"], "seq_len": len(ids), "week3_seq_len": ref["seq_len"], "final_token": tok, "ok": ok})
        if not ok:
            raise AssertionError(f"position check failed: {rows[-1]}")
    return rows


class CandidateBoundaryError(AssertionError):
    pass


def candidate_boundary_report(backend, plan: list[dict]) -> list[dict]:
    """Every role of every planned prompt, via the exact `scoring.candidate_tokens` the evaluation uses.
    Outcome-blind; never raises (see `assert_boundaries`)."""
    rows = []
    for p in plan:
        for role in scoring.ROLES:
            text = scoring.role_candidate_text(p, role)
            ct = scoring.candidate_tokens(backend.tokenize, p["prompt"], text)
            ok = bool(ct.boundary_ok and ct.candidate_ids and ct.start >= 1)
            rows.append({"prompt_id": p["prompt_id"], "prompt_format": p["prompt_format"], "role": role,
                         "candidate_text": text, "prompt_token_ids": list(ct.prompt_ids),
                         "full_token_ids": list(ct.full_ids), "candidate_token_ids": list(ct.candidate_ids),
                         "boundary_ok": ct.boundary_ok, "ok": ok})
    return rows


def assert_boundaries(report: list[dict]) -> None:
    bad = [r for r in report if not r["ok"]]
    if bad:
        raise CandidateBoundaryError(
            f"{len(bad)} candidate boundary failure(s): "
            + "; ".join(f"{r['prompt_id']}/{r['prompt_format']}/{r['role']} candidate_ids={r['candidate_token_ids']}"
                        for r in bad[:5]))


def check_alignment(backend, plan_item: dict) -> dict:
    """One clean forward on a real prompt: compare released-lens logits with our own reconstruction
    from hooked block outputs l-1, l, l+1 (see rq4.alignment)."""
    ids = [int(i) for i in backend.tokenize(plan_item["prompt"])]
    pos = len(ids) - 1
    need = sorted({l + s for l in C.LAYERS for s in (-1, 0, 1)})
    captured = backend.capture(ids, need, pos)
    lens_logits, _, _ = backend.readouts(plan_item["prompt"], ids, None)
    return check_lens_block_alignment(captured, {l: lens_logits[l] for l in C.LAYERS},
                                      backend.lens_formulas(), C.LAYERS)
