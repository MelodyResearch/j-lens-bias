"""Full-continuation teacher-forced scoring and rank readouts (pure torch; model-agnostic)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import config as C


@dataclass(frozen=True)
class CandidateTokens:
    prompt_ids: tuple[int, ...]
    full_ids: tuple[int, ...]
    start: int                 # index in full_ids where candidate tokens begin
    boundary_ok: bool          # tokenize(prompt + candidate) starts with tokenize(prompt)

    @property
    def candidate_ids(self) -> tuple[int, ...]:
        return self.full_ids[self.start:]


def candidate_tokens(tokenize, prompt: str, candidate: str) -> CandidateTokens:
    """Candidate tokens = tokens of tokenize(prompt + candidate) beyond the prompt tokens.
    Never encode(candidate)[0]; if the prompt/candidate boundary merges tokens, boundary_ok is False
    and the candidate must not be scored (it would change the prompt's own tokens)."""
    p = tuple(int(i) for i in tokenize(prompt))
    f = tuple(int(i) for i in tokenize(prompt + candidate))
    ok = len(f) > len(p) and f[: len(p)] == p
    return CandidateTokens(p, f, len(p), ok)


def mc_candidate(letter: str) -> str:
    return C.MC_CANDIDATE_FORMAT.format(letter=letter)


def open_candidate(answer_text: str) -> str:
    return C.OPEN_CANDIDATE_FORMAT.format(answer_text=answer_text)


ROLES = ("stereotype", "counter", "unknown")


def role_candidate_text(plan: dict, role: str) -> str:
    """The exact candidate string scored for `role` of a planned prompt."""
    if plan["prompt_format"] == "multiple_choice":
        letter = next(l for l, r in plan["option_roles"].items() if r == role)
        return mc_candidate(letter)
    return open_candidate(plan[f"{role}_answer_text"])


def continuation_logprob(logits: torch.Tensor, ct: CandidateTokens) -> dict:
    """logits: [T, V] for the full sequence ct.full_ids. Sum of log p(token_i | tokens_<i)."""
    if not ct.boundary_ok:
        return {"status": "boundary_mismatch", "logprob_sum": None, "token_logprobs": None, "n_tokens": 0}
    if logits.shape[0] != len(ct.full_ids):
        raise ValueError(f"logits length {logits.shape[0]} != sequence length {len(ct.full_ids)}")
    lp = torch.log_softmax(logits.float(), dim=-1)
    toks = []
    for i in range(ct.start, len(ct.full_ids)):
        toks.append(float(lp[i - 1, ct.full_ids[i]]))
    s = float(sum(toks))
    if not all(torch.isfinite(torch.tensor(toks)).tolist()):
        raise FloatingPointError("non-finite teacher-forced log probability")
    return {"status": "ok", "logprob_sum": s, "token_logprobs": toks, "n_tokens": len(toks)}


def local_ranks(logits_1d: torch.Tensor, ids) -> list[int]:
    """0-based rank = number of vocabulary entries with strictly larger logit. The GPU runtime
    substitutes jlens' own `_ranks_of` so ranks match Weeks 2-3 exactly."""
    ids_t = torch.as_tensor(list(ids), dtype=torch.long)
    vals = logits_1d.float()[ids_t]
    return [int((logits_1d.float() > v).sum()) for v in vals]


def readout_metrics(logits_1d: torch.Tensor, stereo_ids, counter_ids, rank_fn=local_ranks) -> dict:
    """Week-3 definition (min constituent rank; rank_gap = counter - stereotype, positive = stereotype
    stronger) PLUS the token-set contrast actually used for the intervention (mean-logit margin)."""
    if not bool(torch.isfinite(logits_1d.float()).all()):
        raise FloatingPointError("non-finite readout logits")
    sr, cr = rank_fn(logits_1d, stereo_ids), rank_fn(logits_1d, counter_ids)
    lg = logits_1d.float()
    s_idx = torch.as_tensor(list(stereo_ids), dtype=torch.long)
    c_idx = torch.as_tensor(list(counter_ids), dtype=torch.long)
    return {
        "stereotype_rank": int(min(sr)), "counter_rank": int(min(cr)),
        "rank_gap": int(min(cr)) - int(min(sr)),
        "stereotype_ranks": [int(x) for x in sr], "counter_ranks": [int(x) for x in cr],
        "set_logit_margin": float(lg[s_idx].mean() - lg[c_idx].mean()),
    }


def top_logits(logits_1d: torch.Tensor, k: int = C.TOP_K_LOGITS) -> dict:
    v, i = logits_1d.float().topk(k)
    return {"top_ids": [int(x) for x in i], "top_logits": [float(x) for x in v]}
