"""Stage-0 pre-outcome repairs: special-token paths, candidate-boundary preflight, safe capture cleanup,
measured (bf16) displacement cap. Fakes only: no weights, no inference."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rq4 import config as C
from rq4 import data
from rq4 import preflight as P
from rq4.hooks import DisplacementCapError, Edit, InterventionHook
from rq4.runtime import QwenBackend
from toy import ToyBackend, ToyTokenizer

BOS = 999


class SpecialTokenTok:
    """Adds BOS only when asked; words -> stable ids."""

    def __call__(self, text, add_special_tokens=True):
        ids = [10 + (sum(map(ord, w)) % 500) for w in text.split()]
        return SimpleNamespace(input_ids=([BOS] + ids) if add_special_tokens else ids)

    def decode(self, ids, **kw):
        return ":"


def bare_qwen():
    qb = QwenBackend.__new__(QwenBackend)       # no __init__: no model load
    tok = SpecialTokenTok()
    qb.tokenizer = tok
    qb.model = SimpleNamespace(encode=lambda text: torch.tensor([tok(text, add_special_tokens=True).input_ids]))
    return qb


def test_real_backend_prompt_path_keeps_special_tokens_phrase_path_drops_them():
    qb = bare_qwen()
    assert qb.tokenize("the old man")[0] == BOS                    # == model.encode / Week 3 / lens.apply
    assert BOS not in qb.phrase_tokenize(" the old man")
    mask = torch.ones(2000, dtype=torch.bool)
    pairs, sham = P.resolve_pairs(qb, [{"pair_id": "p", "stereotype_target_text": "old man", "counter_target_text": "young man"}], mask)
    assert BOS not in pairs["p"].raw_stereotype + pairs["p"].raw_counter + sham.raw_stereotype + sham.raw_counter


def test_shim_never_requests_special_tokens_and_falls_back_for_fake_backends():
    with pytest.raises(ValueError):
        P._Shim(bare_qwen())("x", add_special_tokens=True)
    fake = ToyBackend()                                            # no phrase_tokenize -> falls back to tokenize
    assert P._Shim(fake)("a b").input_ids == fake.tokenize("a b")


def _plan(n=2):
    return data.plan_prompts(data.load_items())[:n * 2]


def test_boundary_report_covers_every_role_and_format_and_passes_on_clean_tokenizer():
    plan = _plan(2)
    assert {p["prompt_format"] for p in plan} == {"multiple_choice", "open_ended"} or len(plan) >= 2
    rep = P.candidate_boundary_report(ToyBackend(), plan)
    assert len(rep) == 3 * len(plan)
    assert {(r["prompt_id"], r["role"]) for r in rep} == {(p["prompt_id"], r) for p in plan for r in ("stereotype", "counter", "unknown")}
    assert all(r["ok"] and r["candidate_token_ids"] for r in rep)
    assert {"prompt_id", "prompt_format", "role", "prompt_token_ids", "candidate_token_ids"} <= set(rep[0])
    P.assert_boundaries(rep)
    json.dumps(rep)                                                # saveable


def test_boundary_merge_fails_and_identifies_prompt_format_role_and_ids():
    plan = [p for p in data.plan_prompts(data.load_items()) if p["prompt_format"] == "multiple_choice"][:1]
    rep = P.candidate_boundary_report(ToyBackend(tokenizer=ToyTokenizer(merge_colon_space=True)), plan)
    bad = [r for r in rep if not r["ok"]]
    assert bad and not bad[0]["boundary_ok"]
    assert bad[0]["prompt_id"] == plan[0]["prompt_id"] and bad[0]["prompt_format"] == "multiple_choice" and bad[0]["role"]
    with pytest.raises(P.CandidateBoundaryError, match=plan[0]["prompt_id"]):
        P.assert_boundaries(rep)


def test_boundary_with_no_candidate_tokens_fails():
    plan = _plan(1)[:1]

    class Trunc(ToyBackend):
        def tokenize(self, text):
            return super().tokenize(text)[: len(super().tokenize(plan[0]["prompt"]))]   # drops candidate tokens
    rep = P.candidate_boundary_report(Trunc(), plan)
    assert all(not r["ok"] and r["candidate_token_ids"] == [] for r in rep)
    with pytest.raises(P.CandidateBoundaryError):
        P.assert_boundaries(rep)


def test_boundary_report_uses_the_evaluation_candidate_tokens(monkeypatch):
    from rq4 import scoring
    seen = []
    real = scoring.candidate_tokens
    monkeypatch.setattr(scoring, "candidate_tokens", lambda *a: (seen.append(a[2]), real(*a))[1])
    plan = _plan(1)
    P.candidate_boundary_report(ToyBackend(), plan)
    assert len(seen) == 3 * len(plan)


def test_capture_registration_failure_unwinds_earlier_hooks():
    class Layer(nn.Module):
        def forward(self, x):
            return x

    class Bad(Layer):
        def register_forward_hook(self, *a, **k):
            raise RuntimeError("registration failed")

    layers = nn.ModuleList([Layer(), Layer(), Bad(), Layer()])
    qb = QwenBackend.__new__(QwenBackend)
    qb.torch, qb.device = torch, "cpu"
    qb.model = SimpleNamespace(layers=layers)
    qb.hf = lambda **kw: None
    with pytest.raises(RuntimeError, match="registration failed"):
        qb.capture([1, 2, 3], [0, 1, 2, 3], 2)
    assert all(len(l._forward_hooks) == 0 for l in layers)


def test_capture_success_removes_hooks_and_returns_values():
    class Layer(nn.Module):
        def forward(self, x):
            return x + 1

    layers = nn.ModuleList([Layer(), Layer()])
    qb = QwenBackend.__new__(QwenBackend)
    qb.torch, qb.device = torch, "cpu"
    qb.model = SimpleNamespace(layers=layers)

    def hf(input_ids, use_cache):
        h = torch.zeros(1, input_ids.shape[1], 4)
        for l in layers:
            h = l(h)
        return h
    qb.hf = hf
    out = qb.capture([1, 2, 3], [0, 1], 2)
    assert float(out[0][0]) == 1.0 and float(out[1][0]) == 2.0
    assert all(len(l._forward_hooks) == 0 for l in layers)


def test_measured_bf16_displacement_over_cap_fails_closed():
    mod = nn.Identity()
    h = torch.randn(1, 4, 64, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16)
    u = torch.nn.functional.normalize(torch.randn(64, generator=torch.Generator().manual_seed(1)), dim=0)
    raised = passed = 0
    for seed in range(40):
        hh = torch.randn(1, 4, 64, generator=torch.Generator().manual_seed(seed + 5)).to(torch.bfloat16)
        mag = C.DISPLACEMENT_CAP * float(hh[0, 3].float().norm())            # intended float32 ratio == cap exactly
        hook = InterventionHook(mod, Edit(0, 3, u, "norm_matched", magnitude=mag))
        with hook:
            try:
                mod(hh)
            except DisplacementCapError:
                raised += 1
                assert hook.record.fired == 0 and hook.record.measured_displacement is None   # not recorded as ok
                continue
        passed += 1
        assert hook.record.measured_displacement <= C.DISPLACEMENT_CAP * hook.record.activation_norm * (1 + 1e-6)
    assert raised > 0 and passed > 0
    assert len(mod._forward_hooks) == 0
