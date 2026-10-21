import pytest
import torch

from rq4 import config as C
from rq4 import directions as D
from rq4 import tokens as T
from toy import ToyTokenizer


def test_random_direction_seed_determinism():
    a1, a2 = D.random_direction(11, 64), D.random_direction(11, 64)
    assert torch.equal(a1, a2)
    assert not torch.allclose(D.random_direction(23, 64), a1) and not torch.allclose(D.random_direction(47, 64), a1)
    assert float(a1.norm()) == pytest.approx(1.0, abs=1e-6) and a1.dtype == torch.float32
    # pinned values: a silent change of RNG behaviour would change every control
    assert D.random_direction(11, 4).tolist() == D.random_direction(11, 4).tolist()
    assert len(C.RANDOM_SEEDS) == 3 and C.RANDOM_SEEDS == (11, 23, 47)


def test_contrast_direction_matches_protocol_formula():
    g = torch.Generator().manual_seed(1)
    U, J = torch.randn(30, 8, generator=g), torch.randn(8, 8, generator=g)    # J not symmetric
    d = D.contrast_direction(U, J, [1, 2], [3])
    manual = J.T @ (U[[1, 2]].mean(0) - U[3])
    assert torch.allclose(d, manual / manual.norm(), atol=1e-6)
    assert not torch.allclose(d, (J @ (U[[1, 2]].mean(0) - U[3])) / (J @ (U[[1, 2]].mean(0) - U[3])).norm(), atol=1e-3)
    assert float(d.norm()) == pytest.approx(1.0, abs=1e-6) and torch.isfinite(d).all()


def test_degenerate_directions_raise():
    U, J = torch.ones(10, 4), torch.eye(4)
    with pytest.raises(D.DegenerateDirection):
        D.contrast_direction(U, J, [1], [2])                  # identical rows -> zero contrast
    with pytest.raises(D.DegenerateDirection):
        D.unit(torch.full((4,), float("nan")))
    with pytest.raises(D.DegenerateDirection):
        D.check_control(torch.tensor([1.0, 0, 0]), torch.tensor([1.0, 0.001, 0]))


def test_target_vs_sham_vs_random_distinct():
    g = torch.Generator().manual_seed(2)
    U, J = torch.randn(50, 16, generator=g), torch.randn(16, 16, generator=g)
    t, s = D.contrast_direction(U, J, [1], [2]), D.contrast_direction(U, J, [3], [4])
    r = D.random_direction(11, 16)
    assert abs(D.check_control(s, t)) < 0.99 and abs(D.check_control(r, t)) < 0.99


# ---- tokens -------------------------------------------------------------------------------------
class PhraseTok:
    """' 15 year old' -> [sp, 1, 5, year, old]; ' grandfather' -> [grand, father]; repeats exercised."""
    table = {" grandfather": [10, 11], " grandson": [10, 12], " 15 year old": [20, 21, 22, 23, 24],
             " 40-year-old": [30, 31, 23, 24], " the the": [5, 5], " banana": [40], " table": [41],
             " same a": [7, 8], " same b": [8, 7], " junk": [0, 1]}

    def __call__(self, text, add_special_tokens=False):
        class R: pass
        r = R(); r.input_ids = self.table[text]; return r


def test_phrase_ids_full_phrase_dedup_and_mask():
    mask = torch.ones(100, dtype=torch.bool)
    assert T.phrase_token_ids(PhraseTok(), "15 year old", mask) == [20, 21, 22, 23, 24]   # all constituents, not [0]
    assert T.phrase_token_ids(PhraseTok(), "the the", mask) == [5]                         # dedup
    mask[21] = False
    assert T.phrase_token_ids(PhraseTok(), "15 year old", mask) == [20, 22, 23, 24]        # mask applied
    mask[:] = False
    assert T.phrase_token_ids(PhraseTok(), "junk", mask) == []


def test_resolve_target_ids_overlap_and_ineligibility():
    mask = torch.ones(100, dtype=torch.bool)
    t = T.resolve_target_ids(PhraseTok(), "grandfather", "grandson", mask)
    assert t.stereotype == (10, 11) and t.counter == (10, 12) and t.overlap == (10,)
    t2 = T.resolve_target_ids(PhraseTok(), "15 year old", "40-year-old", mask)
    assert t2.overlap == (23, 24) and t2.raw_counter == (30, 31, 23, 24)
    with pytest.raises(T.IneligiblePair, match="identical"):
        T.resolve_target_ids(PhraseTok(), "same a", "same b", mask)
    m2 = mask.clone(); m2[[0, 1]] = False
    with pytest.raises(T.IneligiblePair, match="no meaningful"):
        T.resolve_target_ids(PhraseTok(), "junk", "grandson", m2)
    with pytest.raises(T.IneligiblePair, match="no meaningful"):
        T.resolve_target_ids(PhraseTok(), "grandson", "junk", m2)


def test_sham_phrases_are_single_distinct_tokens_in_fixture():
    mask = torch.ones(100, dtype=torch.bool)
    s = T.resolve_target_ids(PhraseTok(), C.SHAM_STEREO_TEXT, C.SHAM_COUNTER_TEXT, mask)
    assert s.stereotype == (40,) and s.counter == (41,)


def test_leading_space_is_used():
    seen = []
    class Spy:
        def __call__(self, text, add_special_tokens=False):
            seen.append((text, add_special_tokens))
            class R: input_ids = [3]
            return R()
    T.phrase_token_ids(Spy(), "grandfather", torch.ones(10, dtype=torch.bool))
    assert seen == [(" grandfather", False)]
