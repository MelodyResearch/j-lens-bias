import pytest
import torch

from rq4.alignment import AlignmentError, check_lens_block_alignment

D, V = 12, 40
g = torch.Generator().manual_seed(0)
U = torch.randn(V, D, generator=g)
J = {l: torch.randn(D, D, generator=g) for l in range(0, 24)}
CAP = {b: torch.randn(D, generator=g) for b in range(0, 24)}
FORMULAS = {"h@J@U.T": lambda l, h: (h @ J[l]) @ U.T, "h@J.T@U.T": lambda l, h: (h @ J[l].T) @ U.T}
LAYERS = (8, 16, 22)


def test_alignment_passes_when_lens_reads_block_l():
    lens = {l: (CAP[l] @ J[l].T) @ U.T for l in LAYERS}
    rep = check_lens_block_alignment(CAP, lens, FORMULAS, LAYERS)
    assert rep["shift"] == 0 and rep["formula"] == "h@J.T@U.T" and rep["rel_err"] < 1e-5


@pytest.mark.parametrize("shift", [-1, 1])
def test_alignment_detects_off_by_one(shift):
    lens = {l: (CAP[l + shift] @ J[l]) @ U.T for l in LAYERS}
    with pytest.raises(AlignmentError):
        check_lens_block_alignment(CAP, lens, FORMULAS, LAYERS)


def test_alignment_rejects_unexplained_readout():
    lens = {l: torch.randn(V, generator=g) for l in LAYERS}
    with pytest.raises(AlignmentError):
        check_lens_block_alignment(CAP, lens, FORMULAS, LAYERS)


def test_model_lens_dimension_checks():
    from rq4 import preflight as P
    base = {"source_layers": list(range(23)), "d_model": 8, "model_d_model": 8, "model_n_layers": 24}
    P.check_model_and_lens(base)
    for bad in ({"source_layers": list(range(22))}, {"source_layers": list(range(24))}, {"model_d_model": 9}, {"model_n_layers": 22}):
        with pytest.raises(AssertionError):
            P.check_model_and_lens({**base, **bad})
