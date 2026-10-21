import pytest
import torch

from rq4 import config as C
from rq4.hooks import (CaptureHook, DisplacementCapError, Edit, HookAlignmentError, InterventionHook, check_cap,
                       compute_delta)
from toy import D, ToyLM


def unit_dir(seed=0):
    return torch.nn.functional.normalize(torch.randn(D, generator=torch.Generator().manual_seed(seed)), dim=0)


def run_with(model, ids, edit):
    with InterventionHook(model.layers[edit.layer], edit) as hk, torch.no_grad():
        out = model(ids)
    return out, hk.record


@pytest.mark.parametrize("tuple_out", [False, True])
def test_alpha_zero_is_identity(tuple_out):
    m, ids = ToyLM(tuple_out=tuple_out), torch.arange(1, 9).view(1, -1)
    base = m(ids)
    for kind, mag in (("ablate", None), ("norm_matched", 0.3)):
        out, rec = run_with(m, ids, Edit(8, 7, unit_dir(), kind, magnitude=mag, alpha=0.0))
        assert torch.equal(out, base) and rec.fired == 1 and rec.measured_displacement == 0.0


@pytest.mark.parametrize("tuple_out", [False, True])
def test_edits_only_intended_layer_and_position(tuple_out):
    m, ids = ToyLM(tuple_out=tuple_out), torch.arange(1, 9).view(1, -1)
    pos, layer = 5, 8
    cap_hooks = {l: CaptureHook(m.layers[l], pos) for l in (7, 8, 9)}
    # whole-sequence block outputs before/after, captured by raw hooks
    def grab(edit):
        store, hs = {}, []
        ctx = InterventionHook(m.layers[edit.layer], edit) if edit else None
        if ctx: ctx.__enter__()                       # intervention registered first -> raw hooks see edited output
        try:
            hs = [m.layers[l].register_forward_hook(lambda mod, i, o, l=l: store.__setitem__(l, (o[0] if isinstance(o, tuple) else o).detach().clone())) for l in (7, 8, 9)]
            with torch.no_grad(): m(ids)
        finally:
            for h in hs: h.remove()
            if ctx: ctx.__exit__(None, None, None)
        return store
    clean, edited = grab(None), grab(Edit(layer, pos, unit_dir(), "ablate", cap=1.0))
    assert torch.equal(clean[7], edited[7])                                   # upstream layer untouched
    diff8 = (clean[8] != edited[8]).any(-1)[0]                                # positions changed at the edited block
    assert diff8.nonzero().flatten().tolist() == [pos]
    assert (clean[9] != edited[9]).any(-1)[0, :pos].sum() == 0               # causal: earlier positions unaffected
    assert (clean[9] != edited[9]).any(-1)[0, pos:].all()                    # later layer sees it from pos onward
    del cap_hooks


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_shapes_and_dtypes_preserved(dtype):
    m, ids = ToyLM(dtype=dtype), torch.arange(1, 7).view(1, -1)
    seen = {}
    h0 = m.layers[16].register_forward_hook(lambda mod, i, o: seen.__setitem__("pre", (o.shape, o.dtype)))
    out, rec = run_with(m, ids, Edit(16, 5, unit_dir(), "ablate", cap=1.0))
    h0.remove()
    assert seen["pre"] == (torch.Size([1, 6, D]), dtype) and rec.dtype == str(dtype)
    assert out.dtype == dtype and torch.isfinite(out.float()).all()


def test_no_in_place_mutation_of_block_output():
    m, ids = ToyLM(), torch.arange(1, 9).view(1, -1)
    keep = {}
    def first(mod, i, o):                                   # registered BEFORE the intervention hook
        keep["ref"], keep["copy"] = o, o.detach().clone()
    h = m.layers[8].register_forward_hook(first)
    run_with(m, ids, Edit(8, 7, unit_dir(), "ablate", cap=1.0))
    h.remove()
    assert torch.equal(keep["ref"], keep["copy"])           # original tensor object never modified


def test_ablation_removes_projection_and_reports_displacement():
    m, ids = ToyLM(), torch.arange(1, 9).view(1, -1)
    u = unit_dir()
    with CaptureHook(m.layers[8], 7) as cap, torch.no_grad():
        m(ids)
    h = cap.value
    delta = compute_delta(h, Edit(8, 7, u, "ablate", cap=1.0))
    assert abs(float((h + delta) @ u)) < 1e-5
    _, rec = run_with(m, ids, Edit(8, 7, u, "ablate", cap=1.0))
    assert rec.intended_displacement == pytest.approx(abs(float(h @ u)), rel=1e-5)
    assert rec.measured_displacement == pytest.approx(rec.intended_displacement, rel=1e-4)


def test_norm_matched_has_exact_requested_norm_and_sign():
    h, u = torch.randn(D), unit_dir(3)
    for mag in (0.01, 0.5):
        d = compute_delta(h, Edit(1, 0, u, "norm_matched", magnitude=mag))
        assert float(d.norm()) == pytest.approx(mag, rel=1e-5)
        assert float((h + d) @ u) * float(h @ u) >= 0 or abs(float(h @ u)) < mag    # moves toward removing the component
    assert float(compute_delta(h, Edit(1, 0, u, "norm_matched", magnitude=0.5)) @ u) * float(h @ u) < 0


def test_prefill_only_second_call_untouched():
    m = ToyLM()
    e = Edit(8, 7, unit_dir(), "ablate", cap=1.0)
    with InterventionHook(m.layers[8], e) as hk, torch.no_grad():
        a = m(torch.arange(1, 9).view(1, -1))
        n1 = hk.record.measured_displacement
        b = m(torch.arange(1, 9).view(1, -1))            # second forward: no further edit
    assert hk.record.fired == 1 and hk.record.measured_displacement == n1
    assert not torch.equal(a, b)                          # edited vs. clean second call
    assert torch.equal(b, m(torch.arange(1, 9).view(1, -1)))


def test_hooks_removed_after_exception_in_body_and_in_hook():
    m, ids = ToyLM(), torch.arange(1, 9).view(1, -1)
    with pytest.raises(RuntimeError, match="boom"):
        with InterventionHook(m.layers[8], Edit(8, 7, unit_dir(), "ablate", cap=1.0)):
            raise RuntimeError("boom")
    assert len(m.layers[8]._forward_hooks) == 0
    with pytest.raises(HookAlignmentError):                # hook itself raises (position beyond the sequence)
        with InterventionHook(m.layers[8], Edit(8, 99, unit_dir(), "ablate")):
            m(ids)
    assert len(m.layers[8]._forward_hooks) == 0
    with pytest.raises(RuntimeError):
        with CaptureHook(m.layers[8], 0):
            raise RuntimeError("x")
    assert len(m.layers[8]._forward_hooks) == 0


def test_displacement_cap_enforced():
    h = torch.ones(D) * 3
    big = Edit(8, 0, unit_dir(), "norm_matched", magnitude=0.5 * float(h.norm()))
    with pytest.raises(DisplacementCapError):
        check_cap(h, compute_delta(h, big), C.DISPLACEMENT_CAP)
    m, ids = ToyLM(), torch.arange(1, 9).view(1, -1)
    with pytest.raises(DisplacementCapError):
        run_with(m, ids, Edit(8, 7, unit_dir(), "norm_matched", magnitude=1e6))
    assert len(m.layers[8]._forward_hooks) == 0


def test_nonfinite_activation_is_rejected():
    h = torch.full((D,), float("nan"))
    with pytest.raises(FloatingPointError):
        compute_delta(h, Edit(1, 0, unit_dir(), "ablate"))
