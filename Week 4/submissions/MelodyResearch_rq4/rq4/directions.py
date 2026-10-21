"""Intervention directions: J-Lens token-set contrast, seeded random controls, sham."""

from __future__ import annotations

import math

import torch


class DegenerateDirection(Exception):
    """The direction is zero, non-finite, or (for controls) collinear with the target."""


def unit(v: torch.Tensor, *, rel_eps: float = 1e-8, ref_norm: float | None = None) -> torch.Tensor:
    v = v.detach().double()
    n = float(v.norm())
    if not math.isfinite(n) or n == 0.0 or (ref_norm and n <= rel_eps * ref_norm):
        raise DegenerateDirection(f"norm {n!r} (ref {ref_norm!r})")
    return (v / n).float()


def contrast_direction(U: torch.Tensor, J: torch.Tensor, ids_a, ids_b) -> torch.Tensor:
    """Protocol: J_l.T @ (mean(U[ids_a]) - mean(U[ids_b])), unit-normalised.

    U: [V, d] unembedding rows. J: [d, d] released lens Jacobian for the source layer
    (the row-vector form used elsewhere is U[t] @ J, identical to J.T @ U[t]).
    Float32 on whatever device U/J live on; returns a float32 unit vector.
    """
    ia = torch.as_tensor(list(ids_a), dtype=torch.long, device=U.device)
    ib = torch.as_tensor(list(ids_b), dtype=torch.long, device=U.device)
    diff = U[ia].float().mean(0) - U[ib].float().mean(0)
    raw = J.float().T @ diff
    return unit(raw, ref_norm=float(diff.norm()))


def random_direction(seed: int, d: int) -> torch.Tensor:
    """Deterministic unit Gaussian direction on CPU. The same (seed, d) always gives the same
    vector, independent of prompt, layer and process; only the seed identifies the control."""
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return unit(torch.randn(d, generator=g, dtype=torch.float32))


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return float(a @ b / (a.norm() * b.norm()))


def check_control(control: torch.Tensor, target: torch.Tensor, *, max_abs_cos: float = 0.99) -> float:
    """Return cos(control, target); raise DegenerateDirection if the control is (nearly) the target."""
    c = cosine(control.cpu(), target.cpu())
    if not math.isfinite(c) or abs(c) > max_abs_cos:
        raise DegenerateDirection(f"control collinear with target (cos={c})")
    return c
