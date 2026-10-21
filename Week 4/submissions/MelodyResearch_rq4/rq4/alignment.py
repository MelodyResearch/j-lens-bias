"""Lens source-layer <-> model block-output alignment check (pure torch, backend-agnostic).

For each tested layer l the released lens readout `lens_logits[l]` must be reproducible from the
block-l output captured by our own hook (block l, NOT l-1 or l+1). We recompute the readout from
captured block outputs of l-1, l, l+1 under every candidate formula (the lens/unembed conventions
are not documented in this repository) and require that shift 0 wins clearly.
"""

from __future__ import annotations

from typing import Callable

import torch


class AlignmentError(RuntimeError):
    pass


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def check_lens_block_alignment(
    captured: dict[int, torch.Tensor],            # block index -> output at the source position [d]
    lens_logits: dict[int, torch.Tensor],         # source layer -> released-lens logits [V]
    formulas: dict[str, Callable[[int, torch.Tensor], torch.Tensor]],   # (layer, h) -> logits [V]
    layers: tuple[int, ...],
    *, tol: float = 0.05, ratio: float = 0.5,
) -> dict:
    report = {}
    for name, f in formulas.items():
        for shift in (-1, 0, 1):
            errs = [rel_err(f(l, captured[l + shift]), lens_logits[l]) for l in layers if (l + shift) in captured]
            if len(errs) == len(layers):
                report[(name, shift)] = sum(errs) / len(errs)
    if not report:
        raise AlignmentError("no formula/shift could be evaluated")
    (best_name, best_shift), best = min(report.items(), key=lambda kv: kv[1])
    others = [v for (n, s), v in report.items() if s != 0]
    if best_shift != 0:
        raise AlignmentError(f"best match is block shift {best_shift} ({best_name}, err {best:.4g}), not 0: {report}")
    if best > tol:
        raise AlignmentError(f"best shift-0 reconstruction error {best:.4g} > tol {tol}: {report}")
    if others and best > ratio * min(others):
        raise AlignmentError(f"shift 0 not clearly better than neighbours (err {best:.4g} vs {min(others):.4g})")
    return {"formula": best_name, "shift": 0, "rel_err": best,
            "all": {f"{n}|shift{s}": v for (n, s), v in report.items()}}
