"""Safe prefill-only single-block intervention hook.

The edit is applied once, at one sequence position of one transformer block's output, on
the first forward call (the prompt pass). Later calls (KV-cached generation steps) pass
through untouched. The block output is cloned, never mutated in place, and the hook is
always removed in `finally` (context manager).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from . import config as C


class HookAlignmentError(RuntimeError):
    pass


class DisplacementCapError(RuntimeError):
    def __init__(self, message: str, ratio: float | None = None):
        super().__init__(message)
        self.ratio = ratio      # the measured ratio that breached the cap (None if not computed)


@dataclass(frozen=True)
class Edit:
    layer: int
    position: int
    direction: torch.Tensor          # unit vector [d]
    kind: str                        # "ablate" (clean projection removal) | "norm_matched"
    magnitude: float | None = None   # required for "norm_matched": target displacement norm
    alpha: float = C.ALPHA
    cap: float = C.DISPLACEMENT_CAP


@dataclass
class HookRecord:
    fired: int = 0
    layer: int | None = None
    position: int | None = None
    seq_len: int | None = None
    dtype: str | None = None
    activation_norm: float | None = None
    projection: float | None = None
    intended_displacement: float | None = None
    measured_displacement: float | None = None


def compute_delta(h: torch.Tensor, edit: Edit) -> torch.Tensor:
    """Float32 displacement for activation h [d]. alpha scales it; alpha == 0 gives exactly zero."""
    h = h.detach().float()
    u = edit.direction.to(h.device).float()
    proj = h @ u
    if edit.kind == "ablate":
        delta = -proj * u
    elif edit.kind == "norm_matched":
        if edit.magnitude is None:
            raise ValueError("norm_matched edit needs a magnitude")
        sign = 1.0 if float(proj) >= 0.0 else -1.0
        delta = -sign * float(edit.magnitude) * u
    else:
        raise ValueError(f"unknown edit kind {edit.kind!r}")
    delta = edit.alpha * delta
    if not bool(torch.isfinite(delta).all()):
        raise FloatingPointError("non-finite displacement")
    return delta


def displacement_ratio(h: torch.Tensor, delta: torch.Tensor) -> float:
    return float(delta.double().norm() / h.double().norm())


def check_cap(h: torch.Tensor, delta: torch.Tensor, cap: float, *, rtol: float = 1e-6) -> float:
    ratio = displacement_ratio(h, delta)
    if not math.isfinite(ratio) or ratio > cap * (1 + rtol):
        raise DisplacementCapError(f"displacement ratio {ratio:.5f} exceeds cap {cap}", ratio)
    return ratio


class InterventionHook:
    """Context manager: registers a forward hook on `module` (a block), removes it on exit."""

    def __init__(self, module: torch.nn.Module, edit: Edit):
        self.module, self.edit = module, edit
        self.record = HookRecord(layer=edit.layer, position=edit.position)
        self._handle = None

    def _hook(self, module, inputs, output):
        if self.record.fired:                      # prefill only
            return None
        if torch.is_tensor(output):
            h, rest = output, None
        elif isinstance(output, tuple) and output and torch.is_tensor(output[0]):
            h, rest = output[0], output[1:]
        else:
            raise HookAlignmentError(f"unsupported block output type {type(output)!r}")
        pos = self.edit.position
        if h.dim() != 3 or h.shape[0] != 1 or h.shape[1] <= pos:
            raise HookAlignmentError(f"first call has shape {tuple(h.shape)}, cannot edit position {pos}")
        vec = h[0, pos].detach().float()
        delta = compute_delta(vec, self.edit)
        check_cap(vec, delta, self.edit.cap)
        new = h.clone()                             # never mutate the original block output
        new[0, pos] = (vec + delta).to(h.dtype)
        if not bool(torch.isfinite(new[0, pos]).all()):
            raise FloatingPointError("non-finite edited activation")
        measured = (new[0, pos].detach().float() - vec).norm()
        # bf16 rounding can push the realised displacement past the cap the float32 delta satisfied: fail closed.
        check_cap(vec, new[0, pos].detach().float() - vec, self.edit.cap)
        rec = self.record
        rec.fired, rec.seq_len, rec.dtype = 1, int(h.shape[1]), str(h.dtype)
        rec.activation_norm = float(vec.double().norm())
        rec.projection = float(vec @ self.edit.direction.to(vec.device).float())
        rec.intended_displacement = float(delta.double().norm())
        rec.measured_displacement = float(measured)
        return new if rest is None else (new,) + tuple(rest)

    def __enter__(self) -> "InterventionHook":
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False


class CaptureHook:
    """Read-only capture of the clean block output at one position (detached float32 clone)."""

    def __init__(self, module: torch.nn.Module, position: int):
        self.module, self.position, self.value = module, position, None
        self._handle = None

    def _hook(self, module, inputs, output):
        h = output if torch.is_tensor(output) else output[0]
        if self.value is None:
            if h.shape[1] <= self.position:
                raise HookAlignmentError(f"capture: seq {h.shape[1]} <= position {self.position}")
            self.value = h[0, self.position].detach().float().clone()
        return None

    def __enter__(self):
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False
