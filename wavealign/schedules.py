"""Scale-conjugate sigma grids and baseline alpha scheduling."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch


def logit(value: float) -> float:
    value = min(max(float(value), 1e-12), 1.0 - 1e-12)
    return math.log(value / (1.0 - value))


def shifted_sigma(raw_sigma: float, shift: float) -> float:
    return shift * raw_sigma / (1.0 + (shift - 1.0) * raw_sigma)


def inverse_shifted_sigma(sigma: float, shift: float) -> float:
    return sigma / (shift - (shift - 1.0) * sigma)


def reference_raw_grid(reference_steps: int) -> np.ndarray:
    return np.linspace(1.0, 1.0 / reference_steps, reference_steps, dtype=np.float64)


def retained_tail_entry_sigma(shift: float, retained_steps: int, reference_steps: int) -> float:
    """Resolve the shifted entry sigma for the last K nodes of an R-node raw grid."""
    if not 1 <= retained_steps <= reference_steps:
        raise ValueError("retained_steps must be within the reference grid")
    return shifted_sigma(retained_steps / reference_steps, shift)


def fixed_entry_tail_grid(entry_sigma: float, shift: float, steps: int, reference_steps: int) -> np.ndarray:
    raw_entry = inverse_shifted_sigma(entry_sigma, shift)
    raw_final = 1.0 / reference_steps
    return np.linspace(raw_entry, raw_final, steps, dtype=np.float64)


def alpha_sigma_linear(sigma: float, lock: float, free: float, cap: float = 1.0) -> float:
    if sigma >= lock:
        return float(cap)
    if sigma <= free:
        return 0.0
    return float(cap) * (sigma - free) / (lock - free)


def alpha_sigma_cosine(sigma: float, lock: float, free: float, cap: float = 1.0) -> float:
    """Smooth release from lock to free, as used by the WaveAlign demo."""
    if not all(math.isfinite(v) for v in (sigma, lock, free, cap)) or not lock > free or not 0 <= cap <= 1:
        raise ValueError("invalid cosine release parameters")
    if sigma >= lock:
        return float(cap)
    if sigma <= free:
        return 0.0
    return float(cap) * 0.5 * (1.0 - math.cos(math.pi * (sigma - free) / (lock - free)))


def scheduler_phase_lead(parent_shift: float, child_shift: float) -> float:
    return math.log(child_shift / parent_shift)


def parent_sigma_at_child_sigma(child_sigma: float, lead_logit: float) -> float:
    parent_logit = logit(child_sigma) - lead_logit
    return 1.0 / (1.0 + math.exp(-parent_logit))


def interpolate_logit_trajectory(
    query_sigma: float,
    sigmas: Sequence[float],
    values: Sequence[torch.Tensor],
) -> torch.Tensor:
    if len(sigmas) != len(values) or not sigmas:
        raise ValueError("trajectory sigma/value lengths must match and be nonempty")
    positions = [logit(value) for value in sigmas]
    query = logit(query_sigma)
    if query >= positions[0]:
        return values[0]
    if query <= positions[-1]:
        return values[-1]
    for sigma, value in zip(sigmas, values):
        if abs(query_sigma - float(sigma)) <= 2e-6 * max(query_sigma, float(sigma), 1e-8):
            return value
    for index in range(len(positions) - 1):
        left, right = positions[index], positions[index + 1]
        if left >= query >= right:
            if not math.isfinite(left):
                weight = (float(sigmas[index]) - query_sigma) / (
                    float(sigmas[index]) - float(sigmas[index + 1])
                )
            else:
                weight = (left - query) / (left - right)
            return values[index].lerp(values[index + 1], weight)
    raise RuntimeError("unreachable interpolation interval")
