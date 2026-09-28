"""Orthogonal 2-D Haar operators used by the baseline.

The analysis LL uses the orthonormal convention: a 2x2 block is summed and
divided by two. Its adjoint repeats an LL coefficient over the block and also
divides by two. Consequently ``D @ U = I``.
"""

from __future__ import annotations

import torch


def haar_ll(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 2 or x.shape[-2] % 2 or x.shape[-1] % 2:
        raise ValueError(f"Haar LL needs even spatial dimensions, got {x.shape}")
    return (
        x[..., 0::2, 0::2]
        + x[..., 0::2, 1::2]
        + x[..., 1::2, 0::2]
        + x[..., 1::2, 1::2]
    ) * 0.5


def haar_lift_ll(x: torch.Tensor) -> torch.Tensor:
    return x.repeat_interleave(2, -2).repeat_interleave(2, -1) * 0.5


def haar_project_ll(x: torch.Tensor) -> torch.Tensor:
    return haar_lift_ll(haar_ll(x))


def replace_ll(child: torch.Tensor, target_ll: torch.Tensor) -> torch.Tensor:
    if haar_ll(child).shape != target_ll.shape:
        raise ValueError("target LL has the wrong shape")
    return child + haar_lift_ll(target_ll - haar_ll(child))


def common_ll_guidance(
    child_clean: torch.Tensor,
    parent_clean: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Align only the subspace common to parent and grandparent.

    In the frozen baseline, the outer LL of a child and the parent clean use
    the average-convention clean scale factor 2. The residual is then projected
    once more inside parent space, keeping parent-Nyquist and child detail free.
    """
    outer_ll = haar_ll(child_clean)
    residual = 2.0 * parent_clean - outer_ll
    return child_clean + float(alpha) * haar_lift_ll(haar_project_ll(residual))


def pack_latents(grid: torch.Tensor) -> torch.Tensor:
    if grid.ndim != 4:
        raise ValueError("expected [B,C,H,W]")
    b, c, h, w = grid.shape
    if h % 2 or w % 2:
        raise ValueError("latent grid dimensions must be even")
    return (
        grid.reshape(b, c, h // 2, 2, w // 2, 2)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(b, (h // 2) * (w // 2), c * 4)
    )


def unpack_latents(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if tokens.ndim != 3 or height % 2 or width % 2:
        raise ValueError("invalid packed latent shape")
    b, _, packed_c = tokens.shape
    if packed_c % 4:
        raise ValueError("packed channel dimension must be divisible by four")
    c = packed_c // 4
    return (
        tokens.reshape(b, height // 2, width // 2, c, 2, 2)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(b, c, height, width)
    )
