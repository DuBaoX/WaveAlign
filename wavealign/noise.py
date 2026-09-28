"""Construction of the exact orthogonal shared-noise hierarchy."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib

import torch

from .wavelet import haar_ll, replace_ll

# Historical seed namespace bytes are fixed for exact old-noise reproducibility.
OLD_NOISE_NAMESPACE = bytes.fromhex("636f6e6a666c6f772d636f617273652d686161722d7631").decode("ascii")


def build_noise_hierarchy(
    shapes: Sequence[tuple[int, int, int, int]],
    *,
    generator: torch.Generator,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    coarsest: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Build the frozen shared-noise tree.

    Without an external coarsest tensor the baseline samples the finest 4K
    Gaussian once and restricts it by Haar LL. With an external 1K tensor it
    conditionally samples each new orthogonal detail fiber coarse-to-fine.
    """
    if not shapes:
        raise ValueError("at least one shape is required")
    if coarsest is None:
        finest = torch.randn(shapes[-1], generator=generator, device=device, dtype=dtype)
        hierarchy = [finest]
        for _ in range(len(shapes) - 1):
            hierarchy.append(haar_ll(hierarchy[-1]))
        hierarchy.reverse()
        if [tuple(value.shape) for value in hierarchy] != [tuple(shape) for shape in shapes]:
            raise ValueError("noise shapes are not a valid 2x pyramid")
        return hierarchy
    root = coarsest.to(device=device, dtype=dtype)
    if tuple(root.shape) != tuple(shapes[0]):
        raise ValueError("coarsest noise has the wrong shape")
    hierarchy = [root]
    for shape in shapes[1:]:
        child = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        if tuple(haar_ll(child).shape) != tuple(hierarchy[-1].shape):
            raise ValueError("adjacent noise shapes must differ by 2x spatially")
        hierarchy.append(replace_ll(child, hierarchy[-1]))
    return hierarchy


def build_paper_noise_hierarchy(
    shapes: Sequence[tuple[int, int, int, int]],
    *,
    seed: int,
    coarsest: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """CPU FP32 old-protocol root with independent, reproducible detail streams.

    The stream labels match the frozen old-noise protocol. Drawing a child
    never advances the root generator.
    """
    if not shapes:
        raise ValueError("at least one stage is required")
    if coarsest is None:
        root = torch.randn(
            shapes[0], generator=torch.Generator(device="cpu").manual_seed(seed),
            device="cpu", dtype=torch.float32,
        )
    else:
        if tuple(coarsest.shape) != tuple(shapes[0]) or not torch.isfinite(coarsest).all():
            raise ValueError("external root must have the expected shape and finite values")
        root = coarsest.to(device="cpu", dtype=torch.float32)
    result = [root]
    for level, shape in enumerate(shapes[1:], 1):
        label = f"{OLD_NOISE_NAMESPACE}:{seed}:detail:{level}"
        detail_seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little") % (2**63)
        detail = torch.randn(
            shape, generator=torch.Generator(device="cpu").manual_seed(detail_seed),
            device="cpu", dtype=torch.float32,
        )
        if tuple(haar_ll(detail).shape) != tuple(result[-1].shape):
            raise ValueError("adjacent noise shapes must differ by 2x spatially")
        result.append(replace_ll(detail, result[-1]))
    return result
