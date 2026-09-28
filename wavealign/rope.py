"""Baseline FLUX RoPE: extrapolated text positions and NTK image axes."""

from __future__ import annotations

import torch
from torch import nn


def _rope_matrix(position: torch.Tensor, dim: int, theta: int, factor: float) -> torch.Tensor:
    if dim % 2:
        raise ValueError("RoPE axis dimension must be even")
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=position.device) / dim
    omega = 1.0 / ((theta * factor) ** scale)
    phase = torch.einsum("...n,d->...nd", position, omega)
    cosine, sine = torch.cos(phase), torch.sin(phase)
    return torch.stack((cosine, -sine, sine, cosine), dim=-1).view(
        position.shape[0], -1, dim // 2, 2, 2
    ).float()


class BaselineRotaryEmbedding(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int] | tuple[int, ...]):
        super().__init__()
        self.theta = theta
        self.axes_dim = tuple(axes_dim)
        self.ntk_factor = 1.0
        self.train_length = 1024

    def configure(self, *, ntk_factor: float, train_length: int = 1024) -> None:
        self.ntk_factor = float(ntk_factor)
        self.train_length = int(train_length)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.ndim == 2:
            ids = ids.unsqueeze(0)
        if ids.shape[-1] != len(self.axes_dim):
            raise ValueError("position IDs do not match RoPE axes")
        text = _rope_matrix(ids[..., 0], self.axes_dim[0], self.theta, 1.0)
        image_axes = []
        for axis in range(1, len(self.axes_dim)):
            length = ids[..., axis].shape[1]
            factor = 1.0 if length <= self.train_length else self.ntk_factor
            image_axes.append(_rope_matrix(ids[..., axis], self.axes_dim[axis], self.theta, factor))
        return torch.cat((text, *image_axes), dim=-3).unsqueeze(1)
