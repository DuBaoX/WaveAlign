"""Deterministic overlapping latent-tile scheduling and blending helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TileBox:
    top: int
    left: int
    height: int
    width: int


def _starts(length: int, tile: int, stride: int) -> list[int]:
    if tile >= length:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def tukey_1d(length: int, alpha: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    out = torch.ones(length, device=device, dtype=torch.float32)
    taper = min(max(1, int(round(alpha * length))), length // 2)
    if taper:
        ramp = 0.5 * (
            1.0
            - torch.cos(
                torch.pi
                * (torch.arange(taper, device=device, dtype=torch.float32) + 0.5)
                / taper
            )
        )
        out[:taper] = ramp
        out[-taper:] = ramp.flip(0)
    return out.to(dtype)


def blend_window(
    height: int, width: int, *, alpha: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return (
        tukey_1d(height, alpha, device=device, dtype=dtype)[:, None]
        * tukey_1d(width, alpha, device=device, dtype=dtype)[None, :]
    ).clamp_min(0.02)


def global_image_ids(height: int, width: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ids = torch.zeros(height, width, 3, device=device, dtype=dtype)
    ids[..., 1] = torch.arange(height, device=device, dtype=dtype)[:, None]
    ids[..., 2] = torch.arange(width, device=device, dtype=dtype)[None, :]
    return ids.reshape(1, height * width, 3)


def tile_image_ids(box: TileBox, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if box.top % 2 or box.left % 2 or box.height % 2 or box.width % 2:
        raise ValueError("FLUX packed tile geometry must be even")
    token_h, token_w = box.height // 2, box.width // 2
    ids = torch.zeros(token_h, token_w, 3, device=device, dtype=dtype)
    ids[..., 1] = torch.arange(
        box.top // 2, box.top // 2 + token_h, device=device, dtype=dtype
    )[:, None]
    ids[..., 2] = torch.arange(
        box.left // 2, box.left // 2 + token_w, device=device, dtype=dtype
    )[None, :]
    return ids.reshape(1, token_h * token_w, 3)


class TileScheduler:
    def __init__(self, tile: int, stride: int, *, seed: int):
        if not 0 < stride <= tile:
            raise ValueError("expected 0 < stride <= tile")
        self.tile = tile
        self.stride = stride
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def boxes(self, height: int, width: int, step: int) -> list[TileBox]:
        # A flexible-aspect canvas can be shorter than the nominal square tile
        # along one axis (for example the 2K stage of a 21:9 image).  In that
        # case the tile spans the complete short axis.  Keeping the nominal
        # size here used to make ``tile_image_ids`` longer than the sliced
        # tensor and caused a RoPE/query length mismatch.
        tile_height = min(height, self.tile)
        tile_width = min(width, self.tile)
        ys = _starts(height, tile_height, self.stride)
        xs = _starts(width, tile_width, self.stride)

        def legal_shift(origins: list[int], length: int, tile: int) -> int:
            max_shift = max(2, self.stride)
            low = max(-min(origins), -max_shift)
            high = min(length - tile - max(origins), max_shift)
            low_even = low + (1 if low % 2 else 0)
            high_even = high - (1 if high % 2 else 0)
            if low_even > high_even:
                return 0
            return 2 * int(
                torch.randint(low_even // 2, high_even // 2 + 1, (1,), generator=self.generator)
            )

        shift_y = legal_shift(ys, height, tile_height)
        shift_x = legal_shift(xs, width, tile_width)
        return [
            TileBox(y + shift_y, x + shift_x, tile_height, tile_width)
            for y in ys
            for x in xs
        ]
