"""ScaleDiff-inspired Latent Frequency Mixing for the WaveAlign bridge.

Reference: ScaleDiff, arXiv:2510.25818, Eq. (3), and upstream FLUX refine().
This resize low-pass is not an orthogonal Haar projector or FFT filter.
"""
import math

import torch
import torch.nn.functional as F

from .wavelet import haar_ll, haar_lift_ll


def validate_lfm(strength: float, scale_factor: float) -> None:
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('lfm_strength must be finite and in [0, 1]')
    if not math.isfinite(scale_factor) or not 0 < scale_factor <= 1:
        raise ValueError('lfm_scale_factor must be finite and in (0, 1]')


def resize_lowpass(x: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """Antialiased bilinear downsample, then bilinear upsample (upstream)."""
    validate_lfm(1.0, scale_factor)
    h, w = x.shape[-2:]
    small = (int(h * scale_factor), int(w * scale_factor))
    if min(small) < 1:
        raise ValueError('LFM low-pass size must be at least 1x1')
    down = F.interpolate(x, size=small, mode='bilinear', align_corners=False, antialias=True)
    return F.interpolate(down, size=(h, w), mode='bilinear', align_corners=False)


def haar1_lowpass(x: torch.Tensor) -> torch.Tensor:
    """Orthogonal one-level LL projector U1 D1 in FP32."""
    if x.ndim != 4 or any(n % 2 or n < 2 for n in x.shape[-2:]):
        raise ValueError('One-level Haar-LFM requires BCHW spatial sizes divisible by 2')
    return haar_lift_ll(haar_ll(x.float()))


def haar2_lowpass(x: torch.Tensor) -> torch.Tensor:
    """Orthogonal two-level LL projector U1 U2 D2 D1 in FP32."""
    if x.ndim != 4 or any(n % 4 or n < 4 for n in x.shape[-2:]):
        raise ValueError('Two-level Haar-LFM requires BCHW spatial sizes divisible by 4')
    x = x.float()
    return haar_lift_ll(haar_lift_ll(haar_ll(haar_ll(x))))


def latent_frequency_mix(parent: torch.Tensor, pixel_lift: torch.Tensor, *,
                         strength: float = 1.0, scale_factor: float = 0.25,
                         filter_kind: str = 'resize') -> torch.Tensor:
    """Return RU + strength * (L(LU) - L(RU)), in model-scaled latent space.

    LU is a bicubic 2x interpolation of the SAME clean parent used for RU.
    No sampling, variance renormalization, Haar amplitude factor, or clipping.
    FP32 arithmetic supports rectangular latents and non-power-of-two sizes.
    The disabled branch is an exact identity and does not consume RNG state.
    """
    validate_lfm(strength, scale_factor)
    if filter_kind not in ('resize', 'haar1', 'haar2', 'haar_mid'):
        raise ValueError('lfm_filter must be resize, haar1, haar2 or haar_mid')
    if strength == 0:
        return pixel_lift
    if parent.ndim != 4 or pixel_lift.ndim != 4 or parent.shape[:2] != pixel_lift.shape[:2]:
        raise ValueError('LFM expects matching BCHW batch/channel dimensions')
    if pixel_lift.shape[-2:] != tuple(2 * n for n in parent.shape[-2:]):
        raise ValueError('LFM bridge requires adjacent 2x stages')
    if not parent.is_floating_point() or not pixel_lift.is_floating_point():
        raise ValueError('LFM requires floating-point model-scaled latents')
    if not torch.isfinite(parent).all() or not torch.isfinite(pixel_lift).all():
        raise ValueError('LFM received non-finite latents')
    parent = parent.to(device=pixel_lift.device, dtype=torch.float32)
    ru = pixel_lift.float()
    lu = F.interpolate(parent, size=ru.shape[-2:], mode='bicubic', align_corners=False)
    if filter_kind == 'haar_mid':
        delta = lu - ru
        return ru + strength * (haar1_lowpass(delta) - haar2_lowpass(delta))
    if filter_kind == 'haar1':
        return ru + strength * (haar1_lowpass(lu) - haar1_lowpass(ru))
    if filter_kind == 'haar2':
        return ru + strength * (haar2_lowpass(lu) - haar2_lowpass(ru))
    return ru + strength * (resize_lowpass(lu, scale_factor) - resize_lowpass(ru, scale_factor))
