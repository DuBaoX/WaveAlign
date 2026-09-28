"""Configuration-safe installation of the baseline FLUX runtime."""

from __future__ import annotations

from pathlib import Path

import torch
from diffusers import FluxTransformer2DModel

from .attention import BaselineFluxAttnProcessor, BaselineFluxSingleAttnProcessor
from .rope import BaselineRotaryEmbedding


def install_baseline_runtime(transformer: FluxTransformer2DModel) -> FluxTransformer2DModel:
    """Install parameter-free baseline components without changing state keys."""
    transformer.pos_embed = BaselineRotaryEmbedding(
        theta=10_000, axes_dim=transformer.config.axes_dims_rope
    )
    for block in transformer.transformer_blocks:
        block.attn.set_processor(BaselineFluxAttnProcessor())
    for block in transformer.single_transformer_blocks:
        block.attn.set_processor(BaselineFluxSingleAttnProcessor())
    return transformer


def load_baseline_transformer(model_path: Path, dtype: torch.dtype) -> FluxTransformer2DModel:
    transformer = FluxTransformer2DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=dtype
    )
    return install_baseline_runtime(transformer)


def configure_transformer_runtime(
    transformer: FluxTransformer2DModel,
    *,
    ntk_factor: float,
    rope_train_length: int,
    proportional_attention: bool,
    proportional_sequence_length: int | None = None,
) -> None:
    transformer.pos_embed.configure(ntk_factor=ntk_factor, train_length=rope_train_length)
    for block in transformer.transformer_blocks:
        block.attn.processor.configure(
            proportional_attention=proportional_attention,
            sequence_length_override=proportional_sequence_length,
        )
    for block in transformer.single_transformer_blocks:
        block.attn.processor.configure(
            proportional_attention=proportional_attention,
            sequence_length_override=proportional_sequence_length,
        )
