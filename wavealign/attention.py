"""Only the two attention processors exercised by the frozen baseline."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def apply_rope(query: torch.Tensor, key: torch.Tensor, frequencies: torch.Tensor):
    query_matrix = query.float().reshape(*query.shape[:-1], -1, 1, 2)
    key_matrix = key.float().reshape(*key.shape[:-1], -1, 1, 2)
    query_out = frequencies[..., 0] * query_matrix[..., 0] + frequencies[..., 1] * query_matrix[..., 1]
    key_out = frequencies[..., 0] * key_matrix[..., 0] + frequencies[..., 1] * key_matrix[..., 1]
    return query_out.reshape_as(query).type_as(query), key_out.reshape_as(key).type_as(key)


def _attention_scale(sequence_length: int, head_dim: int, train_length: int, enabled: bool) -> float:
    if not enabled:
        return math.sqrt(1.0 / head_dim)
    return math.sqrt(math.log(sequence_length, train_length) / head_dim)


class _RuntimeProcessor:
    proportional_attention: bool = True
    sequence_length_override: int | None = None

    def configure(self, *, proportional_attention: bool, sequence_length_override: int | None = None) -> None:
        self.proportional_attention = proportional_attention
        self.sequence_length_override = sequence_length_override

    def effective_length(self, actual: int) -> int:
        return self.sequence_length_override or actual


class BaselineFluxSingleAttnProcessor(_RuntimeProcessor):
    """Single-stream SDPA with the historical proportional-temperature base."""

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size = hidden_states.shape[0]
        encoder_hidden_states = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        if image_rotary_emb is not None:
            query, key = apply_rope(query, key, image_rotary_emb)
        scale = _attention_scale(
            self.effective_length(key.shape[2]), head_dim, 64**2 + 512, self.proportional_attention
        )
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False, scale=scale
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, inner_dim).to(query.dtype)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        return hidden_states


class BaselineFluxAttnProcessor(_RuntimeProcessor):
    """Joint text-image SDPA used by FLUX double-stream blocks."""

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        **_: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        context_ndim = encoder_hidden_states.ndim
        if context_ndim == 4:
            context_batch, context_channel, context_height, context_width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(
                context_batch, context_channel, context_height * context_width
            ).transpose(1, 2)
        batch_size = encoder_hidden_states.shape[0]
        query, key, value = attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        def heads(x: torch.Tensor) -> torch.Tensor:
            return x.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        query, key, value = heads(query), heads(key), heads(value)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        context_query = heads(attn.add_q_proj(encoder_hidden_states))
        context_key = heads(attn.add_k_proj(encoder_hidden_states))
        context_value = heads(attn.add_v_proj(encoder_hidden_states))
        if attn.norm_added_q is not None:
            context_query = attn.norm_added_q(context_query)
        if attn.norm_added_k is not None:
            context_key = attn.norm_added_k(context_key)
        query = torch.cat((context_query, query), dim=2)
        key = torch.cat((context_key, key), dim=2)
        value = torch.cat((context_value, value), dim=2)
        if image_rotary_emb is not None:
            query, key = apply_rope(query, key, image_rotary_emb)
        scale = _attention_scale(
            self.effective_length(key.shape[2]), head_dim, 128**2 + 512, self.proportional_attention
        )
        output = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False, scale=scale)
        output = output.transpose(1, 2).reshape(batch_size, -1, inner_dim).to(query.dtype)
        text_length = encoder_hidden_states.shape[1]
        context_output, hidden_output = output[:, :text_length], output[:, text_length:]
        hidden_output = attn.to_out[1](attn.to_out[0](hidden_output))
        context_output = attn.to_add_out(context_output)
        if input_ndim == 4:
            hidden_output = hidden_output.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_ndim == 4:
            context_output = context_output.transpose(-1, -2).reshape(
                context_batch, context_channel, context_height, context_width
            )
        return hidden_output, context_output
