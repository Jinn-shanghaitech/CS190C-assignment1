from __future__ import annotations

import math
import torch
from torch import Tensor
from torch.nn import Module

from .layers import Linear


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_stable = x - x_max
    exp_x = torch.exp(x_stable)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    # Q: (..., queries, d_k)
    # K: (..., keys, d_k)
    # V: (..., keys, d_v)
    d_k = Q.shape[-1]
    scores = (Q @ K.transpose(-1, -2)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    attn = softmax(scores, dim=-1)
    return attn @ V


class RotaryPositionalEmbedding(Module):
    def __init__(self, d_k: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("RoPE requires d_k to be even.")
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.theta = theta

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        d_half = self.d_k // 2
        device = x.device
        dtype = x.dtype

        idx = torch.arange(d_half, device=device, dtype=dtype)
        inv_freq = 1.0 / (self.theta ** (idx / d_half))

        pos = token_positions.to(device=device, dtype=dtype)
        while pos.ndim < x.ndim - 1:
            pos = pos.unsqueeze(0)

        freqs = pos.unsqueeze(-1) * inv_freq
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        out = torch.empty_like(x)
        out[..., 0::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_even * sin + x_odd * cos
        return out


class MultiHeadSelfAttention(Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        use_rope: bool = False,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_rope = use_rope

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

        if use_rope:
            if max_seq_len is None or theta is None:
                raise ValueError("max_seq_len and theta are required when use_rope=True.")
            self.rope = RotaryPositionalEmbedding(
                d_k=self.head_dim,
                max_seq_len=max_seq_len,
                theta=theta,
            )
        else:
            self.rope = None

    def _split_heads(self, x: Tensor) -> Tensor:
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)

    def _merge_heads(self, x: Tensor) -> Tensor:
        # x: (batch, num_heads, seq_len, head_dim)
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        x: Tensor,
        token_positions: Tensor | None = None,
    ) -> Tensor:
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, _ = x.shape

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        if self.use_rope:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        ).unsqueeze(0).unsqueeze(0)

        out = scaled_dot_product_attention(q, k, v, mask=causal_mask)
        out = self._merge_heads(out)
        return self.output_proj(out)