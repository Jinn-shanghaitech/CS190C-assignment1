from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Module

from .layers import Embedding, RMSNorm, SwiGLU, Linear
from .attention import MultiHeadSelfAttention


class ModuleListLite(Module):
    """
    A tiny replacement for torch.nn.ModuleList.
    It keeps state_dict keys like layers.0.*, layers.1.*, ...
    """
    def __init__(self, modules: list[Module]):
        super().__init__()
        self._module_list: list[Module] = []
        for i, module in enumerate(modules):
            self.add_module(str(i), module)
            self._module_list.append(module)

    def __iter__(self):
        return iter(self._module_list)

    def __getitem__(self, idx: int) -> Module:
        return self._module_list[idx]

    def __len__(self) -> int:
        return len(self._module_list)


class TransformerBlock(Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            use_rope=True,
        )
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.ln1(x), token_positions=token_positions)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = ModuleListLite(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, in_indices: Tensor) -> Tensor:
        # in_indices: (batch, seq_len)
        batch_size, seq_len = in_indices.shape
        if seq_len > self.context_length:
            raise ValueError("Input length exceeds context_length.")

        x = self.token_embeddings(in_indices)
        token_positions = torch.arange(seq_len, device=in_indices.device).unsqueeze(0).expand(batch_size, seq_len)

        for layer in self.layers:
            x = layer(x, token_positions=token_positions)

        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits