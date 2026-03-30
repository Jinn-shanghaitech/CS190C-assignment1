from __future__ import annotations

import math
import torch
from torch import Tensor
from torch.nn import Module, Parameter


def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class Linear(Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.weight = Parameter(torch.empty(d_out, d_in))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = math.sqrt(6.0 / (self.d_in + self.d_out))
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., d_in)
        # weight: (d_out, d_in)
        return x @ self.weight.transpose(-1, -2)


class Embedding(Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weight = Parameter(torch.empty(vocab_size, d_model))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight.normal_(mean=0.0, std=1.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


class RMSNorm(Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., d_model)
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class SwiGLU(Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))