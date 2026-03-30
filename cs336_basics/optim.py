from __future__ import annotations

import math
from typing import Iterable
import torch


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return

    total_sq_norm = torch.zeros((), device=grads[0].device, dtype=grads[0].dtype)
    for g in grads:
        total_sq_norm = total_sq_norm + torch.sum(g * g)

    total_norm = torch.sqrt(total_sq_norm)

    if total_norm <= max_l2_norm:
        return

    scale = max_l2_norm / (total_norm + 1e-6)
    for g in grads:
        g.mul_(scale)


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters

    if it > cosine_cycle_iters:
        return min_learning_rate

    decay_ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate")
        if eps < 0.0:
            raise ValueError("Invalid epsilon value")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta1")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta2")
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                step_t = state["step"]

                # Adam moments
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # bias correction
                bias_correction1 = 1.0 - beta1**step_t
                bias_correction2 = 1.0 - beta2**step_t

                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                # decoupled weight decay
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                denom = torch.sqrt(v_hat) + eps
                p.addcdiv_(m_hat, denom, value=-lr)

        return loss