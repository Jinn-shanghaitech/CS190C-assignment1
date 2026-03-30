from __future__ import annotations

import torch
from torch import Tensor


def cross_entropy(
    inputs: Tensor,   # (batch_size, vocab_size)
    targets: Tensor,  # (batch_size,)
) -> Tensor:
    # stable CE: logsumexp - target_logit
    log_denom = torch.logsumexp(inputs, dim=-1)  # (B,)
    target_logits = inputs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # (B,)
    loss = log_denom - target_logits
    return loss.mean()