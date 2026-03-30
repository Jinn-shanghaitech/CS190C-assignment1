from __future__ import annotations

import numpy as np
import torch


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    dataset: 1D numpy array of token ids
    returns:
        x: (batch_size, context_length)
        y: (batch_size, context_length)
    """
    if dataset.ndim != 1:
        raise ValueError("dataset must be a 1D numpy array")
    if len(dataset) < context_length + 1:
        raise ValueError("dataset is too short for the requested context_length")

    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)

    x_list = []
    y_list = []
    for s in starts:
        x_list.append(dataset[s : s + context_length])
        y_list.append(dataset[s + 1 : s + context_length + 1])

    x = torch.tensor(np.stack(x_list), dtype=torch.long, device=device)
    y = torch.tensor(np.stack(y_list), dtype=torch.long, device=device)
    return x, y