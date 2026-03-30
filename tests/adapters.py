from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from cs336_basics.layers import Linear, Embedding, RMSNorm, SwiGLU, silu
from cs336_basics.attention import (
    softmax,
    scaled_dot_product_attention,
    RotaryPositionalEmbedding,
    MultiHeadSelfAttention,
)
from cs336_basics.transformer import TransformerBlock, TransformerLM
from cs336_basics.losses import cross_entropy
from cs336_basics.optim import gradient_clipping


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    assert weights.shape == (d_out, d_in)
    assert in_features.shape[-1] == d_in

    return in_features @ weights.T

def _adapter_softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = x - torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    assert weights.shape == (vocab_size, d_model)
    return weights[token_ids]

def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    assert w1_weight.shape == (d_ff, d_model)
    assert w2_weight.shape == (d_model, d_ff)
    assert w3_weight.shape == (d_ff, d_model)

    x1 = in_features @ w1_weight.T
    x3 = in_features @ w3_weight.T
    silu_x1 = x1 / (1 + torch.exp(-x1))
    gated = silu_x1 * x3
    return gated @ w2_weight.T


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    scores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    attn = _adapter_softmax(scores, dim=-1)
    return attn @ V

def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    seq_len = in_features.shape[-2]
    d_k_total = q_proj_weight.shape[0]
    d_v_total = v_proj_weight.shape[0]

    assert d_k_total % num_heads == 0
    assert d_v_total % num_heads == 0

    head_dim_qk = d_k_total // num_heads
    head_dim_v = d_v_total // num_heads

    q = in_features @ q_proj_weight.T
    k = in_features @ k_proj_weight.T
    v = in_features @ v_proj_weight.T

    q = q.reshape(*in_features.shape[:-2], seq_len, num_heads, head_dim_qk).transpose(-3, -2)
    k = k.reshape(*in_features.shape[:-2], seq_len, num_heads, head_dim_qk).transpose(-3, -2)
    v = v.reshape(*in_features.shape[:-2], seq_len, num_heads, head_dim_v).transpose(-3, -2)

    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=in_features.device)
    )

    attn_out = run_scaled_dot_product_attention(q, k, v, causal_mask)
    attn_out = attn_out.transpose(-3, -2).reshape(*in_features.shape[:-2], seq_len, d_v_total)

    return attn_out @ o_proj_weight.T


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    assert d_k % 2 == 0

    half = d_k // 2
    device = in_query_or_key.device
    dtype = in_query_or_key.dtype

    freq_exponents = torch.arange(half, device=device, dtype=dtype) * (2.0 / d_k)
    inv_freq = 1.0 / (theta ** freq_exponents)

    positions = token_positions.to(device=device, dtype=dtype)
    while positions.ndim < in_query_or_key.ndim - 1:
        positions = positions.unsqueeze(-2)

    angles = positions.unsqueeze(-1) * inv_freq
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    x_even = in_query_or_key[..., 0::2]
    x_odd = in_query_or_key[..., 1::2]

    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos

    out = torch.empty_like(in_query_or_key)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    seq_len = in_features.shape[-2]
    prefix_shape = in_features.shape[:-2]

    d_k_total = q_proj_weight.shape[0]
    d_v_total = v_proj_weight.shape[0]

    assert d_k_total % num_heads == 0
    assert d_v_total % num_heads == 0

    head_dim_qk = d_k_total // num_heads
    head_dim_v = d_v_total // num_heads

    q = in_features @ q_proj_weight.T
    k = in_features @ k_proj_weight.T
    v = in_features @ v_proj_weight.T

    q = q.reshape(*prefix_shape, seq_len, num_heads, head_dim_qk).transpose(-3, -2)
    k = k.reshape(*prefix_shape, seq_len, num_heads, head_dim_qk).transpose(-3, -2)
    v = v.reshape(*prefix_shape, seq_len, num_heads, head_dim_v).transpose(-3, -2)

    if token_positions is None:
        base = torch.arange(seq_len, device=in_features.device, dtype=torch.long)
        if len(prefix_shape) == 0:
            token_positions = base
        else:
            token_positions = base.view(*([1] * len(prefix_shape)), seq_len).expand(*prefix_shape, seq_len)

    q = run_rope(head_dim_qk, theta, max_seq_len, q, token_positions)
    k = run_rope(head_dim_qk, theta, max_seq_len, k, token_positions)

    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=in_features.device)
    )

    attn_out = run_scaled_dot_product_attention(q, k, v, causal_mask)
    attn_out = attn_out.transpose(-3, -2).reshape(*prefix_shape, seq_len, d_v_total)

    return attn_out @ o_proj_weight.T


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    x = in_features

    ln1_out = run_rmsnorm(
        d_model=d_model,
        eps=1e-5,
        weights=weights["ln1.weight"],
        in_features=x,
    )
    attn_out = run_multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        theta=theta,
        q_proj_weight=weights["attn.q_proj.weight"],
        k_proj_weight=weights["attn.k_proj.weight"],
        v_proj_weight=weights["attn.v_proj.weight"],
        o_proj_weight=weights["attn.output_proj.weight"],
        in_features=ln1_out,
    )
    x = x + attn_out

    ln2_out = run_rmsnorm(
        d_model=d_model,
        eps=1e-5,
        weights=weights["ln2.weight"],
        in_features=x,
    )
    ffn_out = run_swiglu(
        d_model=d_model,
        d_ff=d_ff,
        w1_weight=weights["ffn.w1.weight"],
        w2_weight=weights["ffn.w2.weight"],
        w3_weight=weights["ffn.w3.weight"],
        in_features=ln2_out,
    )
    x = x + ffn_out

    return x


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    x = run_embedding(
        vocab_size=vocab_size,
        d_model=d_model,
        weights=weights["token_embeddings.weight"],
        token_ids=in_indices,
    )

    for layer_idx in range(num_layers):
        layer_weights = {
            "attn.q_proj.weight": weights[f"layers.{layer_idx}.attn.q_proj.weight"],
            "attn.k_proj.weight": weights[f"layers.{layer_idx}.attn.k_proj.weight"],
            "attn.v_proj.weight": weights[f"layers.{layer_idx}.attn.v_proj.weight"],
            "attn.output_proj.weight": weights[f"layers.{layer_idx}.attn.output_proj.weight"],
            "ln1.weight": weights[f"layers.{layer_idx}.ln1.weight"],
            "ffn.w1.weight": weights[f"layers.{layer_idx}.ffn.w1.weight"],
            "ffn.w2.weight": weights[f"layers.{layer_idx}.ffn.w2.weight"],
            "ffn.w3.weight": weights[f"layers.{layer_idx}.ffn.w3.weight"],
            "ln2.weight": weights[f"layers.{layer_idx}.ln2.weight"],
        }
        x = run_transformer_block(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=context_length,
            theta=rope_theta,
            weights=layer_weights,
            in_features=x,
        )

    x = run_rmsnorm(
        d_model=d_model,
        eps=1e-5,
        weights=weights["ln_final.weight"],
        in_features=x,
    )
    logits = x @ weights["lm_head.weight"].T
    return logits



def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    assert in_features.shape[-1] == d_model
    assert weights.shape == (d_model,)

    rms = torch.sqrt(torch.mean(in_features * in_features, dim=-1, keepdim=True) + eps)
    normalized = in_features / rms
    return normalized * weights


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    return in_features / (1 + torch.exp(-in_features))


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    from cs336_basics.data import get_batch
    return get_batch(dataset, batch_size, context_length, device)


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    return _adapter_softmax(in_features, dim=dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    return cross_entropy(inputs, targets)


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    gradient_clipping(parameters, max_l2_norm)


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    from cs336_basics.optim import AdamW
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    from cs336_basics.optim import get_lr_cosine_schedule
    return get_lr_cosine_schedule(
        it=it,
        max_learning_rate=max_learning_rate,
        min_learning_rate=min_learning_rate,
        warmup_iters=warmup_iters,
        cosine_cycle_iters=cosine_cycle_iters,
    )


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    from cs336_basics.checkpoint import save_checkpoint
    return save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=iteration,
        out=out,
    )



def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    from cs336_basics.checkpoint import load_checkpoint
    return load_checkpoint(
        src=src,
        model=model,
        optimizer=optimizer,
    )

from cs336_basics.bpe import BPETokenizer, train_bpe

def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
):
    return BPETokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


def run_train_bpe(
    input_path,
    vocab_size,
    special_tokens,
    **kwargs,
):
    return train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        **kwargs,
    )