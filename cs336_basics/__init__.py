import importlib.metadata
from .bpe import BPETokenizer, train_bpe
from .layers import Linear, Embedding, RMSNorm, SwiGLU, silu
from .attention import (
    softmax,
    scaled_dot_product_attention,
    RotaryPositionalEmbedding,
    MultiHeadSelfAttention,
)
from .losses import cross_entropy
from .optim import gradient_clipping
from .transformer import TransformerBlock, TransformerLM

__all__ = [
    "Linear",
    "Embedding",
    "RMSNorm",
    "SwiGLU",
    "silu",
    "softmax",
    "scaled_dot_product_attention",
    "RotaryPositionalEmbedding",
    "MultiHeadSelfAttention",
    "cross_entropy",
    "gradient_clipping",
    "TransformerBlock",
    "TransformerLM",
]

__version__ = importlib.metadata.version("cs336_basics")
