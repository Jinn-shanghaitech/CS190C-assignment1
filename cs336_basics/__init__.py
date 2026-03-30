import importlib.metadata
from .bpe import BPETokenizer, train_bpe

__all__ = ["BPETokenizer", "train_bpe"]

__version__ = importlib.metadata.version("cs336_basics")
