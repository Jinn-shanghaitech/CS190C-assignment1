from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import regex as re


# GPT-2 style pretokenization pattern
DEFAULT_PRETOKENIZER_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+|"""
    r""" ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def _pretokenize_text(text: str, pattern: str = DEFAULT_PRETOKENIZER_PATTERN) -> list[str]:
    return re.findall(pattern, text)


def _bytes_to_tuple(token_bytes: bytes) -> tuple[bytes, ...]:
    return tuple(bytes([b]) for b in token_bytes)


def _count_adjacent_pairs(
    word_counts: Counter[tuple[bytes, ...]],
) -> Counter[tuple[bytes, bytes]]:
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    for word, freq in word_counts.items():
        if len(word) < 2:
            continue
        for i in range(len(word) - 1):
            pair_counts[(word[i], word[i + 1])] += freq
    return pair_counts


def _merge_pair_in_word(
    word: tuple[bytes, ...],
    pair: tuple[bytes, bytes],
) -> tuple[bytes, ...]:
    if len(word) < 2:
        return word

    a, b = pair
    merged = a + b
    out: list[bytes] = []
    i = 0
    n = len(word)

    while i < n:
        if i < n - 1 and word[i] == a and word[i + 1] == b:
            out.append(merged)
            i += 2
        else:
            out.append(word[i])
            i += 1

    return tuple(out)


def _merge_pair_in_word_counts(
    word_counts: Counter[tuple[bytes, ...]],
    pair: tuple[bytes, bytes],
) -> Counter[tuple[bytes, ...]]:
    new_counts: Counter[tuple[bytes, ...]] = Counter()
    for word, freq in word_counts.items():
        new_word = _merge_pair_in_word(word, pair)
        new_counts[new_word] += freq
    return new_counts


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    pattern: str = DEFAULT_PRETOKENIZER_PATTERN,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer.

    Returns:
        vocab: mapping token_id -> token_bytes
        merges: list of merges in creation order, where each item is (left_bytes, right_bytes)
    """
    if special_tokens is None:
        special_tokens = []

    text = Path(input_path).read_text(encoding="utf-8")

    # Initial vocabulary:
    # 1) special tokens first
    # 2) all 256 byte values
    vocab: dict[int, bytes] = {}
    next_id = 0

    for tok in special_tokens:
        vocab[next_id] = tok.encode("utf-8")
        next_id += 1

    for b in range(256):
        vocab[next_id] = bytes([b])
        next_id += 1

    if vocab_size <= len(vocab):
        return {i: vocab[i] for i in range(vocab_size)}, []

    pretokens = _pretokenize_text(text, pattern)

    # Count unique pretokens after converting to byte tuples
    word_counts: Counter[tuple[bytes, ...]] = Counter()
    for token in pretokens:
        token_bytes = token.encode("utf-8")
        word_counts[_bytes_to_tuple(token_bytes)] += 1

    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size:
        pair_counts = _count_adjacent_pairs(word_counts)
        if not pair_counts:
            break

        # Deterministic tie-break:
        # pick highest frequency; if tied, lexicographically smallest pair
        best_pair = min(pair_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]

        merged_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = merged_token
        merges.append(best_pair)

        word_counts = _merge_pair_in_word_counts(word_counts, best_pair)

    return vocab, merges


@dataclass
class BPETokenizer:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str] | None = None
    pattern: str = DEFAULT_PRETOKENIZER_PATTERN

    def __post_init__(self) -> None:
        self.special_tokens = self.special_tokens or []

        self.token_to_id: dict[bytes, int] = {token_bytes: idx for idx, token_bytes in self.vocab.items()}
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {
            pair: rank for rank, pair in enumerate(self.merges)
        }

        # Precompile regex for special tokens, longest-first to avoid prefix issues
        if self.special_tokens:
            escaped = sorted((re.escape(tok) for tok in self.special_tokens), key=len, reverse=True)
            self.special_pattern = re.compile("|".join(escaped))
        else:
            self.special_pattern = None

    def encode(self, text: str) -> list[int]:
        pieces: list[int] = []

        if not self.special_pattern:
            return self._encode_ordinary_text(text)

        last = 0
        for match in self.special_pattern.finditer(text):
            if match.start() > last:
                pieces.extend(self._encode_ordinary_text(text[last:match.start()]))

            special_text = match.group(0)
            special_bytes = special_text.encode("utf-8")
            if special_bytes not in self.token_to_id:
                raise KeyError(f"Special token {special_text!r} not found in vocabulary.")
            pieces.append(self.token_to_id[special_bytes])

            last = match.end()

        if last < len(text):
            pieces.extend(self._encode_ordinary_text(text[last:]))

        return pieces

    def _encode_ordinary_text(self, text: str) -> list[int]:
        token_ids: list[int] = []
        for pretoken in _pretokenize_text(text, self.pattern):
            token_ids.extend(self._encode_pretoken_bytes(pretoken.encode("utf-8")))
        return token_ids

    def _encode_pretoken_bytes(self, token_bytes: bytes) -> list[int]:
        parts: list[bytes] = [bytes([b]) for b in token_bytes]

        while len(parts) >= 2:
            best_idx = None
            best_rank = None

            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_idx = i

            if best_idx is None:
                break

            i = best_idx
            merged = parts[i] + parts[i + 1]
            parts = parts[:i] + [merged] + parts[i + 2 :]

        try:
            return [self.token_to_id[p] for p in parts]
        except KeyError as e:
            raise KeyError(f"Token bytes {e.args[0]!r} not found in vocabulary.") from e

    def decode(self, ids: Iterable[int]) -> str:
        byte_stream = b"".join(self.vocab[i] for i in ids)
        return byte_stream.decode("utf-8", errors="replace")