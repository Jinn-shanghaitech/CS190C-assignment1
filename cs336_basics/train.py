from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from cs336_basics.bpe import BPETokenizer, train_bpe
from cs336_basics.transformer import TransformerLM
from cs336_basics.losses import cross_entropy
from cs336_basics.data import get_batch
from cs336_basics.optim import AdamW, gradient_clipping, get_lr_cosine_schedule
from cs336_basics.attention import softmax
from cs336_basics.checkpoint import save_checkpoint


def build_tokenizer(text_path: str, vocab_size: int = 256):
    """
    这里假设你 Part B 的 train_bpe 返回 (vocab, merges)。
    如果你的 train_bpe 返回格式不一样，你把这一小段改成你自己的版本即可。
    """
    vocab, merges = train_bpe(
        input_path=text_path,
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
    )
    tokenizer = BPETokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=["<|endoftext|>"],
    )
    return tokenizer


def encode_text(tokenizer: BPETokenizer, text: str) -> np.ndarray:
    token_ids = tokenizer.encode(text)
    return np.array(token_ids, dtype=np.int64)


def decode_ids(tokenizer: BPETokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids)


def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens,
    context_length,
    device,
):
    model.eval()

    prompt_ids = tokenizer.encode(prompt)
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            x_cond = x[:, -context_length:]
            logits = model(x_cond)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            x = torch.cat([x, next_token], dim=1)

    return tokenizer.decode(x[0].tolist())
def main():
    # 1. basic config
    data_path = Path("data/tiny.txt")

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    vocab_size_for_bpe = 1000

    context_length = 128
    d_model = 128
    num_layers = 4
    num_heads = 4
    d_ff = 512
    rope_theta = 10000.0

    batch_size = 8
    max_iters = 2000
    warmup_iters = 100
    learning_rate = 2e-4
    min_learning_rate = 2e-5
    weight_decay = 1e-2
    grad_clip = 1.0
    print_every = 100

    # 2. read text
    text = data_path.read_text(encoding="utf-8")

    # 3. build tokenizer + encode
    tokenizer = build_tokenizer(str(data_path), vocab_size=vocab_size_for_bpe)
    token_array = encode_text(tokenizer, text)

    test_text = "hello world"
    test_ids = tokenizer.encode(test_text)
    test_decoded = tokenizer.decode(test_ids)

    print("original:", test_text)
    print("ids:", test_ids)
    print("decoded:", test_decoded)
    print("round-trip ok?", test_text == test_decoded)

    actual_vocab_size = len(tokenizer.vocab)

    split_idx = int(0.9 * len(token_array))
    train_data = token_array[:split_idx]
    val_data = token_array[split_idx:]

    # 4. build model

    model = TransformerLM(
        vocab_size=actual_vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # 5. train loop
    model.train()

    for it in range(max_iters):
        # set lr from schedule
        lr = get_lr_cosine_schedule(
            it=it,
            max_learning_rate=learning_rate,
            min_learning_rate=min_learning_rate,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=max_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(
            dataset=train_data,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )

        logits = model(x)  # (B, T, V)
        B, T, V = logits.shape

        loss = cross_entropy(
            logits.reshape(B * T, V),
            y.reshape(B * T),
        )

        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), grad_clip)
        optimizer.step()

        if it % print_every == 0 or it == max_iters - 1:
            print(f"iter {it:4d} | lr {lr:.6f} | train loss {loss.item():.4f}")

    # 6. save checkpoint
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=max_iters,
        out="toy_checkpoint.pt",
    )

    # 7. quick generation
    prompt = "Lila smiled and nodded. Her hair"
    print("PROMPT USED:", repr(prompt))

    sample = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=120,
        context_length=context_length,
        device=device,
    )

    print("\n===== GENERATED TEXT =====")
    print(sample)


if __name__ == "__main__":
    main()