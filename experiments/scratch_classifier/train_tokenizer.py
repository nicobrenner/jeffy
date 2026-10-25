"""Train a custom BPE tokenizer on TinyStories for minimal vocabulary tax.

GPT-2's 50K vocab subsets to ~15K on TinyStories — still full of code
symbols, CJK, and subwords that never appear in children's stories. A
custom BPE trained directly on the data produces a tight vocab (~5-6K)
that maximizes the transformer-to-embedding ratio.

Usage:
    python train_tokenizer.py
    python train_tokenizer.py --vocab-size 5500 --max-stories 50000
    python train_tokenizer.py --vocab-size 8000 --out tokenizer_8k.json
"""

import argparse
import time

from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer on TinyStories")
    parser.add_argument("--vocab-size", type=int, default=5500)
    parser.add_argument("--max-stories", type=int, default=50000)
    parser.add_argument("--out", type=str, default="tokenizer_tinystories.json")
    args = parser.parse_args()

    special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]

    # --- Load stories ---
    print(f"Loading TinyStories ({args.max_stories:,} stories)...", flush=True)
    t0 = time.time()
    ds = load_dataset("roneneldan/TinyStories", split="train")
    n = min(args.max_stories, len(ds))
    stories = [ds[i]["text"] for i in range(n)]
    print(f"  {n:,} stories loaded ({time.time() - t0:.1f}s)", flush=True)

    # --- Train BPE ---
    print(f"Training BPE tokenizer (vocab_size={args.vocab_size})...", flush=True)
    t0 = time.time()

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True,
    )
    tokenizer.train_from_iterator(stories, trainer=trainer)

    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    print(f"  Trained in {time.time() - t0:.1f}s", flush=True)
    print(f"  Final vocab size: {tokenizer.get_vocab_size()}", flush=True)

    # Set special token IDs
    pad_id = tokenizer.token_to_id("<pad>")
    eos_id = tokenizer.token_to_id("<eos>")
    bos_id = tokenizer.token_to_id("<bos>")

    # --- Save ---
    tokenizer.save(args.out)
    print(f"  Saved to {args.out}", flush=True)

    # --- Stats ---
    print("\n--- Tokenizer Stats ---", flush=True)
    total_tokens = 0
    total_chars = 0
    for story in stories:
        enc = tokenizer.encode(story)
        total_tokens += len(enc.ids)
        total_chars += len(story)

    avg_tokens = total_tokens / n
    avg_chars = total_chars / n
    chars_per_token = total_chars / total_tokens

    print(f"  Stories: {n:,}", flush=True)
    print(f"  Avg tokens/story: {avg_tokens:.1f}", flush=True)
    print(f"  Avg chars/story: {avg_chars:.1f}", flush=True)
    print(f"  Chars/token (compression): {chars_per_token:.2f}", flush=True)
    print(f"  Total tokens: {total_tokens:,}", flush=True)

    # --- GPT-2 comparison ---
    print("\n--- GPT-2 Comparison ---", flush=True)
    gpt2_tok = AutoTokenizer.from_pretrained("gpt2")

    gpt2_total = 0
    for story in stories:
        gpt2_total += len(gpt2_tok.encode(story))

    gpt2_avg = gpt2_total / n
    gpt2_cpt = total_chars / gpt2_total
    print(f"  GPT-2 avg tokens/story: {gpt2_avg:.1f}", flush=True)
    print(f"  GPT-2 chars/token: {gpt2_cpt:.2f}", flush=True)
    print(f"  Custom avg tokens/story: {avg_tokens:.1f}", flush=True)
    print(f"  Custom chars/token: {chars_per_token:.2f}", flush=True)
    ratio = avg_tokens / gpt2_avg
    print(f"  Token ratio (custom/GPT-2): {ratio:.2f}x", flush=True)
    if ratio > 1:
        print(f"  → Custom uses {ratio:.2f}x MORE tokens per story (smaller vocab = longer sequences)")
    else:
        print(f"  → Custom uses {1/ratio:.2f}x FEWER tokens per story")

    # --- Side-by-side on 10 samples ---
    print("\n--- Sample Comparison (first 10 stories) ---", flush=True)
    print(f"{'Story':>6} | {'GPT-2':>8} | {'Custom':>8} | {'Ratio':>6} | First 60 chars", flush=True)
    print(f"{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*40}", flush=True)
    for i in range(min(10, n)):
        story = stories[i]
        g = len(gpt2_tok.encode(story))
        c = len(tokenizer.encode(story).ids)
        r = c / g if g > 0 else 0
        preview = story[:60].replace("\n", " ")
        print(f"{i+1:>6} | {g:>8} | {c:>8} | {r:>5.2f}x | {preview}", flush=True)

    # --- Embedding size comparison ---
    print("\n--- Embedding Size Impact (d_model=384) ---", flush=True)
    d = 384
    for name, v in [("GPT-2 full", 50257), ("GPT-2 subset (typical)", 15271),
                     ("TinyStories curated", 10000),
                     (f"Custom BPE ({args.vocab_size})", tokenizer.get_vocab_size())]:
        emb = v * d
        print(f"  {name:>30}: {v:>6} vocab → {emb/1e6:.2f}M embedding params", flush=True)


if __name__ == "__main__":
    main()
