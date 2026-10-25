"""
Fast Pile tokenizer: download raw .jsonl.zst shard from HuggingFace,
stream-decompress and tokenize, write binary file for training.

Much faster than HuggingFace datasets streaming (~28K tok/s) because:
- Direct HTTP download to disk (datacenter bandwidth)
- Sequential local disk read through zstd decompression
- No API overhead

Usage:
    python fast_tokenize_pile.py --output-dir pile_runs/pile_20m --tokens 2.2e9
"""

import argparse
import io
import json
import os
import subprocess
import time

import numpy as np
import zstandard

SHARD_URL = "https://huggingface.co/datasets/monology/pile-uncopyrighted/resolve/main/train/00.jsonl.zst"
SHARD_FILE = "pile_shard_00.jsonl.zst"
VAL_SHARD_URL = "https://huggingface.co/datasets/monology/pile-uncopyrighted/resolve/main/val.jsonl.zst"
VAL_SHARD_FILE = "pile_val.jsonl.zst"


def download_if_needed(url, path):
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"  {path} exists ({size/1e9:.1f} GB), skipping download")
        return
    print(f"  Downloading {url} ...")
    t0 = time.time()
    subprocess.run(["wget", "-q", "--show-progress", "-O", path, url], check=True)
    elapsed = time.time() - t0
    size = os.path.getsize(path)
    print(f"  Downloaded {size/1e9:.1f} GB in {elapsed:.0f}s ({size/elapsed/1e6:.0f} MB/s)")


def tokenize_shard(shard_path, output_path, tokenizer, seq_len, max_tokens, skip_docs=0):
    eos_id = tokenizer.eos_token_id
    dctx = zstandard.ZstdDecompressor()

    buf = []
    total_tokens = 0
    n_docs = 0
    batch = []
    t0 = time.time()

    if os.path.exists(output_path):
        os.remove(output_path)

    with open(shard_path, "rb") as fh:
        reader = dctx.stream_reader(fh, read_size=2**20)
        text_stream = io.TextIOWrapper(reader, encoding="utf-8")

        for line in text_stream:
            n_docs += 1
            if n_docs <= skip_docs:
                continue

            doc = json.loads(line)
            ids = tokenizer.encode(doc["text"])
            buf.extend(ids)
            buf.append(eos_id)

            while len(buf) >= seq_len:
                chunk = buf[:seq_len]
                buf = buf[seq_len:]
                batch.extend(chunk)
                total_tokens += seq_len

                if len(batch) >= 500_000:
                    arr = np.array(batch, dtype=np.uint16)
                    with open(output_path, "ab") as f:
                        f.write(arr.tobytes())
                    batch = []

                    elapsed = time.time() - t0
                    rate = total_tokens / elapsed
                    eta = (max_tokens - total_tokens) / rate if rate > 0 else 0
                    print(
                        f"  {total_tokens/1e6:.0f}M tok | "
                        f"{n_docs - skip_docs:,} docs | "
                        f"{rate/1e3:.0f}K tok/s | "
                        f"ETA {eta/60:.0f}min",
                        flush=True,
                    )

            if total_tokens >= max_tokens:
                break

    if batch:
        arr = np.array(batch, dtype=np.uint16)
        with open(output_path, "ab") as f:
            f.write(arr.tobytes())

    elapsed = time.time() - t0
    file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
    print(
        f"  Done: {total_tokens/1e6:.0f}M tokens from {n_docs:,} docs "
        f"in {elapsed/60:.1f}min ({file_size/1e9:.2f} GB)"
    )
    return total_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokens", type=str, default="2.2e9")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    args = parser.parse_args()

    max_tokens = int(float(args.tokens))
    train_tokens = int(max_tokens * (1 - args.val_fraction))
    val_tokens = int(max_tokens * args.val_fraction)

    os.makedirs(args.output_dir, exist_ok=True)

    from transformers import AutoTokenizer

    tok_path = os.path.expanduser("~/neox_tokenizer")
    if os.path.exists(tok_path):
        tokenizer = AutoTokenizer.from_pretrained(tok_path)
        print(f"Tokenizer: {tok_path}")
    else:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        print("Tokenizer: EleutherAI/gpt-neox-20b")

    train_bin = os.path.join(args.output_dir, "train.bin")
    val_bin = os.path.join(args.output_dir, "val.bin")

    # Download train shard
    if not os.path.exists(train_bin):
        print(f"\n=== Downloading train shard ===")
        download_if_needed(SHARD_URL, SHARD_FILE)

        print(f"\n=== Tokenizing train split ({train_tokens/1e9:.1f}B tokens) ===")
        tokenize_shard(SHARD_FILE, train_bin, tokenizer, args.seq_len, train_tokens)
    else:
        print(f"Train binary exists: {train_bin}")

    # Download val shard (much smaller)
    if not os.path.exists(val_bin):
        print(f"\n=== Downloading val shard ===")
        download_if_needed(VAL_SHARD_URL, VAL_SHARD_FILE)

        print(f"\n=== Tokenizing val split ({val_tokens/1e6:.0f}M tokens) ===")
        tokenize_shard(VAL_SHARD_FILE, val_bin, tokenizer, args.seq_len, val_tokens)
    else:
        print(f"Val binary exists: {val_bin}")

    # Cleanup compressed files
    for f in [SHARD_FILE, VAL_SHARD_FILE]:
        if os.path.exists(f):
            os.remove(f)
            print(f"Cleaned up {f}")

    print(f"\n=== Done ===")
    print(f"Train: {os.path.getsize(train_bin)/1e9:.2f} GB")
    print(f"Val:   {os.path.getsize(val_bin)/1e6:.0f} MB")


if __name__ == "__main__":
    main()
