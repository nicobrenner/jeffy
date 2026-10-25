"""Experiment: MLM-only baseline (no classification).

Trains the same architecture as experiment 1b but with ONLY the masked
language modeling objective. Compares MLM loss against the hybrid model
to test whether classification gradients improve language modeling.

If hybrid's MLM loss < MLM-only's loss at the same epoch, it means
classification signal helps the model learn language structure better.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from jeffy.build_pack import DATASETS, load_and_split

HERE = Path(__file__).resolve().parent

DEFAULT_DATASETS = [
    "banking77", "ag_news", "imdb", "emotion", "sst2", "sms_spam",
]


class TextOnlyDataset(Dataset):
    """Just texts, no labels — for MLM-only training."""
    def __init__(self, texts, tokenizer, max_len=128):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=max_len, return_tensors="pt",
        )

    def __len__(self):
        return self.encodings["input_ids"].size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }


class MLMOnlyTransformer(nn.Module):
    """Same architecture as HybridTransformer but without classification heads."""

    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2,
                 d_ff=512, max_seq_len=128, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size),
        )

    def forward(self, input_ids, attention_mask):
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        src_key_padding_mask = attention_mask == 0
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)
        return self.mlm_head(x)


def mask_tokens(input_ids, attention_mask, mask_prob=0.15, mask_token_id=0):
    device = input_ids.device
    labels = input_ids.clone()
    prob_matrix = torch.full(input_ids.shape, mask_prob, device=device)
    prob_matrix[attention_mask == 0] = 0.0
    prob_matrix[:, 0] = 0.0
    masked_indices = torch.bernoulli(prob_matrix).bool()
    labels[~masked_indices] = -100

    indices_replaced = masked_indices & (torch.rand(input_ids.shape, device=device) < 0.8)
    input_ids[indices_replaced] = mask_token_id

    indices_random = masked_indices & ~indices_replaced & (torch.rand(input_ids.shape, device=device) < 0.5)
    random_tokens = torch.randint(0, input_ids.max().item() + 1, input_ids.shape, device=device)
    input_ids[indices_random] = random_tokens[indices_random]

    return input_ids, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"MLM-only training (no classification)")
    print(f"Mask prob: {args.mask_prob}")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading datasets (text only, no labels needed)...")
    all_texts = []
    for ds_name in DEFAULT_DATASETS:
        if ds_name not in DATASETS:
            continue
        config = DATASETS[ds_name]
        print(f"  Loading {ds_name}...", end=" ", flush=True)
        t0 = time.time()
        train_texts, train_labels, test_texts, test_labels, label_map = \
            load_and_split(ds_name, config, max_train=10000, max_test=2000)
        all_texts.extend(train_texts)
        print(f"{len(train_texts)} texts ({time.time()-t0:.1f}s)")

    print(f"\nTotal training texts: {len(all_texts)}")

    # Build vocab subset (same as hybrid)
    print("Building vocab subset...")
    enc_all = tokenizer(all_texts, truncation=True, padding=False, max_length=128)
    token_counts = {}
    for ids in enc_all["input_ids"]:
        for tid in ids:
            token_counts[tid] = token_counts.get(tid, 0) + 1

    special_ids = set(tokenizer.all_special_ids)
    kept = sorted(list(set(token_counts.keys()) | special_ids))
    full_to_compact = {full_id: compact_id for compact_id, full_id in enumerate(kept)}
    vocab_size = len(kept)
    print(f"Vocab: {tokenizer.vocab_size} -> {vocab_size}")

    # Create dataset
    dataset = TextOnlyDataset(all_texts, tokenizer, max_len=128)

    # Remap tokens
    ids = dataset.encodings["input_ids"]
    unk_id = 0
    remapped = torch.tensor([
        [full_to_compact.get(t.item(), unk_id) for t in row]
        for row in ids
    ], dtype=torch.long)
    dataset.encodings["input_ids"] = remapped

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = MLMOnlyTransformer(
        vocab_size=vocab_size, d_model=128, n_heads=4, n_layers=2,
        d_ff=512, max_seq_len=128, dropout=0.1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(loader)
    )

    out_dir = HERE / "checkpoints"
    out_dir.mkdir(exist_ok=True)

    start_epoch = 0
    best_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch} (best_loss={best_loss:.4f})")

    print(f"\nTraining MLM-only for {args.epochs} epochs (starting from {start_epoch})...\n")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            input_ids = batch["input_ids"].clone().to(device)
            attention_mask = batch["attention_mask"].to(device)

            masked_ids, mlm_labels = mask_tokens(
                input_ids, attention_mask,
                mask_prob=args.mask_prob, mask_token_id=0,
            )

            logits = model(masked_ids, attention_mask)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                mlm_labels.to(device).view(-1),
                ignore_index=-100,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

            if device == "mps" and n_batches % 40 == 0:
                torch.mps.empty_cache()
                torch.mps.synchronize()

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1}/{args.epochs}  mlm_loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch, "mlm_loss": avg_loss, "best_loss": best_loss,
                "vocab_size": vocab_size, "vocab_map": full_to_compact,
            }, out_dir / "best_mlm_only.pt")

        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch, "mlm_loss": avg_loss, "best_loss": best_loss,
            "vocab_size": vocab_size, "vocab_map": full_to_compact,
        }, out_dir / "last_mlm_only.pt")

    print(f"\n{'='*60}")
    print("MLM-only training complete.")
    print(f"Final MLM loss: {avg_loss:.4f}")
    print(f"Best MLM loss: {best_loss:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
