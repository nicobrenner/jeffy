"""Experiment 1: Train a small transformer classifier from scratch.

Multi-task text classification across Jeffy's datasets.
No pretrained weights -- the model learns to classify from raw tokens.

Usage:
    python -m experiments.scratch_classifier.train
    python -m experiments.scratch_classifier.train --datasets banking77 ag_news imdb
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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# Use Jeffy's dataset loading infrastructure
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from jeffy.build_pack import DATASETS, load_and_split

HERE = Path(__file__).resolve().parent

# --- Datasets to start with (fast, diverse, well-understood) ---
DEFAULT_DATASETS = [
    "banking77",      # 77 classes, intent
    "ag_news",        # 4 classes, topic
    "imdb",           # 2 classes, sentiment
    "emotion",        # 6 classes, emotion
    "sst2",           # 2 classes, sentiment
    "sms_spam",       # 2 classes, spam
]


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=max_len, return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label": self.labels[idx],
        }


class ScratchTransformerClassifier(nn.Module):
    """Tiny transformer trained from scratch for multi-task classification.

    Architecture: token embeddings + positional embeddings + transformer layers
    + per-task classification heads.
    """

    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2,
                 d_ff=512, max_seq_len=128, dropout=0.1, task_classes=None):
        super().__init__()
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        # Per-task classification heads
        self.heads = nn.ModuleDict()
        if task_classes:
            for task_name, n_classes in task_classes.items():
                self.heads[task_name] = nn.Linear(d_model, n_classes)

    def add_head(self, task_name, n_classes):
        self.heads[task_name] = nn.Linear(self.d_model, n_classes)

    def encode(self, input_ids, attention_mask):
        """Get sequence representation (mean pooling over non-padding tokens)."""
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        # Transformer expects mask where True = ignore
        src_key_padding_mask = attention_mask == 0
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)

        # Mean pool over non-padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        return pooled

    def forward(self, input_ids, attention_mask, task_name):
        pooled = self.encode(input_ids, attention_mask)
        return self.heads[task_name](pooled)


def load_task_data(ds_names, tokenizer, max_len=128, max_train=10000, max_test=2000):
    """Load multiple datasets, tokenize, and return DataLoaders + metadata."""
    tasks = {}

    for ds_name in ds_names:
        if ds_name not in DATASETS:
            print(f"  Skipping unknown dataset: {ds_name}")
            continue

        config = DATASETS[ds_name]
        print(f"  Loading {ds_name}...", end=" ", flush=True)
        t0 = time.time()

        try:
            train_texts, train_labels, test_texts, test_labels, label_map = \
                load_and_split(ds_name, config, max_train=max_train, max_test=max_test)
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        # Encode labels as integers
        unique_labels = sorted(set(train_labels) | set(test_labels))
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        train_label_ids = [label_to_idx[l] for l in train_labels]
        test_label_ids = [label_to_idx[l] for l in test_labels]

        train_ds = TextClassificationDataset(train_texts, train_label_ids, tokenizer, max_len)
        test_ds = TextClassificationDataset(test_texts, test_label_ids, tokenizer, max_len)

        tasks[ds_name] = {
            "train": train_ds,
            "test": test_ds,
            "n_classes": len(unique_labels),
            "label_map": label_map,
            "label_to_idx": label_to_idx,
            "n_train": len(train_label_ids),
            "n_test": len(test_label_ids),
        }
        print(f"{len(train_label_ids)} train, {len(test_label_ids)} test, "
              f"{len(unique_labels)} classes ({time.time()-t0:.1f}s)")

    return tasks


def build_vocab_subset(tokenizer, tasks, min_freq=1):
    """Find which tokens actually appear in our data. Returns token->compact_id mapping."""
    token_counts = {}
    for task in tasks.values():
        for i in range(len(task["train"])):
            for tid in task["train"].encodings["input_ids"][i].tolist():
                token_counts[tid] = token_counts.get(tid, 0) + 1

    # Keep tokens that appear at least min_freq times, plus special tokens
    special_ids = set(tokenizer.all_special_ids)
    kept = sorted(
        [tid for tid, count in token_counts.items() if count >= min_freq]
        + list(special_ids)
    )
    full_to_compact = {full_id: compact_id for compact_id, full_id in enumerate(kept)}

    # Add UNK mapping for any token not in our subset
    unk_compact = full_to_compact.get(tokenizer.unk_token_id, 0)

    return full_to_compact, unk_compact, len(kept)


def remap_tokens(dataset, full_to_compact, unk_id):
    """Remap token IDs to compact vocabulary."""
    ids = dataset.encodings["input_ids"]
    remapped = torch.tensor([
        [full_to_compact.get(t.item(), unk_id) for t in row]
        for row in ids
    ], dtype=torch.long)
    dataset.encodings["input_ids"] = remapped


def evaluate(model, tasks, device, batch_size=128):
    """Evaluate on all tasks, return per-task and average accuracy."""
    model.eval()
    results = {}
    with torch.no_grad():
        for task_name, task_data in tasks.items():
            loader = DataLoader(task_data["test"], batch_size=batch_size, shuffle=False)
            correct, total = 0, 0
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                logits = model(input_ids, attention_mask, task_name)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            acc = correct / total
            results[task_name] = acc

    results["average"] = np.mean(list(results.values()))
    return results


def train(args):
    device = "mps" if torch.backends.mps.is_available() and not args.cpu else "cpu"
    print(f"Device: {device}")
    print(f"Loading tokenizer: Qwen/Qwen2.5-0.5B")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    ds_names = args.datasets or DEFAULT_DATASETS
    print(f"\nLoading {len(ds_names)} datasets:")
    tasks = load_task_data(ds_names, tokenizer, max_len=args.max_len,
                           max_train=args.max_train, max_test=args.max_test)

    if not tasks:
        print("No datasets loaded!")
        return

    # Build compact vocabulary from actual data
    print(f"\nBuilding vocab subset from training data...", end=" ")
    full_to_compact, unk_id, compact_vocab_size = build_vocab_subset(tokenizer, tasks)
    print(f"{compact_vocab_size} tokens (from {len(tokenizer)} original)")

    # Remap all datasets to compact vocab
    for task_data in tasks.values():
        remap_tokens(task_data["train"], full_to_compact, unk_id)
        remap_tokens(task_data["test"], full_to_compact, unk_id)

    # Build model
    task_classes = {name: data["n_classes"] for name, data in tasks.items()}
    model = ScratchTransformerClassifier(
        vocab_size=compact_vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_len,
        dropout=args.dropout,
        task_classes=task_classes,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {n_params:,} params ({n_trainable:,} trainable)")
    print(f"  vocab={compact_vocab_size}, d_model={args.d_model}, "
          f"n_layers={args.n_layers}, n_heads={args.n_heads}, d_ff={args.d_ff}")

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * sum(
        math.ceil(data["n_train"] / args.batch_size) for data in tasks.values()
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Training loop: round-robin across tasks
    print(f"\nTraining for {args.epochs} epochs ({total_steps} steps)")
    print(f"{'='*70}")

    best_avg = 0.0
    out_dir = HERE / "checkpoints"
    out_dir.mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        # Create DataLoaders for this epoch (re-shuffle)
        loaders = {
            name: iter(DataLoader(data["train"], batch_size=args.batch_size, shuffle=True))
            for name, data in tasks.items()
        }
        active_tasks = set(loaders.keys())

        # Round-robin: one batch per task, repeat until all exhausted
        while active_tasks:
            for task_name in list(active_tasks):
                try:
                    batch = next(loaders[task_name])
                except StopIteration:
                    active_tasks.discard(task_name)
                    continue

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                logits = model(input_ids, attention_mask, task_name)
                loss = F.cross_entropy(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                epoch_steps += 1

        avg_loss = epoch_loss / max(epoch_steps, 1)

        # Evaluate
        results = evaluate(model, tasks, device, batch_size=args.batch_size)
        avg_acc = results.pop("average")

        task_str = "  ".join(f"{k}={v:.3f}" for k, v in sorted(results.items()))
        print(f"Epoch {epoch+1}/{args.epochs}  loss={avg_loss:.4f}  avg_acc={avg_acc:.3f}  {task_str}")

        if avg_acc > best_avg:
            best_avg = avg_acc
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "vocab_size": compact_vocab_size,
                    "d_model": args.d_model,
                    "n_heads": args.n_heads,
                    "n_layers": args.n_layers,
                    "d_ff": args.d_ff,
                    "max_seq_len": args.max_len,
                    "task_classes": task_classes,
                },
                "vocab_map": full_to_compact,
                "epoch": epoch,
                "avg_acc": avg_acc,
                "per_task": results,
            }, out_dir / "best.pt")

    print(f"\n{'='*70}")
    print(f"Best average accuracy: {best_avg:.4f}")

    # Final comparison with Jeffy
    print(f"\nComparison (this model vs Jeffy on same test sets):")
    print(f"{'Dataset':<20} {'Scratch':>10} {'Jeffy':>10} {'Gap':>10}")
    print(f"{'-'*50}")

    # Jeffy test accuracies from build_pack manifests (approximate)
    jeffy_acc = {
        "banking77": 0.943, "ag_news": 0.906, "imdb": 0.948,
        "emotion": 0.755, "sst2": 0.901, "sms_spam": 0.991,
    }
    for task_name, acc in sorted(results.items()):
        j_acc = jeffy_acc.get(task_name, None)
        gap = f"{acc - j_acc:+.3f}" if j_acc else "n/a"
        j_str = f"{j_acc:.3f}" if j_acc else "n/a"
        print(f"{task_name:<20} {acc:>10.3f} {j_str:>10} {gap:>10}")


def main():
    parser = argparse.ArgumentParser(description="Train scratch transformer classifier")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-train", type=int, default=10000)
    parser.add_argument("--max-test", type=int, default=2000)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
