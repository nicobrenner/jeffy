"""Experiment 1b: Hybrid classifier + language model from scratch.

Same architecture as experiment 1a, but with an auxiliary masked language
modeling objective. The model learns classification AND language structure
simultaneously.

Usage:
    python -m experiments.scratch_classifier.train_hybrid
    python -m experiments.scratch_classifier.train_hybrid --lm-weight 0.5
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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from jeffy.build_pack import DATASETS, load_and_split

HERE = Path(__file__).resolve().parent

DEFAULT_DATASETS = [
    "banking77", "ag_news", "imdb", "emotion", "sst2", "sms_spam",
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


class HybridTransformer(nn.Module):
    """Transformer with both classification heads and a masked LM head."""

    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2,
                 d_ff=512, max_seq_len=128, dropout=0.1, task_classes=None,
                 tie_weights=False, n_iterations=1, iter_aware=False):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_iterations = n_iterations
        self.iter_aware = iter_aware

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        if iter_aware and n_iterations > 1:
            self.iter_emb = nn.Embedding(n_iterations, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        # Classification heads
        self.heads = nn.ModuleDict()
        if task_classes:
            for task_name, n_classes in task_classes.items():
                self.heads[task_name] = nn.Linear(d_model, n_classes)

        # Masked LM head (predicts original token at masked positions)
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size),
        )

        if tie_weights:
            self.mlm_head[-1].weight = self.token_emb.weight

    def _transformer_forward(self, input_ids, attention_mask):
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        src_key_padding_mask = attention_mask == 0
        for i in range(self.n_iterations):
            if self.iter_aware and self.n_iterations > 1:
                x = x + self.iter_emb(torch.tensor(i, device=x.device))
            x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)
        return x

    def encode(self, input_ids, attention_mask):
        """Mean-pool for classification."""
        x = self._transformer_forward(input_ids, attention_mask)
        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        return pooled

    def forward_classify(self, input_ids, attention_mask, task_name):
        pooled = self.encode(input_ids, attention_mask)
        return self.heads[task_name](pooled)

    def forward_mlm(self, input_ids, attention_mask):
        """Return per-token logits for MLM."""
        x = self._transformer_forward(input_ids, attention_mask)
        return self.mlm_head(x)


def mask_tokens(input_ids, attention_mask, mask_prob=0.15, mask_token_id=0):
    """BERT-style masking: replace 15% of tokens with mask_token_id.

    Returns masked input_ids and labels (original token at masked positions, -100 elsewhere).
    """
    labels = input_ids.clone()
    prob_matrix = torch.full(input_ids.shape, mask_prob)

    # Don't mask padding
    prob_matrix[attention_mask == 0] = 0.0
    # Don't mask first token (often special)
    prob_matrix[:, 0] = 0.0

    masked_indices = torch.bernoulli(prob_matrix).bool()

    # Labels: -100 for non-masked (ignored by cross_entropy)
    labels[~masked_indices] = -100

    # 80% of the time, replace with mask token
    # 10% random token, 10% keep original (standard BERT recipe)
    indices_replaced = masked_indices & (torch.rand(input_ids.shape) < 0.8)
    input_ids[indices_replaced] = mask_token_id

    indices_random = masked_indices & ~indices_replaced & (torch.rand(input_ids.shape) < 0.5)
    random_tokens = torch.randint(0, input_ids.max().item() + 1, input_ids.shape)
    input_ids[indices_random] = random_tokens[indices_random]

    return input_ids, labels


def load_task_data(ds_names, tokenizer, max_len=128, max_train=10000, max_test=2000):
    tasks = {}
    for ds_name in ds_names:
        if ds_name not in DATASETS:
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

        unique_labels = sorted(set(train_labels) | set(test_labels))
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        train_label_ids = [label_to_idx[l] for l in train_labels]
        test_label_ids = [label_to_idx[l] for l in test_labels]

        train_ds = TextClassificationDataset(train_texts, train_label_ids, tokenizer, max_len)
        test_ds = TextClassificationDataset(test_texts, test_label_ids, tokenizer, max_len)

        tasks[ds_name] = {
            "train": train_ds, "test": test_ds,
            "n_classes": len(unique_labels),
            "n_train": len(train_label_ids), "n_test": len(test_label_ids),
        }
        print(f"{len(train_label_ids)} train, {len(test_label_ids)} test, "
              f"{len(unique_labels)} classes ({time.time()-t0:.1f}s)")
    return tasks


def build_vocab_subset(tokenizer, tasks, min_freq=1):
    token_counts = {}
    for task in tasks.values():
        for i in range(len(task["train"])):
            for tid in task["train"].encodings["input_ids"][i].tolist():
                token_counts[tid] = token_counts.get(tid, 0) + 1

    special_ids = set(tokenizer.all_special_ids)
    kept = sorted(
        [tid for tid, count in token_counts.items() if count >= min_freq]
        + list(special_ids)
    )
    full_to_compact = {full_id: compact_id for compact_id, full_id in enumerate(kept)}
    unk_compact = full_to_compact.get(tokenizer.unk_token_id, 0)
    return full_to_compact, unk_compact, len(kept)


def remap_tokens(dataset, full_to_compact, unk_id):
    ids = dataset.encodings["input_ids"]
    remapped = torch.tensor([
        [full_to_compact.get(t.item(), unk_id) for t in row]
        for row in ids
    ], dtype=torch.long)
    dataset.encodings["input_ids"] = remapped


def evaluate(model, tasks, device, batch_size=128):
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
                logits = model.forward_classify(input_ids, attention_mask, task_name)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            results[task_name] = correct / total
    results["average"] = np.mean(list(results.values()))
    return results


def train(args):
    device = "mps" if torch.backends.mps.is_available() and not args.cpu else "cpu"
    print(f"Device: {device}")
    print(f"LM weight: {args.lm_weight}")
    print(f"Mask prob: {args.mask_prob}")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds_names = args.datasets or DEFAULT_DATASETS
    print(f"\nLoading {len(ds_names)} datasets:")
    tasks = load_task_data(ds_names, tokenizer, max_len=args.max_len,
                           max_train=args.max_train, max_test=args.max_test)
    if not tasks:
        print("No datasets loaded!")
        return

    print(f"\nBuilding vocab subset...", end=" ")
    full_to_compact, unk_id, compact_vocab_size = build_vocab_subset(tokenizer, tasks)
    print(f"{compact_vocab_size} tokens")

    for task_data in tasks.values():
        remap_tokens(task_data["train"], full_to_compact, unk_id)
        remap_tokens(task_data["test"], full_to_compact, unk_id)

    task_classes = {name: data["n_classes"] for name, data in tasks.items()}
    model = HybridTransformer(
        vocab_size=compact_vocab_size,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_ff, max_seq_len=args.max_len, dropout=args.dropout,
        task_classes=task_classes,
        tie_weights=args.tie_weights,
        n_iterations=args.n_iterations,
        iter_aware=args.iter_aware,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    mlm_params = sum(p.numel() for p in model.mlm_head.parameters())
    flags = []
    if args.tie_weights:
        flags.append("weight-tied")
    if args.n_iterations > 1:
        flags.append(f"{args.n_iterations}x depth-recurrent")
    if args.iter_aware:
        flags.append("iteration-aware")
    flag_str = f" ({', '.join(flags)})" if flags else ""
    print(f"\nModel: {n_params:,} params (MLM head: {mlm_params:,}){flag_str}")
    print(f"  vocab={compact_vocab_size}, d_model={args.d_model}, "
          f"n_layers={args.n_layers}, n_heads={args.n_heads}, d_ff={args.d_ff}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * sum(
        math.ceil(data["n_train"] / args.batch_size) for data in tasks.values()
    ) * 2  # x2 because we do one clf step + one MLM step per batch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    out_dir = HERE / "checkpoints"
    out_dir.mkdir(exist_ok=True)

    start_epoch = 0
    best_avg = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_avg = ckpt.get("best_avg", ckpt.get("avg_acc", 0.0))
        print(f"Resumed from epoch {start_epoch} (best_avg={best_avg:.4f})")

    print(f"\nTraining for {args.epochs} epochs (starting from {start_epoch})")
    print(f"{'='*80}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        clf_loss_sum, mlm_loss_sum = 0.0, 0.0
        epoch_steps = 0

        loaders = {
            name: iter(DataLoader(data["train"], batch_size=args.batch_size, shuffle=True))
            for name, data in tasks.items()
        }
        active_tasks = set(loaders.keys())

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

                # --- Classification step ---
                clf_logits = model.forward_classify(input_ids, attention_mask, task_name)
                clf_loss = F.cross_entropy(clf_logits, labels)

                if args.lm_weight > 0:
                    # --- MLM step (on same batch) ---
                    mlm_input = input_ids.clone().cpu()
                    mlm_mask = attention_mask.cpu()
                    mlm_input, mlm_labels = mask_tokens(
                        mlm_input, mlm_mask, mask_prob=args.mask_prob, mask_token_id=unk_id
                    )
                    mlm_input = mlm_input.to(device)
                    mlm_labels = mlm_labels.to(device)
                    mlm_logits = model.forward_mlm(mlm_input, attention_mask)
                    mlm_loss = F.cross_entropy(
                        mlm_logits.view(-1, compact_vocab_size),
                        mlm_labels.view(-1),
                        ignore_index=-100,
                    )
                    total_loss = clf_loss + args.lm_weight * mlm_loss
                else:
                    mlm_loss = torch.tensor(0.0)
                    total_loss = clf_loss

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                clf_loss_sum += clf_loss.item()
                mlm_loss_sum += mlm_loss.item()
                epoch_steps += 1

                # Prevent MPS hang (same workaround as jeffy engine._encode)
                if device == "mps" and epoch_steps % 10 == 0:
                    torch.mps.empty_cache()
                    torch.mps.synchronize()

        avg_clf = clf_loss_sum / max(epoch_steps, 1)
        avg_mlm = mlm_loss_sum / max(epoch_steps, 1)

        results = evaluate(model, tasks, device, batch_size=args.batch_size)
        avg_acc = results.pop("average")

        task_str = "  ".join(f"{k}={v:.3f}" for k, v in sorted(results.items()))
        print(f"Epoch {epoch+1}/{args.epochs}  clf={avg_clf:.4f}  mlm={avg_mlm:.4f}  "
              f"avg_acc={avg_acc:.3f}  {task_str}")

        if avg_acc > best_avg:
            best_avg = avg_acc
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "vocab_size": compact_vocab_size,
                    "d_model": args.d_model, "n_heads": args.n_heads,
                    "n_layers": args.n_layers, "d_ff": args.d_ff,
                    "max_seq_len": args.max_len, "task_classes": task_classes,
                    "tie_weights": args.tie_weights,
                    "n_iterations": args.n_iterations,
                    "iter_aware": args.iter_aware,
                },
                "vocab_map": full_to_compact,
                "epoch": epoch, "avg_acc": avg_acc, "per_task": results,
                "lm_weight": args.lm_weight, "mask_prob": args.mask_prob,
            }, out_dir / f"best_hybrid_{args.tag}.pt")

        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "config": {
                "vocab_size": compact_vocab_size,
                "d_model": args.d_model, "n_heads": args.n_heads,
                "n_layers": args.n_layers, "d_ff": args.d_ff,
                "max_seq_len": args.max_len, "task_classes": task_classes,
                "tie_weights": args.tie_weights,
                "n_iterations": args.n_iterations,
                "iter_aware": args.iter_aware,
            },
            "vocab_map": full_to_compact,
            "epoch": epoch, "best_avg": best_avg,
            "lm_weight": args.lm_weight, "mask_prob": args.mask_prob,
        }, out_dir / f"last_hybrid_{args.tag}.pt")

    print(f"\n{'='*80}")
    print(f"Best average accuracy: {best_avg:.4f}")

    # Comparison
    print(f"\n{'Dataset':<20} {'Hybrid':>10} {'Pure clf':>10} {'Jeffy':>10}")
    print(f"{'-'*50}")
    jeffy_acc = {
        "banking77": 0.943, "ag_news": 0.906, "imdb": 0.948,
        "emotion": 0.755, "sst2": 0.901, "sms_spam": 0.991,
    }
    pure_clf_acc = {
        "banking77": 0.832, "ag_news": 0.824, "imdb": 0.757,
        "emotion": 0.815, "sst2": 0.751, "sms_spam": 0.978,
    }
    for task_name, acc in sorted(results.items()):
        j = jeffy_acc.get(task_name, 0)
        p = pure_clf_acc.get(task_name, 0)
        print(f"{task_name:<20} {acc:>10.3f} {p:>10.3f} {j:>10.3f}")


def main():
    parser = argparse.ArgumentParser(description="Train hybrid classifier + LM")
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
    parser.add_argument("--lm-weight", type=float, default=0.5,
                        help="Weight of MLM loss relative to classification loss")
    parser.add_argument("--mask-prob", type=float, default=0.15,
                        help="Probability of masking each token for MLM")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--tie-weights", action="store_true",
                        help="Tie MLM output projection to input embeddings (Press & Wolf 2016)")
    parser.add_argument("--n-iterations", type=int, default=1,
                        help="Depth recurrence: run transformer block N times (Huginn-style)")
    parser.add_argument("--iter-aware", action="store_true",
                        help="Add iteration embeddings so the model knows which pass it's on")
    parser.add_argument("--tag", type=str, default=None,
                        help="Tag for checkpoint filenames (default: mask{mask_prob})")
    args = parser.parse_args()
    if args.tag is None:
        args.tag = f"mask{args.mask_prob}"
    train(args)


if __name__ == "__main__":
    main()
