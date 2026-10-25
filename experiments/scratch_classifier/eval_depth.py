"""Test depth recurrence scaling: load a trained checkpoint, eval at various depths.

The key question: can a model trained at 2x depth benefit from MORE depth at
inference time, without retraining? If yes, we get "quiet reasoning" for free.

Usage:
    python eval_depth.py --checkpoint checkpoints/best_hybrid_1a_iter2.pt
    python eval_depth.py --checkpoint checkpoints/best_hybrid_tied_iter2.pt
    python eval_depth.py --checkpoint checkpoints/best_hybrid_1a_iter4.pt
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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


def load_task_data(ds_names, tokenizer, max_len=128, max_train=100, max_test=2000):
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
        test_label_ids = [label_to_idx[l] for l in test_labels]

        test_ds = TextClassificationDataset(test_texts, test_label_ids, tokenizer, max_len)

        tasks[ds_name] = {
            "test": test_ds,
            "n_classes": len(unique_labels),
        }
        print(f"{len(test_label_ids)} test, {len(unique_labels)} classes ({time.time()-t0:.1f}s)")
    return tasks


class HybridTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2,
                 d_ff=512, max_seq_len=128, dropout=0.1, task_classes=None,
                 tie_weights=False, n_iterations=1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_iterations = n_iterations

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        self.heads = nn.ModuleDict()
        if task_classes:
            for task_name, n_classes in task_classes.items():
                self.heads[task_name] = nn.Linear(d_model, n_classes)

        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size),
        )

        if tie_weights:
            self.mlm_head[-1].weight = self.token_emb.weight

    def encode(self, input_ids, attention_mask, n_iterations=None):
        iters = n_iterations or self.n_iterations
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        src_key_padding_mask = attention_mask == 0
        for _ in range(iters):
            x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)

        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        return pooled

    def forward_classify(self, input_ids, attention_mask, task_name, n_iterations=None):
        pooled = self.encode(input_ids, attention_mask, n_iterations=n_iterations)
        return self.heads[task_name](pooled)


def evaluate(model, tasks, device, n_iterations, batch_size=128):
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
                logits = model.forward_classify(
                    input_ids, attention_mask, task_name,
                    n_iterations=n_iterations,
                )
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            results[task_name] = correct / total
    results["average"] = np.mean(list(results.values()))
    return results


def benchmark_latency(model, tasks, device, n_iterations, n_runs=100):
    model.eval()
    task_name = list(tasks.keys())[0]
    sample = tasks[task_name]["test"][0]
    input_ids = sample["input_ids"].unsqueeze(0).to(device)
    attention_mask = sample["attention_mask"].unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(10):
            model.forward_classify(input_ids, attention_mask, task_name,
                                   n_iterations=n_iterations)
        if device == "mps":
            torch.mps.synchronize()

        start = time.perf_counter()
        for _ in range(n_runs):
            model.forward_classify(input_ids, attention_mask, task_name,
                                   n_iterations=n_iterations)
        if device == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - start

    return (elapsed / n_runs) * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--depths", type=str, default="1,2,4,8,16,32,64",
                        help="Comma-separated depth iterations to test")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-test", type=int, default=2000)
    args = parser.parse_args()

    depths = [int(d) for d in args.depths.split(",")]

    device = "mps" if torch.backends.mps.is_available() and not args.cpu else "cpu"
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    vocab_map = ckpt["vocab_map"]
    train_depth = config.get("n_iterations", 1)
    tied = config.get("tie_weights", False)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"  Trained at depth: {train_depth}x")
    print(f"  d_model={config['d_model']}, n_layers={config['n_layers']}, "
          f"n_heads={config['n_heads']}, tied={tied}")
    print(f"  Saved accuracy: {ckpt.get('avg_acc', 'N/A')}")
    print()

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading datasets...")
    tasks = load_task_data(DEFAULT_DATASETS, tokenizer,
                           max_len=config["max_seq_len"], max_test=args.max_test)

    compact_vocab_size = config["vocab_size"]
    unk_id = vocab_map.get(tokenizer.unk_token_id, 0)
    max_id = max(vocab_map.keys()) + 1
    lookup = torch.full((max_id,), unk_id, dtype=torch.long)
    for full_id, compact_id in vocab_map.items():
        lookup[full_id] = compact_id

    for task_data in tasks.values():
        ds = task_data["test"]
        ds.encodings["input_ids"] = lookup[ds.encodings["input_ids"]]

    model = HybridTransformer(
        vocab_size=compact_vocab_size,
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config["d_ff"],
        max_seq_len=config["max_seq_len"],
        task_classes=config["task_classes"],
        tie_weights=config.get("tie_weights", False),
        n_iterations=train_depth,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    print(f"Model loaded. Testing depths: {depths}\n")

    print(f"{'Depth':>6} | {'Avg Acc':>8} | {'Latency':>10} | {'vs Train':>10} | Per-task")
    print(f"{'-'*6}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}-+{'-'*50}")

    base_acc = None
    for depth in depths:
        results = evaluate(model, tasks, device, n_iterations=depth)
        avg = results.pop("average")

        latency = benchmark_latency(model, tasks, device, n_iterations=depth,
                                    n_runs=50 if depth <= 16 else 20)

        if depth == train_depth:
            base_acc = avg
            delta = "baseline"
        elif base_acc is not None:
            delta = f"{avg - base_acc:+.2%}"
        else:
            delta = "—"

        task_str = "  ".join(f"{k[:4]}={v:.3f}" for k, v in sorted(results.items()))
        print(f"{depth:>5}x | {avg:>7.2%} | {latency:>8.2f}ms | {delta:>10} | {task_str}")

    print(f"\nDone. Train depth was {train_depth}x.")


if __name__ == "__main__":
    main()
