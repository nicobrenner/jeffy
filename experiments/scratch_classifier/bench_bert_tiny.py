"""Benchmark BERT-Tiny on our 6-task suite for direct comparison.

Fine-tunes google/bert_uncased_L-2_H-128_A-2 on the same 6 datasets
with the same data splits as our scratch experiments.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from jeffy.build_pack import DATASETS, load_and_split

BERT_TINY = "google/bert_uncased_L-2_H-128_A-2"
TASKS = ["banking77", "ag_news", "imdb", "emotion", "sst2", "sms_spam"]
MAX_TRAIN = 10000
MAX_TEST = 2000
MAX_LEN = 128
EPOCHS = 20
LR = 3e-4
BATCH_SIZE = 64


class TextDataset(Dataset):
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


class BertTinyClassifier(nn.Module):
    """BERT-Tiny encoder with per-task classification heads."""

    def __init__(self, model_name, task_classes):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        d_model = self.encoder.config.hidden_size
        self.heads = nn.ModuleDict({
            name: nn.Linear(d_model, n_cls)
            for name, n_cls in task_classes.items()
        })

    def forward(self, input_ids, attention_mask, task_name):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]  # CLS token
        return self.heads[task_name](pooled)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {BERT_TINY}")

    tokenizer = AutoTokenizer.from_pretrained(BERT_TINY)

    print("\nLoading datasets...")
    tasks = {}
    for ds_name in TASKS:
        if ds_name not in DATASETS:
            print(f"  {ds_name}: not found in DATASETS, skipping")
            continue
        config = DATASETS[ds_name]
        print(f"  Loading {ds_name}...", end=" ", flush=True)
        t0 = time.time()
        train_texts, train_labels, test_texts, test_labels, label_map = \
            load_and_split(ds_name, config, max_train=MAX_TRAIN, max_test=MAX_TEST)

        unique_labels = sorted(set(train_labels) | set(test_labels))
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        train_label_ids = [label_to_idx[l] for l in train_labels]
        test_label_ids = [label_to_idx[l] for l in test_labels]

        train_ds = TextDataset(train_texts, train_label_ids, tokenizer, MAX_LEN)
        test_ds = TextDataset(test_texts, test_label_ids, tokenizer, MAX_LEN)

        tasks[ds_name] = {
            "train": train_ds, "test": test_ds,
            "n_classes": len(unique_labels),
        }
        print(f"{len(train_label_ids)} train, {len(test_label_ids)} test, "
              f"{len(unique_labels)} classes ({time.time()-t0:.1f}s)")

    task_classes = {name: t["n_classes"] for name, t in tasks.items()}
    model = BertTinyClassifier(BERT_TINY, task_classes).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    print(f"\nTotal params: {total_params:,} (encoder: {encoder_params:,})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    n_batches = sum(
        len(t["train"]) // BATCH_SIZE + (1 if len(t["train"]) % BATCH_SIZE else 0)
        for t in tasks.values()
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * n_batches
    )

    task_loaders = {
        name: DataLoader(t["train"], batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        for name, t in tasks.items()
    }
    criterion = nn.CrossEntropyLoss()

    best_avg = 0.0
    print(f"\nTraining for {EPOCHS} epochs (round-robin across {len(tasks)} tasks)...\n")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_steps = 0

        iters = {name: iter(loader) for name, loader in task_loaders.items()}
        active = set(iters.keys())

        while active:
            for task_name in list(active):
                try:
                    batch = next(iters[task_name])
                except StopIteration:
                    active.discard(task_name)
                    continue

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                logits = model(input_ids, attention_mask, task_name)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                n_steps += 1

                if device == "mps" and n_steps % 40 == 0:
                    torch.mps.empty_cache()
                    torch.mps.synchronize()

        # Evaluate
        model.eval()
        accs = {}
        with torch.no_grad():
            for task_name, task_data in tasks.items():
                loader = DataLoader(task_data["test"], batch_size=128, shuffle=False)
                correct, total = 0, 0
                for batch in loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["label"].to(device)
                    logits = model(input_ids, attention_mask, task_name)
                    preds = logits.argmax(dim=-1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                accs[task_name] = correct / total

        avg_acc = np.mean(list(accs.values()))
        if avg_acc > best_avg:
            best_avg = avg_acc

        acc_str = "  ".join(f"{k}={v:.3f}" for k, v in accs.items())
        print(f"Epoch {epoch+1}/{EPOCHS}  loss={epoch_loss/max(n_steps,1):.4f}  "
              f"avg_acc={avg_acc:.3f}  {acc_str}")

    print(f"\n{'='*60}")
    print(f"BERT-Tiny Best Average: {best_avg:.4f}")
    print(f"{'='*60}")
    print("\nFinal per-task comparison:")
    print(f"{'Task':<20} {'BERT-Tiny':>10} {'Our Hybrid':>12} {'Diff':>8}")
    print("-" * 52)
    our_results = {
        "banking77": 0.854, "ag_news": 0.835, "emotion": 0.844,
        "imdb": 0.759, "sms_spam": 0.987, "sst2": 0.757,
    }
    for task in TASKS:
        if task in accs:
            ours = our_results.get(task, 0)
            diff = accs[task] - ours
            print(f"{task:<20} {accs[task]:>10.3f} {ours:>12.3f} {diff:>+8.3f}")
    print(f"{'Average':<20} {avg_acc:>10.3f} {np.mean(list(our_results.values())):>12.3f} "
          f"{avg_acc - np.mean(list(our_results.values())):>+8.3f}")


if __name__ == "__main__":
    main()
