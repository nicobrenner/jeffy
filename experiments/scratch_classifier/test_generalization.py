"""Test generalization: freeze trained transformers, train logreg on unseen datasets.

Compares:
  1. Scratch transformer (1a, pure clf) embeddings + logreg
  2. Hybrid transformer (1b, clf+MLM) embeddings + logreg
  3. BGE-large embeddings + logreg (Jeffy's approach)
  4. Random embeddings + logreg (baseline)

This tells us if the MLM auxiliary objective produces better-generalizing representations.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jeffy.build_pack import DATASETS, load_and_split

HERE = Path(__file__).resolve().parent

UNSEEN_DATASETS = [
    "financialphrasebank",
    "rottentomatoes",
    "dbpedia",
    "clinc_oos",
    "tweet_eval_offensive",
]

SEEN_DATASETS = ["ag_news", "banking77", "emotion", "imdb", "sms_spam", "sst2"]


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_scratch_model(checkpoint_path, device):
    from train import ScratchTransformerClassifier
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    vocab_map = ckpt["vocab_map"]
    model = ScratchTransformerClassifier(
        vocab_size=config["vocab_size"], d_model=config["d_model"],
        n_heads=config["n_heads"], n_layers=config["n_layers"],
        d_ff=config["d_ff"], max_seq_len=config["max_seq_len"],
        task_classes=config["task_classes"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, vocab_map, config


def load_hybrid_model(checkpoint_path, device):
    from train_hybrid import HybridTransformer
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    vocab_map = ckpt["vocab_map"]
    model = HybridTransformer(
        vocab_size=config["vocab_size"], d_model=config["d_model"],
        n_heads=config["n_heads"], n_layers=config["n_layers"],
        d_ff=config["d_ff"], max_seq_len=config["max_seq_len"],
        task_classes=config["task_classes"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, vocab_map, config


def encode_with_model(model, vocab_map, tokenizer, texts, device,
                      max_len=128, batch_size=128):
    unk_id = 0
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(batch_texts, truncation=True, padding="max_length",
                        max_length=max_len, return_tensors="pt")
        remapped = torch.tensor([
            [vocab_map.get(t.item(), unk_id) for t in row]
            for row in enc["input_ids"]
        ], dtype=torch.long)
        with torch.no_grad():
            emb = model.encode(remapped.to(device), enc["attention_mask"].to(device))
        all_embeddings.append(emb.cpu().numpy())
        if device == "mps" and i % (batch_size * 20) == 0 and i > 0:
            torch.mps.empty_cache()
            torch.mps.synchronize()
    return np.concatenate(all_embeddings, axis=0)


def encode_with_bge(texts, device, batch_size=128):
    from sentence_transformers import SentenceTransformer
    bge_device = device if device != "mps" else "mps"
    encoder = SentenceTransformer("BAAI/bge-large-en-v1.5", device=bge_device)
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        emb = encoder.encode(batch, batch_size=batch_size, show_progress_bar=False)
        all_embeddings.append(emb)
        if device == "mps" and i % (batch_size * 20) == 0 and i > 0:
            torch.mps.empty_cache()
            torch.mps.synchronize()
    return np.concatenate(all_embeddings, axis=0)


def encode_random(n_samples, dim=128):
    np.random.seed(42)
    return np.random.randn(n_samples, dim).astype(np.float32)


def eval_logreg(train_X, train_y, test_X, test_y, C=0.01):
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)
    clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs", multi_class="multinomial")
    clf.fit(train_X, train_y)
    preds = clf.predict(test_X)
    acc = accuracy_score(test_y, preds)
    f1 = f1_score(test_y, preds, average="macro")
    return acc, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unseen-only", action="store_true")
    parser.add_argument("--skip-bge", action="store_true")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}", flush=True)

    print("Loading scratch model (1a)...", flush=True)
    model_1a, vocab_1a, config_1a = load_scratch_model(
        HERE / "checkpoints" / "best.pt", device)
    print(f"  1a: d_model={config_1a['d_model']}, "
          f"{sum(p.numel() for p in model_1a.parameters()):,} params")

    print("Loading hybrid model (1b)...", flush=True)
    model_1b, vocab_1b, config_1b = load_hybrid_model(
        HERE / "checkpoints" / "best_hybrid.pt", device)
    print(f"  1b: d_model={config_1b['d_model']}, "
          f"{sum(p.numel() for p in model_1b.parameters()):,} params")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    datasets_to_test = UNSEEN_DATASETS if args.unseen_only else UNSEEN_DATASETS + SEEN_DATASETS
    print(f"\nTesting {len(datasets_to_test)} datasets "
          f"({'unseen only' if args.unseen_only else f'{len(UNSEEN_DATASETS)} unseen + {len(SEEN_DATASETS)} seen'})")
    print(f"{'='*90}", flush=True)

    results = []

    for ds_name in datasets_to_test:
        if ds_name not in DATASETS:
            print(f"  Skipping {ds_name} (not in DATASETS)")
            continue

        config_ds = DATASETS[ds_name]
        tag = "UNSEEN" if ds_name in UNSEEN_DATASETS else "seen"
        print(f"\n{ds_name} ({tag}):", flush=True)

        try:
            train_texts, train_labels, test_texts, test_labels, label_map = \
                load_and_split(ds_name, config_ds, max_train=5000, max_test=1000)
        except Exception as e:
            print(f"  Failed to load: {e}")
            continue

        n_classes = len(set(train_labels))
        print(f"  {len(train_texts)} train, {len(test_texts)} test, {n_classes} classes")

        unique = sorted(set(train_labels) | set(test_labels))
        l2i = {l: i for i, l in enumerate(unique)}
        train_y = [l2i[l] for l in train_labels]
        test_y = [l2i[l] for l in test_labels]

        # 1a: Scratch transformer
        print("  Encoding with 1a (pure clf)...", end=" ", flush=True)
        t0 = time.time()
        scratch_train = encode_with_model(model_1a, vocab_1a, tokenizer, train_texts, device)
        scratch_test = encode_with_model(model_1a, vocab_1a, tokenizer, test_texts, device)
        print(f"({time.time()-t0:.1f}s)", flush=True)
        scratch_acc, scratch_f1 = eval_logreg(scratch_train, train_y, scratch_test, test_y)

        # 1b: Hybrid transformer
        print("  Encoding with 1b (hybrid)...", end=" ", flush=True)
        t0 = time.time()
        hybrid_train = encode_with_model(model_1b, vocab_1b, tokenizer, train_texts, device)
        hybrid_test = encode_with_model(model_1b, vocab_1b, tokenizer, test_texts, device)
        print(f"({time.time()-t0:.1f}s)", flush=True)
        hybrid_acc, hybrid_f1 = eval_logreg(hybrid_train, train_y, hybrid_test, test_y)

        # BGE-large
        bge_acc, bge_f1 = 0.0, 0.0
        if not args.skip_bge:
            print("  Encoding with BGE-large...", end=" ", flush=True)
            t0 = time.time()
            bge_train = encode_with_bge(train_texts, device)
            bge_test = encode_with_bge(test_texts, device)
            print(f"({time.time()-t0:.1f}s)", flush=True)
            bge_acc, bge_f1 = eval_logreg(bge_train, train_y, bge_test, test_y)

        # Random baseline
        random_train = encode_random(len(train_texts), dim=128)
        random_test = encode_random(len(test_texts), dim=128)
        rand_acc, rand_f1 = eval_logreg(random_train, train_y, random_test, test_y)

        seen = ds_name not in UNSEEN_DATASETS
        results.append({
            "dataset": ds_name, "seen": seen, "n_classes": n_classes,
            "scratch_acc": scratch_acc, "hybrid_acc": hybrid_acc,
            "bge_acc": bge_acc, "random_acc": rand_acc,
            "scratch_f1": scratch_f1, "hybrid_f1": hybrid_f1,
            "bge_f1": bge_f1, "random_f1": rand_f1,
        })
        bge_str = f"{bge_acc:.3f}" if not args.skip_bge else "skip"
        print(f"  1a={scratch_acc:.3f}  1b={hybrid_acc:.3f}  BGE={bge_str}  Random={rand_acc:.3f}")

    # Summary
    print(f"\n{'='*90}")
    header = f"{'Dataset':<24} {'Type':<8} {'Cls':>4} {'1a(clf)':>9} {'1b(hyb)':>9}"
    if not args.skip_bge:
        header += f" {'BGE-lg':>9}"
    header += f" {'Random':>9} {'1b-1a':>7}"
    print(header)
    print(f"{'-'*90}")

    for r in results:
        gap = r["hybrid_acc"] - r["scratch_acc"]
        tag = "seen" if r["seen"] else "UNSEEN"
        line = (f"{r['dataset']:<24} {tag:<8} {r['n_classes']:>4} "
                f"{r['scratch_acc']:>9.3f} {r['hybrid_acc']:>9.3f}")
        if not args.skip_bge:
            line += f" {r['bge_acc']:>9.3f}"
        line += f" {r['random_acc']:>9.3f} {gap:>+7.3f}"
        print(line)

    for label, subset in [("Unseen avg", False), ("Seen avg", True)]:
        group = [r for r in results if r["seen"] == subset]
        if not group:
            continue
        avg_1a = np.mean([r["scratch_acc"] for r in group])
        avg_1b = np.mean([r["hybrid_acc"] for r in group])
        avg_r = np.mean([r["random_acc"] for r in group])
        line = (f"\n{label:<24} {'':8} {'':>4} {avg_1a:>9.3f} {avg_1b:>9.3f}")
        if not args.skip_bge:
            avg_bge = np.mean([r["bge_acc"] for r in group])
            line += f" {avg_bge:>9.3f}"
        line += f" {avg_r:>9.3f} {avg_1b - avg_1a:>+7.3f}"
        print(line)

    out_path = HERE / "generalization_results_1b.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
