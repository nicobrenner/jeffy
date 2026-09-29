"""Evaluate shipped model pack artifacts on held-out test sets.

Loads the exact frozen artifacts (no retraining), reproduces evaluation
with documented splits, saves per-example predictions, and runs baselines.

Usage:
    python -m jeffy.evaluate
    python -m jeffy.evaluate --tasks banking77 ag_news
    python -m jeffy.evaluate --baselines
"""

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from .build_pack import DATASETS, load_and_split
from .engine import Engine
from .model_pack import load_artifact


def ci_95(values):
    """95% CI via bootstrap (1000 resamples)."""
    n = len(values)
    if n < 2:
        return float(np.mean(values)), 0.0, 0.0
    rng = np.random.RandomState(42)
    boots = [np.mean(rng.choice(values, size=n, replace=True)) for _ in range(1000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(np.mean(values)), float(lo), float(hi)


def measure_latency(engine, task_id, texts, n_warmup=3, n_measure=50):
    """Measure single-example inference latency."""
    sample = texts[:max(n_warmup + n_measure, len(texts))]

    # Warmup
    for t in sample[:n_warmup]:
        engine.predict(task_id, t)

    latencies = []
    embed_times = []
    clf_times = []
    for t in sample[n_warmup:n_warmup + n_measure]:
        r = engine.predict(task_id, t)
        latencies.append(r["latency_ms"])
        embed_times.append(r["embedding_ms"])
        clf_times.append(r["classifier_ms"])

    return {
        "total_p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "total_p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "embed_p50_ms": round(float(np.percentile(embed_times, 50)), 2),
        "embed_p95_ms": round(float(np.percentile(embed_times, 95)), 2),
        "clf_p50_ms": round(float(np.percentile(clf_times, 50)), 4),
        "clf_p95_ms": round(float(np.percentile(clf_times, 95)), 4),
        "n_measured": len(latencies),
    }


def run_baselines(train_texts, train_labels, test_texts, test_labels):
    """Run majority-class and tuned TF-IDF+LR baselines.

    TF-IDF+LR is tuned via 3-fold cross-validation on training data over
    a grid of regularization strengths and feature configurations.
    Vectorizer is fit inside each CV fold to prevent leakage.
    """
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline

    results = {}

    # Majority class
    dc = DummyClassifier(strategy="most_frequent")
    dc.fit(train_texts, train_labels)
    maj_preds = dc.predict(test_texts)
    results["majority"] = {
        "accuracy": round(accuracy_score(test_labels, maj_preds), 6),
        "macro_f1": round(f1_score(test_labels, maj_preds, average="macro", zero_division=0), 6),
    }

    # TF-IDF + Logistic Regression with CV-tuned hyperparameters.
    # Pipeline ensures vectorizer is fit inside each CV fold.
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=3000, solver="newton-cg", random_state=42)),
    ])
    param_grid = {
        "tfidf__max_features": [10000, 30000],
        "tfidf__analyzer": ["word", "char_wb"],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.01, 0.1, 1.0, 10.0],
    }
    grid = GridSearchCV(
        pipe, param_grid, cv=3, scoring="accuracy",
        n_jobs=1, refit=True,
    )
    grid.fit(train_texts, train_labels)
    tfidf_preds = grid.predict(test_texts)

    results["tfidf_lr"] = {
        "accuracy": round(accuracy_score(test_labels, tfidf_preds), 6),
        "macro_f1": round(f1_score(test_labels, tfidf_preds, average="macro", zero_division=0), 6),
        "best_params": {k: v for k, v in grid.best_params_.items()},
        "cv_best_score": round(grid.best_score_, 6),
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate model pack")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--pack-dir", default="data/model_pack")
    parser.add_argument("--out", default="data/eval_results")
    parser.add_argument("--max-test", type=int, default=2000)
    parser.add_argument("--baselines", action="store_true", help="Run tuned baselines")
    parser.add_argument("--latency", action="store_true", help="Measure latency")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"],
                        help="Inference device for encoder")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pack_dir = Path(args.pack_dir)

    # Pin thread counts for reproducible latency
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    # Load engine with frozen artifacts on explicit device
    print(f"Loading engine from model pack (device={args.device})...")
    engine = Engine(str(pack_dir), device=args.device)
    engine.load()
    print(f"  {len(engine.capabilities)} capabilities loaded")
    print(f"  Encoder: {engine.encoder_name} on {args.device} (loaded in {engine.encoder_load_time:.1f}s)")

    tasks = args.tasks or sorted(engine.capabilities.keys())
    results = []

    for task_id in tasks:
        if task_id not in engine.capabilities:
            print(f"\n  SKIP {task_id}: not in model pack")
            continue

        cap = engine.capabilities[task_id]
        ds_config = DATASETS.get(task_id)
        if ds_config is None:
            print(f"\n  SKIP {task_id}: no dataset config")
            continue

        print(f"\n{'='*60}")
        print(f"{task_id}: {cap.name}")
        print(f"{'='*60}")

        # Load artifact manifest for documentation
        manifest_path = pack_dir / task_id / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        # Load test set using same config as build
        try:
            train_texts, train_labels, test_texts, test_labels, labels = \
                load_and_split(task_id, ds_config, max_test=args.max_test)
        except Exception as e:
            print(f"  FAILED to load dataset: {e}")
            continue

        # Document split details
        hf_source = ds_config["hf"]
        hf_config = ds_config.get("hf_config", None)
        split_name = ds_config.get("split_test", "test")
        filter_label = ds_config.get("filter_label", None)

        print(f"  Source: {hf_source}" + (f" config={hf_config}" if hf_config else ""))
        print(f"  Eval split: {split_name}")
        print(f"  Test examples: {len(test_texts)} (max_test={args.max_test}, seed=42)")
        print(f"  Train examples: {len(train_texts)} (for baselines)")
        if filter_label is not None:
            print(f"  Filtered label: {filter_label}")

        # Task-specific notes
        notes = []
        if task_id == "sst2":
            notes.append("Evaluation uses 'validation' split (official test set labels are not public)")
        elif task_id == "sms_spam":
            notes.append("Single random split (test_size=0.2, seed=42) — no standard train/test")
        elif task_id == "snli":
            notes.append("Labels -1 (unlabeled) filtered. Input: 'premise [SEP] hypothesis'")
        elif task_id == "clinc_oos":
            notes.append("151 classes including out-of-scope. Config: 'plus'")
        elif task_id == "massive_intent":
            notes.append("English subset only. Config: 'en'. String labels used directly")

        for note in notes:
            print(f"  NOTE: {note}")

        # Run predictions
        predictions = []
        pred_labels = []
        for text in test_texts:
            r = engine.predict(task_id, text)
            predictions.append(r)
            pred_labels.append(r["label_id"])

        # Accuracy and F1
        correct = sum(1 for p, t in zip(pred_labels, test_labels) if p == t)
        accuracy = correct / len(test_texts)
        unique = sorted(set(test_labels) | set(pred_labels))
        macro_f1 = f1_score(test_labels, pred_labels, labels=unique, average="macro", zero_division=0)

        # Per-correct vector for CI
        correct_vec = [1 if p == t else 0 for p, t in zip(pred_labels, test_labels)]
        acc_mean, acc_lo, acc_hi = ci_95(correct_vec)

        print(f"  Accuracy:  {accuracy:.4f} [{acc_lo:.4f}, {acc_hi:.4f}]")
        print(f"  Macro F1:  {macro_f1:.4f}")
        print(f"  Train acc: {manifest['train_accuracy']} (from build)")

        # Verify matches build-time results
        build_acc = manifest.get("test_accuracy")
        if build_acc is not None:
            delta = abs(accuracy - build_acc)
            status = "MATCH" if delta < 0.001 else f"DISCREPANCY (Δ={delta:.4f})"
            print(f"  Build-time test_acc: {build_acc:.4f} — {status}")

        # CLINC: separate in-scope vs out-of-scope if applicable
        clinc_detail = None
        if task_id == "clinc_oos":
            # Label "42" is the OOS class (index 42 = "oos" in CLINC-plus)
            oos_label = None
            for lid, lname in labels.items():
                if "oos" in lname.lower():
                    oos_label = lid
                    break
            if oos_label:
                in_scope_mask = [t != oos_label for t in test_labels]
                oos_mask = [t == oos_label for t in test_labels]
                in_preds = [p for p, m in zip(pred_labels, in_scope_mask) if m]
                in_true = [t for t, m in zip(test_labels, in_scope_mask) if m]
                oos_preds = [p for p, m in zip(pred_labels, oos_mask) if m]
                oos_true = [t for t, m in zip(test_labels, oos_mask) if m]
                if in_true:
                    in_acc = accuracy_score(in_true, in_preds)
                    print(f"  In-scope accuracy: {in_acc:.4f} ({len(in_true)} examples)")
                if oos_true:
                    oos_acc = accuracy_score(oos_true, oos_preds)
                    print(f"  Out-of-scope accuracy: {oos_acc:.4f} ({len(oos_true)} examples)")
                    clinc_detail = {"in_scope_acc": round(in_acc, 4), "oos_acc": round(oos_acc, 4),
                                   "in_scope_n": len(in_true), "oos_n": len(oos_true)}

        # Baselines
        baseline_results = None
        if args.baselines:
            print(f"  Running baselines...")
            baseline_results = run_baselines(train_texts, train_labels, test_texts, test_labels)
            for name, br in baseline_results.items():
                print(f"    {name}: acc={br['accuracy']:.4f} F1={br['macro_f1']:.4f}")

        # Latency
        latency_results = None
        if args.latency:
            print(f"  Measuring latency...")
            latency_results = measure_latency(engine, task_id, test_texts)
            print(f"    Total p50={latency_results['total_p50_ms']:.1f}ms "
                  f"p95={latency_results['total_p95_ms']:.1f}ms")
            print(f"    Embed p50={latency_results['embed_p50_ms']:.1f}ms "
                  f"Clf p50={latency_results['clf_p50_ms']:.4f}ms")

        # Save per-example predictions
        preds_path = out_dir / f"{task_id}_predictions.jsonl"
        with open(preds_path, "w") as f:
            for text, true_label, pred in zip(test_texts, test_labels, predictions):
                human_true = labels.get(true_label, true_label)
                f.write(json.dumps({
                    "text": text[:500],
                    "true_label_id": true_label,
                    "true_label": human_true,
                    "predicted_id": pred["label_id"],
                    "predicted": pred["label"],
                    "correct": true_label == pred["label_id"],
                    "confidence": pred["confidence"],
                }) + "\n")

        result = {
            "task_id": task_id,
            "name": cap.name,
            "n_classes": cap.n_classes,
            "task_type": cap.task_type,
            "hf_source": hf_source,
            "hf_config": hf_config,
            "eval_split": split_name,
            "test_examples": len(test_texts),
            "train_examples": len(train_texts),
            "max_test": args.max_test,
            "sampling_seed": 42,
            "filter_label": filter_label,
            "accuracy": round(accuracy, 6),
            "accuracy_ci_lo": round(acc_lo, 6),
            "accuracy_ci_hi": round(acc_hi, 6),
            "macro_f1": round(macro_f1, 6),
            "train_accuracy": manifest["train_accuracy"],
            "build_test_accuracy": build_acc,
            "encoder": manifest["encoder"],
            "license": manifest["license"],
            "notes": notes,
            "baselines": baseline_results,
            "latency": latency_results,
            "clinc_detail": clinc_detail,
        }
        results.append(result)

    # Summary table
    print(f"\n{'='*90}")
    print("EVALUATION SUMMARY")
    print(f"{'='*90}")
    header = f"{'Task':<25s} {'Cls':>4s} {'N':>5s} {'Acc':>8s} {'95% CI':>17s} {'F1':>8s}"
    if args.baselines:
        header += f" {'Maj':>6s} {'TfIdf':>6s}"
    print(header)
    print("-" * len(header))

    for r in sorted(results, key=lambda x: -x["accuracy"]):
        row = (f"{r['task_id']:<25s} {r['n_classes']:>4d} {r['test_examples']:>5d} "
               f"{r['accuracy']:>8.4f} [{r['accuracy_ci_lo']:.4f},{r['accuracy_ci_hi']:.4f}] "
               f"{r['macro_f1']:>8.4f}")
        if args.baselines and r["baselines"]:
            row += (f" {r['baselines']['majority']['accuracy']:>6.3f}"
                    f" {r['baselines']['tfidf_lr']['accuracy']:>6.3f}")
        print(row)

    if results:
        avg_acc = np.mean([r["accuracy"] for r in results])
        print(f"\n{'Average':<25s} {'':>4s} {'':>5s} {avg_acc:>8.4f}")

    # Hardware info
    import psutil
    import torch as _torch
    hw = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "python": platform.python_version(),
        "torch_version": _torch.__version__,
        "device": args.device,
        "torch_threads": _torch.get_num_threads(),
        "torch_interop_threads": _torch.get_num_interop_threads(),
        "omp_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
    }

    # Artifact sizes
    pack_size = sum(f.stat().st_size for f in pack_dir.rglob("*") if f.is_file())
    encoder_size_note = "~1.2 GB on disk (BAAI/bge-large-en-v1.5)"

    # Save full results
    eval_output = {
        "results": results,
        "encoder": engine.encoder_name,
        "pack_dir": str(pack_dir),
        "pack_total_bytes": pack_size,
        "encoder_size": encoder_size_note,
        "encoder_load_time_s": round(engine.encoder_load_time, 2),
        "hardware": hw,
        "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": ("Independent evaluation of frozen model pack artifacts. "
                 "No retraining. Same random seed (42) for test set sampling. "
                 "TF-IDF+LR baseline tuned via 3-fold CV on training data."),
    }

    results_path = out_dir / "benchmark.json"
    with open(results_path, "w") as f:
        json.dump(eval_output, f, indent=2)
    print(f"\nSaved to {results_path}")

    # Print sizes
    print(f"\nArtifact sizes:")
    for task_id in sorted(engine.capabilities.keys()):
        model_path = pack_dir / task_id / "model.pkl"
        if model_path.exists():
            size_kb = model_path.stat().st_size / 1024
            print(f"  {task_id}: {size_kb:.0f} KB")
    print(f"  Total pack: {pack_size / 1024:.0f} KB")
    print(f"  Encoder: {encoder_size_note}")


if __name__ == "__main__":
    main()
