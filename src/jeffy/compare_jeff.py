"""Compare Jeff, QuickClassify, and tuned TF-IDF on identical evaluation examples.

Saves per-example predictions for all three systems. Runs QC and TF-IDF on
the exact same examples used for Jeff (not the full 2000-example eval sets).
Reports failures explicitly. Persists predictions for reuse.

Usage:
    python -m jeffy.compare_jeff
    python -m jeffy.compare_jeff --tasks banking77 ag_news sst2
    python -m jeffy.compare_jeff --reuse  # skip Jeff, reuse saved predictions
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from .build_pack import DATASETS, load_and_split
from .engine import Engine


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def build_jeff_question(labels: dict[str, str], task_type: str, description: str):
    """Build a Jeff-format question with clear task instructions."""
    if task_type == "noul" and len(labels) == 2:
        label_vals = list(labels.values())
        return {
            "type": "noul",
            "instructions": description,
            "criteria": {"true": label_vals[1], "false": label_vals[0]},
        }
    else:
        criteria = {v: v for v in labels.values()}
        return {
            "type": "choice",
            "instructions": description,
            "criteria": criteria,
        }


def run_jeff_on_examples(model, test_texts, test_labels, labels, question,
                         task_id, reverse_map):
    """Run Jeff on all examples, saving per-example results including errors."""
    from jeff.model import answer as jeff_answer

    predictions = []
    for i, (text, true_label) in enumerate(zip(test_texts, test_labels)):
        text_id = text_hash(text)
        t0 = time.perf_counter()
        error = None
        pred_id = None
        pred_human = None
        raw_answer = None

        try:
            row = {"state": text, "question": question}
            with torch.no_grad():
                probs_list = model.predict([row])

            if not probs_list or not probs_list[0]:
                error = "empty prediction"
            else:
                probs = probs_list[0]
                ans = jeff_answer(question, probs)
                raw_answer = {k: v for k, v in ans.items()
                              if k not in ("probabilities", "legend")}

                if question["type"] == "choice":
                    chosen = ans.get("choice", "")
                    pred_id = reverse_map.get(chosen, None)
                    pred_human = chosen
                    if pred_id is None:
                        error = f"Jeff returned '{chosen}' which is not in label map"
                elif question["type"] == "noul":
                    noul_prob = ans.get("noul", 0.5)
                    pred_id = "1" if noul_prob > 0.5 else "0"
                    pred_human = labels.get(pred_id, pred_id)
                    raw_answer["noul_prob"] = noul_prob

        except Exception as e:
            error = str(e)

        latency_ms = (time.perf_counter() - t0) * 1000
        true_human = labels.get(true_label, true_label)

        predictions.append({
            "index": i,
            "text_hash": text_id,
            "text": text[:500],
            "true_label_id": true_label,
            "true_label": true_human,
            "pred_label_id": pred_id,
            "pred_label": pred_human,
            "correct": pred_id == true_label if pred_id is not None else None,
            "error": error,
            "latency_ms": round(latency_ms, 1),
            "raw_answer": raw_answer,
        })

        if (i + 1) % 50 == 0:
            n_ok = sum(1 for p in predictions if p["error"] is None)
            print(f"    {i+1}/{len(test_texts)} ({n_ok} ok, {i+1-n_ok} errors)")

    return predictions


def run_qc_on_examples(engine, task_id, test_texts, test_labels, labels):
    """Run QuickClassify on the exact same examples."""
    predictions = []
    for i, (text, true_label) in enumerate(zip(test_texts, test_labels)):
        r = engine.predict(task_id, text)
        true_human = labels.get(true_label, true_label)
        predictions.append({
            "index": i,
            "text_hash": text_hash(text),
            "true_label_id": true_label,
            "true_label": true_human,
            "pred_label_id": r["label_id"],
            "pred_label": r["label"],
            "correct": r["label_id"] == true_label,
            "confidence": r["confidence"],
            "latency_ms": r["latency_ms"],
            "error": None,
        })
    return predictions


def run_tfidf_on_examples(train_texts, train_labels, test_texts, test_labels, labels):
    """Run tuned TF-IDF+LR on the exact same examples."""
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
    grid = GridSearchCV(pipe, param_grid, cv=3, scoring="accuracy", n_jobs=1, refit=True)
    grid.fit(train_texts, train_labels)
    preds = grid.predict(test_texts)

    predictions = []
    for i, (text, true_label, pred) in enumerate(zip(test_texts, test_labels, preds)):
        true_human = labels.get(true_label, true_label)
        pred_human = labels.get(pred, pred)
        predictions.append({
            "index": i,
            "text_hash": text_hash(text),
            "true_label_id": true_label,
            "pred_label_id": pred,
            "pred_label": pred_human,
            "correct": pred == true_label,
            "error": None,
        })
    return predictions, grid.best_params_


def compute_paired_accuracy(preds_a, preds_b, name_a, name_b):
    """Compute accuracy and paired bootstrap CI for the difference."""
    assert len(preds_a) == len(preds_b)
    correct_a = np.array([1 if p["correct"] else 0 for p in preds_a])
    correct_b = np.array([1 if p["correct"] else 0 for p in preds_b])
    # Include failures as incorrect
    for i, p in enumerate(preds_a):
        if p.get("error") is not None:
            correct_a[i] = 0
    for i, p in enumerate(preds_b):
        if p.get("error") is not None:
            correct_b[i] = 0

    acc_a = correct_a.mean()
    acc_b = correct_b.mean()
    diff = correct_a - correct_b

    # Bootstrap CI for the difference
    rng = np.random.RandomState(42)
    n = len(diff)
    boot_diffs = [rng.choice(diff, size=n, replace=True).mean() for _ in range(2000)]
    lo, hi = np.percentile(boot_diffs, [2.5, 97.5])

    return {
        f"{name_a}_acc": round(float(acc_a), 6),
        f"{name_b}_acc": round(float(acc_b), 6),
        "delta": round(float(acc_a - acc_b), 6),
        "ci_lo": round(float(lo), 6),
        "ci_hi": round(float(hi), 6),
        "n": n,
    }


def main():
    parser = argparse.ArgumentParser(description="Three-way comparison on identical examples")
    parser.add_argument("--tasks", nargs="+", default=["ag_news", "sst2", "banking77"])
    parser.add_argument("--jeff-checkpoint", default="data/jeff_model")
    parser.add_argument("--pack-dir", default="data/model_pack")
    parser.add_argument("--max-test", type=int, default=200)
    parser.add_argument("--out", default="data/eval_results")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reuse", action="store_true", help="Reuse saved Jeff predictions")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pin threads for fair latency comparison
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    print(f"Device: {args.device}")
    print(f"Threads: torch={torch.get_num_threads()}, OMP={os.environ.get('OMP_NUM_THREADS')}")

    # Load Jeff model (unless reusing)
    jeff_model = None
    if not args.reuse:
        print(f"\nLoading Jeff model from {args.jeff_checkpoint}...")
        from jeff.models import load_decision_model
        t0 = time.perf_counter()
        jeff_model = load_decision_model(args.jeff_checkpoint, device=args.device)
        jeff_load_time = time.perf_counter() - t0
        print(f"  Loaded in {jeff_load_time:.1f}s")
    else:
        jeff_load_time = 0.0
        print("Reusing saved Jeff predictions")

    # Load QuickClassify engine
    print(f"\nLoading QuickClassify from {args.pack_dir}...")
    engine = Engine(str(args.pack_dir), device=args.device)
    engine.load()
    print(f"  {len(engine.capabilities)} capabilities, loaded in {engine.encoder_load_time:.1f}s")

    pack_dir = Path(args.pack_dir)
    all_results = []

    for task_id in args.tasks:
        if task_id not in DATASETS:
            print(f"\nSKIP {task_id}: no dataset config")
            continue

        ds_config = DATASETS[task_id]
        manifest_path = pack_dir / task_id / "manifest.json"
        if not manifest_path.exists():
            print(f"\nSKIP {task_id}: no model pack artifact")
            continue

        with open(manifest_path) as f:
            manifest = json.load(f)
        labels = manifest["labels"]
        reverse_map = {v: k for k, v in labels.items()}

        print(f"\n{'='*60}")
        print(f"{task_id}: {manifest['description']}")

        # Load data — same max_test and seed=42
        try:
            train_texts, train_labels, test_texts, test_labels, _ = \
                load_and_split(task_id, ds_config, max_test=args.max_test)
        except Exception as e:
            print(f"  FAILED to load: {e}")
            continue

        print(f"  Examples: {len(test_texts)} test, {len(train_texts)} train")
        print(f"  Classes: {len(set(test_labels))}, Options to Jeff: {len(labels)}")

        question = build_jeff_question(labels, manifest["task_type"], manifest["description"])
        print(f"  Question: type={question['type']}, instructions='{question.get('instructions','')[:60]}...'")

        # --- Jeff ---
        jeff_preds_path = out_dir / f"jeff_{task_id}_predictions.jsonl"
        if args.reuse and jeff_preds_path.exists():
            print(f"  Jeff: loading saved predictions from {jeff_preds_path}")
            jeff_preds = []
            with open(jeff_preds_path) as f:
                for line in f:
                    jeff_preds.append(json.loads(line))
            # Verify same examples
            saved_hashes = [p["text_hash"] for p in jeff_preds]
            current_hashes = [text_hash(t) for t in test_texts]
            if saved_hashes != current_hashes:
                print(f"  WARNING: saved predictions don't match current examples, re-running")
                jeff_preds = None
            else:
                print(f"  Jeff: reusing {len(jeff_preds)} predictions")
        else:
            jeff_preds = None

        if jeff_preds is None and jeff_model is not None:
            print(f"  Jeff: running {len(test_texts)} predictions...")
            jeff_preds = run_jeff_on_examples(
                jeff_model, test_texts, test_labels, labels, question,
                task_id, reverse_map)
            # Save per-example predictions
            with open(jeff_preds_path, "w") as f:
                for p in jeff_preds:
                    f.write(json.dumps(p) + "\n")
            print(f"  Jeff: saved to {jeff_preds_path}")
        elif jeff_preds is None:
            print(f"  Jeff: skipped (no model loaded and no saved predictions)")
            jeff_preds = []

        # --- QuickClassify ---
        print(f"  QC: running {len(test_texts)} predictions...")
        qc_preds = run_qc_on_examples(engine, task_id, test_texts, test_labels, labels)
        qc_preds_path = out_dir / f"qc_{task_id}_predictions.jsonl"
        with open(qc_preds_path, "w") as f:
            for p in qc_preds:
                f.write(json.dumps(p) + "\n")

        # --- TF-IDF+LR ---
        print(f"  TF-IDF: tuning on {len(train_texts)} train, predicting {len(test_texts)} test...")
        tfidf_preds, tfidf_params = run_tfidf_on_examples(
            train_texts, train_labels, test_texts, test_labels, labels)
        print(f"  TF-IDF best params: {tfidf_params}")

        # --- Comparison ---
        n_total = len(test_texts)

        # Jeff stats (include failures in headline)
        jeff_errors = sum(1 for p in jeff_preds if p.get("error") is not None)
        jeff_correct = sum(1 for p in jeff_preds if p.get("correct") is True)
        jeff_acc_all = jeff_correct / n_total  # failures count as wrong
        jeff_acc_valid = jeff_correct / (n_total - jeff_errors) if jeff_errors < n_total else 0
        jeff_f1 = 0.0
        if jeff_preds:
            # For F1, treat errors as a special wrong prediction
            jp = [p["pred_label_id"] if p["pred_label_id"] is not None else "__ERROR__"
                  for p in jeff_preds]
            jeff_f1 = f1_score(test_labels, jp, average="macro", zero_division=0)

        qc_acc = sum(1 for p in qc_preds if p["correct"]) / n_total
        qc_f1 = f1_score(
            [p["true_label_id"] for p in qc_preds],
            [p["pred_label_id"] for p in qc_preds],
            average="macro", zero_division=0)

        tfidf_acc = sum(1 for p in tfidf_preds if p["correct"]) / n_total
        tfidf_f1 = f1_score(
            [p["true_label_id"] for p in tfidf_preds],
            [p["pred_label_id"] for p in tfidf_preds],
            average="macro", zero_division=0)

        # Paired CIs
        paired_jeff_qc = compute_paired_accuracy(qc_preds, jeff_preds, "qc", "jeff") if jeff_preds else None
        paired_qc_tfidf = compute_paired_accuracy(qc_preds, tfidf_preds, "qc", "tfidf")

        # Latencies
        jeff_lats = [p["latency_ms"] for p in jeff_preds if p.get("latency_ms")]
        qc_lats = [p["latency_ms"] for p in qc_preds]

        print(f"\n  Results on {n_total} identical examples:")
        print(f"    Jeff:   acc={jeff_acc_all:.4f} (errors={jeff_errors}/{n_total}, "
              f"valid-only={jeff_acc_valid:.4f}) F1={jeff_f1:.4f} "
              f"p50={np.percentile(jeff_lats, 50):.0f}ms" if jeff_lats else "")
        print(f"    QC:     acc={qc_acc:.4f} F1={qc_f1:.4f} "
              f"p50={np.percentile(qc_lats, 50):.0f}ms")
        print(f"    TF-IDF: acc={tfidf_acc:.4f} F1={tfidf_f1:.4f}")
        if paired_jeff_qc:
            print(f"    QC−Jeff: {paired_jeff_qc['delta']:+.4f} "
                  f"[{paired_jeff_qc['ci_lo']:+.4f}, {paired_jeff_qc['ci_hi']:+.4f}]")
        print(f"    QC−TfIdf: {paired_qc_tfidf['delta']:+.4f} "
              f"[{paired_qc_tfidf['ci_lo']:+.4f}, {paired_qc_tfidf['ci_hi']:+.4f}]")

        result = {
            "task_id": task_id,
            "n_examples": n_total,
            "sampling_seed": 42,
            "jeff": {
                "accuracy_all": round(jeff_acc_all, 6),
                "accuracy_valid": round(jeff_acc_valid, 6),
                "macro_f1": round(jeff_f1, 6),
                "errors": jeff_errors,
                "latency_p50_ms": round(np.percentile(jeff_lats, 50), 1) if jeff_lats else None,
                "latency_p95_ms": round(np.percentile(jeff_lats, 95), 1) if jeff_lats else None,
            },
            "jeffy": {
                "accuracy": round(qc_acc, 6),
                "macro_f1": round(qc_f1, 6),
                "latency_p50_ms": round(np.percentile(qc_lats, 50), 1),
                "latency_p95_ms": round(np.percentile(qc_lats, 95), 1),
            },
            "tfidf_lr": {
                "accuracy": round(tfidf_acc, 6),
                "macro_f1": round(tfidf_f1, 6),
                "best_params": tfidf_params,
            },
            "paired_qc_vs_jeff": paired_jeff_qc,
            "paired_qc_vs_tfidf": paired_qc_tfidf,
        }
        all_results.append(result)

    # Summary table
    print(f"\n{'='*90}")
    print("THREE-WAY COMPARISON (identical examples per task)")
    print(f"{'='*90}")
    print(f"{'Task':<20s} {'N':>4s} {'Jeff':>8s} {'err':>4s} {'QC':>8s} {'TfIdf':>8s} "
          f"{'QC−Jeff':>8s} {'CI':>20s}")
    print("-" * 85)
    for r in all_results:
        j = r["jeff"]
        q = r["jeffy"]
        t = r["tfidf_lr"]
        p = r.get("paired_qc_vs_jeff", {})
        ci = f"[{p.get('ci_lo',0):+.4f},{p.get('ci_hi',0):+.4f}]" if p else "—"
        print(f"{r['task_id']:<20s} {r['n_examples']:>4d} {j['accuracy_all']:>8.4f} "
              f"{j['errors']:>4d} {q['accuracy']:>8.4f} {t['accuracy']:>8.4f} "
              f"{p.get('delta',0):>+8.4f} {ci:>20s}")

    # Save
    out_path = out_dir / "jeff_comparison.json"
    with open(out_path, "w") as f:
        json.dump({
            "results": all_results,
            "jeff_model": "mstrasser/Jeff-Qwen3.5-0.8B",
            "jeff_device": args.device,
            "jeff_load_time_s": round(jeff_load_time, 1),
            "qc_encoder": engine.encoder_name,
            "qc_device": args.device,
            "qc_encoder_load_time_s": round(engine.encoder_load_time, 1),
            "threads": {
                "torch": torch.get_num_threads(),
                "omp": os.environ.get("OMP_NUM_THREADS"),
            },
            "comparison_type": "zero-shot Jeff vs task-trained QuickClassify vs tuned TF-IDF+LR",
            "note": (
                "Jeff receives no task-specific training data. QuickClassify heads are "
                "trained on up to 10,000 examples per task. TF-IDF+LR is tuned via 3-fold CV "
                "on the same training data. All three are evaluated on identical test examples. "
                "Jeff failures (errors) count as incorrect in headline accuracy."
            ),
            "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(f"Per-example predictions: {out_dir}/jeff_*_predictions.jsonl, qc_*_predictions.jsonl")


if __name__ == "__main__":
    main()
