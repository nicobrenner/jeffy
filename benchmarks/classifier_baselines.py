#!/usr/bin/env python3
"""
JeffyBench classifier baselines: SetFit and FastText on sst2, ag_news, banking77.

Rewrite of the (never committed) runner behind data/eval_results/setfit_fasttext_benchmark.json.

Protocol (matches benchmarks/jeffybench.py):
  * Test data: the same split jeffybench.py uses (sst2/validation, ag_news/test, banking77/test),
    100 held-out samples drawn with ds.shuffle(seed=42).select(range(100)). Those 100 give the
    top-level accuracy / macro_f1 / latency numbers. Accuracy and macro-F1 on the FULL split are
    reported alongside under "full_split" (lower variance; the published SetFit numbers such as
    0.8142 on sst2 are k/872, i.e. the original runner scored the full split there).
  * Training: standard training splits. FastText trains on the full training split. SetFit is a
    few-shot method whose contrastive pair set grows quadratically with the training set, so it
    trains on a class-balanced subsample (--shots per class, default 20) with num_iterations
    pair sampling (SetFit paper default R=20); everything else is the SetFit library default.
  * Latency: CPU, batch size 1, after warm-up, p50 over the 100 evaluation texts.
  * Model size: SetFit = sum of parameter bytes (body + logistic head); FastText = saved .bin size.

Usage:
    python benchmarks/classifier_baselines.py                # everything
    python benchmarks/classifier_baselines.py --tasks sst2 --skip-setfit
    python benchmarks/classifier_baselines.py --setfit-model sentence-transformers/all-MiniLM-L6-v2

Writes data/eval_results/setfit_fasttext_benchmark_rerun.json (never the published file).
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PUBLISHED = REPO / "data" / "eval_results" / "setfit_fasttext_benchmark.json"
DEFAULT_OUTPUT = REPO / "data" / "eval_results" / "setfit_fasttext_benchmark_rerun.json"

# Same datasets/splits as jeffybench.py. banking77 is served from the HF parquet mirror
# (legacy-datasets/banking77) because PolyAI/banking77 is a script dataset, which
# `datasets` >= 4 refuses to load; rows, order and ClassLabel names are identical.
TASKS = {
    "sst2": {
        "hf_dataset": "stanfordnlp/sst2",
        "train_split": "train",
        "eval_split": "validation",
        "text_col": "sentence",
        "label_col": "label",
    },
    "ag_news": {
        "hf_dataset": "fancyzhx/ag_news",
        "train_split": "train",
        "eval_split": "test",
        "text_col": "text",
        "label_col": "label",
    },
    "banking77": {
        "hf_dataset": "legacy-datasets/banking77",
        "jeffybench_dataset": "PolyAI/banking77",
        "train_split": "train",
        "eval_split": "test",
        "text_col": "text",
        "label_col": "label",
    },
}

SEED = 42


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_task(task_id, n_samples):
    from datasets import load_dataset

    cfg = TASKS[task_id]
    train = load_dataset(cfg["hf_dataset"], split=cfg["train_split"])
    full = load_dataset(cfg["hf_dataset"], split=cfg["eval_split"])
    subset = full.shuffle(seed=SEED).select(range(min(n_samples, len(full))))  # == jeffybench.py
    return {
        "train_texts": train[cfg["text_col"]],
        "train_labels": train[cfg["label_col"]],
        "full_texts": full[cfg["text_col"]],
        "full_labels": full[cfg["label_col"]],
        "sub_texts": subset[cfg["text_col"]],
        "sub_labels": subset[cfg["label_col"]],
        "label_names": full.features[cfg["label_col"]].names,
        "n_train": len(train),
        "n_full": len(full),
        "n_sub": len(subset),
    }


def scores(y_true, y_pred):
    from sklearn.metrics import accuracy_score, f1_score

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }


def latency_ms(predict_one, texts, warmup=10):
    """p50/p95 wall-clock ms for batch-size-1 prediction, after warm-up."""
    for t in texts[:warmup]:
        predict_one(t)
    lat = []
    for t in texts:
        t0 = time.perf_counter()
        predict_one(t)
        lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()
    return {
        "latency_p50_ms": round(statistics.median(lat), 2),
        "latency_p95_ms": round(lat[int(len(lat) * 0.95)], 2),
    }


# ----------------------------------------------------------------------------- SetFit


def pick_train_device():
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def run_setfit(task_id, data, model_name, shots, num_iterations, train_device, batch_size):
    import numpy as np
    import torch
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset

    train_ds = Dataset.from_dict({"text": data["train_texts"], "label": data["train_labels"]})
    few = sample_dataset(train_ds, label_column="label", num_samples=shots, seed=SEED)
    log(f"  SetFit {task_id}: {len(few)} training examples ({shots}/class), "
        f"num_iterations={num_iterations}, device={train_device}")

    model = SetFitModel.from_pretrained(model_name, device=train_device)
    with tempfile.TemporaryDirectory() as ckpt_dir:
        args = TrainingArguments(
            output_dir=ckpt_dir,
            num_iterations=num_iterations,
            batch_size=batch_size,
            report_to="none",
            show_progress_bar=False,
            logging_steps=200,
            seed=SEED,
        )
        trainer = Trainer(model=model, args=args, train_dataset=few)
        t0 = time.perf_counter()
        trainer.train()
        train_time = time.perf_counter() - t0
    log(f"  SetFit {task_id}: trained in {train_time:.1f}s")

    # Accuracy on the full split and the 100-sample jeffybench subset (batched, train device).
    full_pred = [int(p) for p in model.predict(data["full_texts"], batch_size=64)]
    sub_pred = [int(p) for p in model.predict(data["sub_texts"], batch_size=64)]
    full_scores = scores(data["full_labels"], full_pred)
    sub_scores = scores(data["sub_labels"], sub_pred)

    # Latency: CPU, batch 1.
    model.model_body.to("cpu")
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    lat = latency_ms(lambda t: model.predict([t]), data["sub_texts"])

    body_bytes = sum(p.numel() * p.element_size() for p in model.model_body.parameters())
    head = model.model_head
    head_bytes = 0
    if hasattr(head, "coef_"):
        head_bytes = int(np.asarray(head.coef_).nbytes + np.asarray(head.intercept_).nbytes)
    elif hasattr(head, "parameters"):
        head_bytes = sum(p.numel() * p.element_size() for p in head.parameters())
    n_params = sum(p.numel() for p in model.model_body.parameters())

    result = {
        **sub_scores,
        **lat,
        "training_time_s": round(train_time, 1),
        "model_size_mb": round((body_bytes + head_bytes) / 1e6, 1),
        "model_size_mib": round((body_bytes + head_bytes) / 2**20, 1),
        "full_split": {"n": data["n_full"], **full_scores},
        "n_train_examples": len(few),
        "body_params": n_params,
    }
    del model, trainer
    return result


# ----------------------------------------------------------------------------- FastText


def fasttext_line(text, label):
    return f"__label__{label} {' '.join(str(text).split())}\n"


def run_fasttext(task_id, data, cfg_name, params, train_file):
    import fasttext

    log(f"  FastText[{cfg_name}] {task_id}: {params}")
    t0 = time.perf_counter()
    model = fasttext.train_supervised(input=str(train_file), verbose=0, **params)
    train_time = time.perf_counter() - t0

    def predict_one(text):
        # fasttext-wheel 0.9.2's single-string predict() calls np.array(copy=False), which
        # raises under NumPy 2; use the bound C++ predict directly (same work, no numpy).
        preds = model.f.predict(" ".join(str(text).split()) + "\n", 1, 0.0, "strict")
        return int(preds[0][1].replace("__label__", ""))

    def predict_many(texts):
        clean = [" ".join(str(t).split()) for t in texts]
        labels, _ = model.predict(clean)
        return [int(l[0].replace("__label__", "")) for l in labels]

    full_scores = scores(data["full_labels"], predict_many(data["full_texts"]))
    sub_scores = scores(data["sub_labels"], predict_many(data["sub_texts"]))
    lat = latency_ms(predict_one, data["sub_texts"])

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        bin_path = f.name
    model.save_model(bin_path)
    size_mb = os.path.getsize(bin_path) / 1e6
    os.remove(bin_path)

    return {
        **sub_scores,
        **lat,
        "training_time_s": round(train_time, 1),
        "model_size_mb": round(size_mb, 1),
        "model_size_mib": round(size_mb * 1e6 / 2**20, 1),
        "full_split": {"n": data["n_full"], **full_scores},
        "n_train_examples": data["n_train"],
        "params": params,
    }


FASTTEXT_CONFIGS = {
    # fasttext.train_supervised defaults (what the published 0.57/0.37/0.011 numbers look like)
    "fasttext": {"lr": 0.1, "epoch": 5, "wordNgrams": 1, "dim": 100},
    # diagnostic: defaults + bigrams. wordNgrams>1 switches on the 2M-bucket hash table, which is
    # the only way to get the published 763 MB (= 2e6 x 100 x 4 bytes in MiB) model size.
    "fasttext_defaults_bigram": {"lr": 0.1, "epoch": 5, "wordNgrams": 2, "dim": 100},
    # the tuned recipe from the FastText text-classification tutorial
    "fasttext_tuned": {"lr": 1.0, "epoch": 25, "wordNgrams": 2, "dim": 100},
}


# ----------------------------------------------------------------------------- main


def lib_versions():
    out = {"python": platform.python_version()}
    for mod in ["setfit", "sentence_transformers", "transformers", "torch", "datasets", "sklearn", "fasttext", "numpy"]:
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception as e:  # noqa: BLE001
            out[mod] = f"missing ({e.__class__.__name__})"
    return out


def cpu_name():
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:  # noqa: BLE001
        return platform.processor() or platform.machine()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", default="sst2,ag_news,banking77")
    p.add_argument("--samples", type=int, default=100, help="held-out samples per task (jeffybench default 100)")
    p.add_argument("--setfit-model", default="sentence-transformers/paraphrase-mpnet-base-v2",
                   help="SetFit library default sentence-transformer")
    p.add_argument("--shots", type=int, default=20, help="SetFit training examples per class")
    p.add_argument("--num-iterations", type=int, default=20, help="SetFit contrastive pair iterations (paper default R=20)")
    p.add_argument("--setfit-batch-size", type=int, default=16)
    p.add_argument("--skip-setfit", action="store_true")
    p.add_argument("--skip-fasttext", action="store_true")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = p.parse_args()

    output = Path(args.output).resolve()
    if output == PUBLISHED.resolve():
        sys.exit(f"Refusing to overwrite the published results file {PUBLISHED}")

    task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
    train_device = pick_train_device() if not args.skip_setfit else None
    log(f"tasks={task_ids} setfit_model={args.setfit_model} shots={args.shots} "
        f"num_iterations={args.num_iterations} train_device={train_device}")

    results = {"setfit": {}, **{k: {} for k in FASTTEXT_CONFIGS}}
    counts = {}
    scratch = Path(tempfile.mkdtemp(prefix="classifier_baselines_"))

    for task_id in task_ids:
        log(f"=== {task_id}: loading {TASKS[task_id]['hf_dataset']} ===")
        data = load_task(task_id, args.samples)
        counts[task_id] = {
            "n_train": data["n_train"],
            "eval_split": TASKS[task_id]["eval_split"],
            "n_eval_full": data["n_full"],
            "n_eval_jeffybench": data["n_sub"],
            "n_classes": len(data["label_names"]),
        }
        log(f"  train={data['n_train']} eval_full={data['n_full']} eval_subset={data['n_sub']} "
            f"classes={len(data['label_names'])}")

        if not args.skip_fasttext:
            train_file = scratch / f"{task_id}.train.txt"
            with open(train_file, "w") as f:
                for t, l in zip(data["train_texts"], data["train_labels"]):
                    f.write(fasttext_line(t, l))
            for cfg_name, params in FASTTEXT_CONFIGS.items():
                r = run_fasttext(task_id, data, cfg_name, params, train_file)
                results[cfg_name][task_id] = r
                log(f"  {cfg_name} {task_id}: acc={r['accuracy']} f1={r['macro_f1']} "
                    f"(full {r['full_split']['accuracy']}/{r['full_split']['macro_f1']}) "
                    f"p50={r['latency_p50_ms']}ms train={r['training_time_s']}s size={r['model_size_mb']}MB")
            train_file.unlink()

        if not args.skip_setfit:
            r = run_setfit(task_id, data, args.setfit_model, args.shots, args.num_iterations,
                           train_device, args.setfit_batch_size)
            results["setfit"][task_id] = r
            log(f"  setfit {task_id}: acc={r['accuracy']} f1={r['macro_f1']} "
                f"(full {r['full_split']['accuracy']}/{r['full_split']['macro_f1']}) "
                f"p50={r['latency_p50_ms']}ms train={r['training_time_s']}s size={r['model_size_mb']}MB")

        # Save after every task so a crash keeps earlier results.
        write_output(output, args, results, counts, train_device)

    log(f"Results saved to {output}")


def write_output(output, args, results, counts, train_device):
    published = json.loads(PUBLISHED.read_text()) if PUBLISHED.exists() else {}
    body = {
        "jeffy": {**published.get("jeffy", {}), "_note": "copied from setfit_fasttext_benchmark.json, not rerun"},
        **{k: v for k, v in results.items() if v},
        "config": {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "script": "benchmarks/classifier_baselines.py",
            "protocol": {
                "eval_samples": "top-level metrics on the jeffybench.py held-out subset "
                                f"(split.shuffle(seed={SEED}).select(range({args.samples}))); "
                                "full_split = the entire evaluation split",
                "latency": "CPU, batch size 1, 10 warm-up calls, p50 over the held-out subset",
                "seed": SEED,
            },
            "datasets": {
                t: {
                    "hf_dataset": TASKS[t]["hf_dataset"],
                    "jeffybench_dataset": TASKS[t].get("jeffybench_dataset", TASKS[t]["hf_dataset"]),
                    "train_split": TASKS[t]["train_split"],
                    **counts[t],
                }
                for t in counts
            },
            "setfit": None if args.skip_setfit else {
                "model": args.setfit_model,
                "train_protocol": f"{args.shots} examples per class (setfit.sample_dataset, seed {SEED}); "
                                  f"TrainingArguments defaults except num_iterations={args.num_iterations}, "
                                  f"batch_size={args.setfit_batch_size}; logistic-regression head (default)",
                "why_subsampled": "SetFit contrastive training is quadratic in the training-set size; the full "
                                  "training splits (67k/120k/10k examples) are far beyond a bounded run, so the "
                                  "few-shot protocol from the SetFit paper is used",
                "train_device": train_device,
                "latency_device": "cpu",
                "model_size": "sum of parameter bytes of the sentence-transformer body (fp32) + logistic head",
            },
            "fasttext": None if args.skip_fasttext else {
                "train_protocol": "full training split, text whitespace-normalised, no lowercasing",
                "configs": FASTTEXT_CONFIGS,
                "model_size": "size of the saved .bin (default bucket=2,000,000 hashes x dim 100 dominates)",
                "device": "cpu",
            },
            "hardware": {"cpu": cpu_name(), "machine": platform.machine(), "os": platform.platform(),
                         "cpu_threads": os.cpu_count()},
            "library_versions": lib_versions(),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(body, indent=2))


if __name__ == "__main__":
    main()
