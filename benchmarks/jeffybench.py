#!/usr/bin/env python3
"""
JeffyBench: Compare Jeffy classification vs local LLMs via Ollama.

Usage:
    pip install jeffy-classify datasets
    ollama pull phi4          # or any model you want to test
    python benchmarks/jeffybench.py

Options:
    --tasks sst2,ag_news,banking77    Comma-separated task list (default: all three)
    --models phi4,mistral             Comma-separated Ollama model names (default: phi4)
    --samples 100                     Number of test samples per task (default: 100)
    --output results.json             Output file (default: benchmarks/results.json)

Results are printed as tables and saved to JSON.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

TASKS = {
    "sst2": {
        "hf_dataset": "stanfordnlp/sst2",
        "hf_split": "validation",
        "text_col": "sentence",
        "label_col": "label",
        "label_map": {0: "negative", 1: "positive"},
        "n_classes": 2,
        "category": "sentiment",
    },
    "ag_news": {
        "hf_dataset": "fancyzhx/ag_news",
        "hf_split": "test",
        "text_col": "text",
        "label_col": "label",
        "label_map": {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
        "n_classes": 4,
        "category": "topic",
    },
    "banking77": {
        "hf_dataset": "PolyAI/banking77",
        "hf_split": "test",
        "text_col": "text",
        "label_col": "label",
        "label_map": None,
        "n_classes": 77,
        "category": "intent",
    },
}


def load_test_data(task_id, n_samples):
    from datasets import load_dataset

    cfg = TASKS[task_id]
    ds = load_dataset(cfg["hf_dataset"], split=cfg["hf_split"])
    if n_samples < len(ds):
        ds = ds.shuffle(seed=42).select(range(n_samples))

    if task_id == "banking77":
        label_names = ds.features["label"].names
        label_map = {i: name for i, name in enumerate(label_names)}
    else:
        label_map = cfg["label_map"]

    texts = ds[cfg["text_col"]]
    labels = [label_map[l] for l in ds[cfg["label_col"]]]
    return texts, labels, label_map


def run_jeffy(task_id, texts, labels):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    os.environ.setdefault("JEFFY_BACKEND", "onnx")
    from jeffy.engine import Engine

    engine = Engine()

    correct = 0
    latencies = []
    for text, expected in zip(texts, labels):
        t0 = time.perf_counter()
        result = engine.predict(task_id, text)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)
        if result["label"] == expected:
            correct += 1

    latencies.sort()
    return {
        "model": "Jeffy",
        "accuracy": correct / len(texts),
        "latency_p50_ms": round(latencies[len(latencies) // 2], 1),
        "latency_p95_ms": round(latencies[int(len(latencies) * 0.95)], 1),
        "n_samples": len(texts),
    }


def build_ollama_prompt(task_id, text, label_map):
    labels_str = ", ".join(sorted(set(label_map.values())))
    return (
        f"Classify the following text into EXACTLY ONE of these categories: {labels_str}\n\n"
        f"Text: {text}\n\n"
        f"Reply with ONLY the category name, nothing else."
    )


def run_ollama(model_name, task_id, texts, labels, label_map):
    import urllib.request

    correct = 0
    latencies = []
    valid_labels = set(label_map.values())

    for text, expected in zip(texts, labels):
        prompt = build_ollama_prompt(task_id, text, label_map)
        payload = json.dumps({
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 50},
        }).encode()

        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
            elapsed_ms = (time.perf_counter() - t0) * 1000

            raw = result.get("response", "").strip().strip("'\"").lower()
            predicted = None
            for lbl in valid_labels:
                if lbl.lower() == raw or lbl.lower() in raw:
                    predicted = lbl
                    break

            if predicted == expected:
                correct += 1
            latencies.append(elapsed_ms)
        except Exception as e:
            print(f"  Error on '{text[:40]}...': {e}", file=sys.stderr)
            latencies.append(120000)

    latencies.sort()
    return {
        "model": model_name,
        "accuracy": correct / len(texts) if texts else 0,
        "latency_p50_ms": round(latencies[len(latencies) // 2], 1) if latencies else 0,
        "latency_p95_ms": round(latencies[int(len(latencies) * 0.95)], 1) if latencies else 0,
        "n_samples": len(texts),
    }


def print_table(task_id, results):
    cfg = TASKS[task_id]
    print(f"\n  {task_id} ({cfg['n_classes']} classes — {cfg['category']})\n")
    print(f"  {'Model':<20} {'Accuracy':>8}  {'Latency p50':>11}  {'Latency p95':>11}")
    print(f"  {'─' * 20} {'─' * 8}  {'─' * 11}  {'─' * 11}")
    for r in sorted(results, key=lambda x: -x["accuracy"]):
        lat = (
            f"{r['latency_p50_ms']:.0f}ms"
            if r["latency_p50_ms"] < 1000
            else f"{r['latency_p50_ms'] / 1000:.1f}s"
        )
        lat95 = (
            f"{r['latency_p95_ms']:.0f}ms"
            if r["latency_p95_ms"] < 1000
            else f"{r['latency_p95_ms'] / 1000:.1f}s"
        )
        print(f"  {r['model']:<20} {r['accuracy']:>7.1%}  {lat:>11}  {lat95:>11}")


def main():
    parser = argparse.ArgumentParser(description="JeffyBench: Jeffy vs LLMs on classification")
    parser.add_argument("--tasks", default="sst2,ag_news,banking77")
    parser.add_argument("--models", default="phi4")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output", default="benchmarks/results.json")
    args = parser.parse_args()

    task_ids = [t.strip() for t in args.tasks.split(",")]
    model_names = [m.strip() for m in args.models.split(",")]

    all_results = {}
    for task_id in task_ids:
        if task_id not in TASKS:
            print(f"Unknown task: {task_id}. Available: {', '.join(TASKS.keys())}")
            continue

        print(f"\n{'=' * 60}")
        print(f"  {task_id} — loading {args.samples} test samples...")
        print(f"{'=' * 60}")

        texts, labels, label_map = load_test_data(task_id, args.samples)
        results = []

        print(f"  Running Jeffy...")
        jeffy_result = run_jeffy(task_id, texts, labels)
        results.append(jeffy_result)
        print(f"    {jeffy_result['accuracy']:.1%} accuracy, {jeffy_result['latency_p50_ms']:.0f}ms p50")

        for model_name in model_names:
            print(f"  Running {model_name}...")
            ollama_result = run_ollama(model_name, task_id, texts, labels, label_map)
            results.append(ollama_result)
            print(f"    {ollama_result['accuracy']:.1%} accuracy, {ollama_result['latency_p50_ms']:.0f}ms p50")

        print_table(task_id, results)
        all_results[task_id] = {
            "n_classes": TASKS[task_id]["n_classes"],
            "category": TASKS[task_id]["category"],
            "n_samples": len(texts),
            "results": results,
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "benchmark": "jeffybench",
        "date": time.strftime("%Y-%m-%d"),
        "tasks": all_results,
    }
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
