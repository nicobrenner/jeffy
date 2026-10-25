"""Benchmark inference latency for our scratch models and BERT-Tiny.

Measures single-sample and batched inference time on CPU and MPS
for a fair comparison across model sizes.
"""

import time
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

BERT_TINY = "google/bert_uncased_L-2_H-128_A-2"
SAMPLE_TEXTS = [
    "I love this product, it works great!",
    "The customer service was terrible and I want a refund for my account.",
    "Scientists have discovered a new species of deep-sea fish in the Pacific Ocean that glows in the dark.",
    "Breaking: Stock markets tumble as inflation data comes in higher than expected across all sectors.",
]
N_WARMUP = 10
N_RUNS = 100


def load_scratch_model(ckpt_path, device="cpu"):
    from experiments.scratch_classifier.train_hybrid import HybridTransformer
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = HybridTransformer(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config["d_ff"],
        max_seq_len=128,
        task_classes=config.get("task_classes", {}),
        tie_weights=config.get("tie_weights", False),
        n_iterations=config.get("n_iterations", 1),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, config, ckpt["vocab_map"]


def bench_scratch(model, tokenizer, vocab_map, text, task_name, device, n_iterations=1):
    tokens = tokenizer.encode(text, add_special_tokens=False)[:126]
    compact = [vocab_map.get(t, 0) for t in tokens]
    input_ids = torch.tensor([compact], device=device)
    attention_mask = torch.ones_like(input_ids)

    for _ in range(N_WARMUP):
        with torch.no_grad():
            model.forward_classify(input_ids, attention_mask, task_name)
        if device == "mps":
            torch.mps.synchronize()

    times = []
    for _ in range(N_RUNS):
        if device == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.forward_classify(input_ids, attention_mask, task_name)
        if device == "mps":
            torch.mps.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return np.median(times), np.mean(times), np.std(times)


def bench_bert_tiny(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=128, padding="max_length").to(device)

    for _ in range(N_WARMUP):
        with torch.no_grad():
            model(**inputs)
        if device == "mps":
            torch.mps.synchronize()

    times = []
    for _ in range(N_RUNS):
        if device == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(**inputs)
        if device == "mps":
            torch.mps.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return np.median(times), np.mean(times), np.std(times)


def get_model_size_mb(model):
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return param_bytes / (1024 * 1024)


def main():
    checkpoints = {
        "3M (d=64, tied)": "experiments/scratch_classifier/checkpoints/best_hybrid_scale_3M.pt",
        "4.4M (d=96, tied)": "experiments/scratch_classifier/checkpoints/best_hybrid_scale_4.4M.pt",
        "6M (d=128, tied)": "experiments/scratch_classifier/checkpoints/best_hybrid_tied.pt",
        "6M 1a+2x (d=128)": "experiments/scratch_classifier/checkpoints/best_hybrid_1a_iter2.pt",
        "11.7M (d=128, untied)": "experiments/scratch_classifier/checkpoints/best_hybrid.pt",
    }

    qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    if qwen_tok.pad_token is None:
        qwen_tok.pad_token = qwen_tok.eos_token

    bert_tok = AutoTokenizer.from_pretrained(BERT_TINY)

    text = SAMPLE_TEXTS[1]  # medium length

    for device_name in ["cpu", "mps"]:
        if device_name == "mps" and not torch.backends.mps.is_available():
            continue

        print(f"\n{'='*70}")
        print(f"Device: {device_name.upper()}")
        print(f"{'='*70}")
        print(f"{'Model':<28} {'Size (MB)':>10} {'Median (ms)':>12} {'Mean (ms)':>10} {'Std (ms)':>10}")
        print("-" * 72)

        for name, ckpt_path in checkpoints.items():
            if not Path(ckpt_path).exists():
                print(f"{name:<28} {'N/A':>10} -- checkpoint not found")
                continue
            model, config, vocab_map = load_scratch_model(ckpt_path, device_name)
            size_mb = get_model_size_mb(model)
            task = list(config["task_classes"].keys())[0]
            median, mean, std = bench_scratch(model, qwen_tok, vocab_map, text, task, device_name)
            print(f"{name:<28} {size_mb:>10.1f} {median:>12.2f} {mean:>10.2f} {std:>10.2f}")
            del model

        # BERT-Tiny
        bert_model = AutoModel.from_pretrained(BERT_TINY).to(device_name)
        bert_model.eval()
        size_mb = get_model_size_mb(bert_model)
        median, mean, std = bench_bert_tiny(bert_model, bert_tok, text, device_name)
        print(f"{'BERT-Tiny (pretrained)':<28} {size_mb:>10.1f} {median:>12.2f} {mean:>10.2f} {std:>10.2f}")
        del bert_model

    print(f"\nText: \"{text[:60]}...\"")
    print(f"Runs: {N_RUNS} (after {N_WARMUP} warmup)")


if __name__ == "__main__":
    main()
