"""Convert trained PyTorch models to Core ML and benchmark across compute units.

Benchmarks inference on CPU, GPU (MPS), and Neural Engine (ANE) to compare
latency for our micro-scale models on Apple Silicon.

Prerequisites:
    pip install coremltools

Usage:
    python convert_coreml.py
    python convert_coreml.py --checkpoint checkpoints/best_lm_base.pt
    python convert_coreml.py --checkpoint checkpoints/best_hybrid_tied_iter2.pt --model-type classifier --task sst2
    python convert_coreml.py --seq-len 64 --n-iter 500
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent


# ── Minimal model definitions (avoids importing training scripts) ────────────

class CausalTransformerTrace(nn.Module):
    """CausalTransformer with a trace-friendly forward (fixed seq_len)."""

    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff,
                 max_seq_len, n_iterations=1, tie_weights=False):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.n_iterations = n_iterations

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_emb(input_ids) + self.pos_emb(positions)

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device),
            diagonal=1,
        )

        for _ in range(self.n_iterations):
            x = self.transformer(x, mask=causal_mask)
        x = self.norm(x)
        return self.lm_head(x)


class ClassifierTrace(nn.Module):
    """HybridTransformer forward_classify path, trace-friendly."""

    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff,
                 max_seq_len, n_classes, n_iterations=1, tie_weights=False,
                 iter_aware=False):
        super().__init__()
        self.d_model = d_model
        self.n_iterations = n_iterations
        self.iter_aware = iter_aware

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        if iter_aware and n_iterations > 1:
            self.iter_emb = nn.Embedding(n_iterations, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, input_ids, attention_mask):
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_emb(input_ids) + self.pos_emb(positions)

        src_key_padding_mask = attention_mask == 0
        for i in range(self.n_iterations):
            if self.iter_aware and self.n_iterations > 1:
                x = x + self.iter_emb(torch.tensor(i, device=x.device))
            x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)

        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        return self.head(pooled)


# ── Loading ──────────────────────────────────────────────────────────────────

def load_lm(path, seq_len):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = CausalTransformerTrace(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        max_seq_len=cfg["max_seq_len"],
        n_iterations=cfg.get("n_iterations", 1),
        tie_weights=cfg.get("tie_weights", False),
    )
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded LM: {n_params/1e6:.1f}M params")
    print(f"  d_model={cfg['d_model']}, n_layers={cfg['n_layers']}, "
          f"n_heads={cfg['n_heads']}, vocab={cfg['vocab_size']}")
    return model, cfg


def load_classifier(path, task, seq_len):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    task_classes = cfg.get("task_classes", {})
    if task not in task_classes:
        avail = list(task_classes.keys())
        raise ValueError(f"Task '{task}' not in checkpoint. Available: {avail}")
    n_classes = task_classes[task]
    model = ClassifierTrace(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        max_seq_len=cfg["max_seq_len"],
        n_classes=n_classes,
        n_iterations=cfg.get("n_iterations", 1),
        tie_weights=cfg.get("tie_weights", False),
        iter_aware=cfg.get("iter_aware", False),
    )
    state = ckpt["model_state"]
    # Remap task head: checkpoint uses heads.{task}.* but trace model uses head.*
    remapped = {}
    for k, v in state.items():
        if k == f"heads.{task}.weight":
            remapped["head.weight"] = v
        elif k == f"heads.{task}.bias":
            remapped["head.bias"] = v
        elif k.startswith("heads.") or k.startswith("mlm_head."):
            continue  # skip other task heads and MLM head
        else:
            remapped[k] = v
    model.load_state_dict(remapped, strict=False)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded classifier: {n_params/1e6:.1f}M params ({task}, {n_classes} classes)")
    print(f"  d_model={cfg['d_model']}, n_layers={cfg['n_layers']}, "
          f"n_heads={cfg['n_heads']}, vocab={cfg['vocab_size']}")
    return model, cfg


# ── Conversion ───────────────────────────────────────────────────────────────

def convert_to_coreml(model, example_inputs, model_type, out_path):
    import coremltools as ct

    print("\nTracing model with torch.jit.trace...", flush=True)
    traced = torch.jit.trace(model, example_inputs, check_trace=False)

    print("Converting to Core ML...", flush=True)
    if model_type == "lm":
        input_shape = ct.Shape(shape=(1, ct.RangeDim(lower_bound=1, upper_bound=256, default=128)))
        ml_input = ct.TensorType(name="input_ids", shape=input_shape, dtype=np.int32)
    else:
        seq_len = example_inputs[0].shape[1]
        ml_input = [
            ct.TensorType(name="input_ids", shape=(1, seq_len), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, seq_len), dtype=np.int32),
        ]

    mlmodel = ct.convert(
        traced,
        inputs=[ml_input] if model_type == "lm" else ml_input,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
    )

    mlmodel.save(out_path)
    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"Saved Core ML model: {out_path} ({size_mb:.1f} MB)")
    return mlmodel


# ── Benchmarking ─────────────────────────────────────────────────────────────

def benchmark_pytorch_cpu(model, example_inputs, n_iter):
    model_cpu = model.cpu()
    inputs_cpu = tuple(t.cpu() for t in example_inputs)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model_cpu(*inputs_cpu)

    t0 = time.perf_counter()
    for _ in range(n_iter):
        with torch.no_grad():
            model_cpu(*inputs_cpu)
    elapsed = time.perf_counter() - t0
    return elapsed / n_iter * 1000


def benchmark_pytorch_mps(model, example_inputs, n_iter):
    if not torch.backends.mps.is_available():
        return None

    device = torch.device("mps")
    model_mps = model.to(device)
    inputs_mps = tuple(t.to(device) for t in example_inputs)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model_mps(*inputs_mps)
    torch.mps.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        with torch.no_grad():
            model_mps(*inputs_mps)
    torch.mps.synchronize()
    elapsed = time.perf_counter() - t0

    model.cpu()
    return elapsed / n_iter * 1000


def benchmark_coreml(mlmodel_path, example_inputs_np, compute_unit, n_iter):
    import coremltools as ct

    cu_map = {
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "ALL": ct.ComputeUnit.ALL,
    }
    cu = cu_map[compute_unit]

    mlmodel = ct.models.MLModel(str(mlmodel_path), compute_units=cu)

    # Warmup
    for _ in range(10):
        mlmodel.predict(example_inputs_np)

    t0 = time.perf_counter()
    for _ in range(n_iter):
        mlmodel.predict(example_inputs_np)
    elapsed = time.perf_counter() - t0
    return elapsed / n_iter * 1000


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch model to Core ML & benchmark CPU/GPU/ANE")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/best_lm_base.pt")
    parser.add_argument("--model-type", type=str, default="lm",
                        choices=["lm", "classifier"])
    parser.add_argument("--task", type=str, default="sst2",
                        help="Classifier task head (ignored for LM)")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--n-iter", type=int, default=200,
                        help="Iterations for benchmarking")
    parser.add_argument("--skip-convert", action="store_true",
                        help="Skip conversion, benchmark existing .mlpackage")
    args = parser.parse_args()

    ckpt_path = HERE / args.checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    tag = ckpt_path.stem
    out_path = HERE / f"{tag}.mlpackage"

    # ── Load model ──
    print(f"Loading {args.model_type} from {ckpt_path}...", flush=True)
    if args.model_type == "lm":
        model, cfg = load_lm(str(ckpt_path), args.seq_len)
        example_input = (torch.randint(0, cfg["vocab_size"], (1, args.seq_len)),)
    else:
        model, cfg = load_classifier(str(ckpt_path), args.task, args.seq_len)
        example_input = (
            torch.randint(0, cfg["vocab_size"], (1, args.seq_len)),
            torch.ones(1, args.seq_len, dtype=torch.long),
        )

    n_params = sum(p.numel() for p in model.parameters())

    # ── Convert ──
    if not args.skip_convert:
        mlmodel = convert_to_coreml(model, example_input, args.model_type, str(out_path))
    else:
        print(f"Skipping conversion, using {out_path}")

    # ── Prepare inputs ──
    if args.model_type == "lm":
        example_np = {"input_ids": np.random.randint(0, cfg["vocab_size"],
                       (1, args.seq_len)).astype(np.int32)}
    else:
        example_np = {
            "input_ids": np.random.randint(0, cfg["vocab_size"],
                         (1, args.seq_len)).astype(np.int32),
            "attention_mask": np.ones((1, args.seq_len), dtype=np.int32),
        }

    # ── Benchmark ──
    print(f"\nBenchmarking ({args.n_iter} iterations, seq_len={args.seq_len})...",
          flush=True)
    results = {}

    # PyTorch CPU
    print("  PyTorch CPU...", end=" ", flush=True)
    ms = benchmark_pytorch_cpu(model, example_input, args.n_iter)
    results["PyTorch CPU"] = ms
    print(f"{ms:.2f} ms")

    # PyTorch MPS
    print("  PyTorch MPS...", end=" ", flush=True)
    ms = benchmark_pytorch_mps(model, example_input, args.n_iter)
    if ms is not None:
        results["PyTorch MPS (GPU)"] = ms
        print(f"{ms:.2f} ms")
    else:
        print("not available")

    # Core ML variants
    import coremltools as ct
    for cu_name in ["CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"]:
        label = f"Core ML {cu_name}"
        print(f"  {label}...", end=" ", flush=True)
        try:
            ms = benchmark_coreml(out_path, example_np, cu_name, args.n_iter)
            results[label] = ms
            print(f"{ms:.2f} ms")
        except Exception as e:
            print(f"failed ({e})")

    # ── Results ──
    print(f"\n{'='*65}")
    print(f"  Benchmark Results: {tag}")
    print(f"  {n_params/1e6:.1f}M params | seq_len={args.seq_len} | "
          f"{args.n_iter} iterations")
    print(f"{'='*65}")

    fastest = min(results.values())
    print(f"\n  {'Backend':<25} {'Latency':>10} {'vs Fastest':>12} {'Throughput':>14}")
    print(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*14}")
    for name, ms in sorted(results.items(), key=lambda x: x[1]):
        ratio = ms / fastest
        tps = 1000 / ms
        marker = " <<<" if ms == fastest else ""
        print(f"  {name:<25} {ms:>8.2f}ms {ratio:>10.1f}x {tps:>10.0f} inf/s{marker}")

    # Summary
    if "Core ML CPU_AND_NE" in results and "PyTorch CPU" in results:
        ane_ms = results["Core ML CPU_AND_NE"]
        cpu_ms = results["PyTorch CPU"]
        speedup = cpu_ms / ane_ms
        print(f"\n  ANE speedup vs PyTorch CPU: {speedup:.1f}x")

    if "Core ML CPU_AND_NE" in results:
        ane_ms = results["Core ML CPU_AND_NE"]
        print(f"\n  At {ane_ms:.2f}ms/inference:")
        print(f"    GAG best-of-10 loop: {ane_ms*10:.1f}ms (generate + 10x classify)")
        print(f"    100 verify cycles:   {ane_ms*100:.1f}ms")
        print(f"    1000 verify cycles:  {ane_ms*1000:.1f}ms")


if __name__ == "__main__":
    main()
