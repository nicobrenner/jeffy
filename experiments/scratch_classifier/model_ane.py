"""ANE-optimized transformer models for Apple Neural Engine inference.

Rewrites the standard PyTorch transformer to use the BC1S (batch, channels, 1, seq_len)
tensor format and nn.Conv2d projections, following Apple's ane-transformers patterns.
Loads weights from standard training checkpoints with automatic conversion.

Usage:
    python model_ane.py
    python model_ane.py --checkpoint checkpoints/best_lm_base.pt --seq-len 128
    python model_ane.py --checkpoint checkpoints/best_lm_8m.pt --seq-len 128
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
EPS = 1e-5


class LayerNormANEStd(nn.Module):
    """LayerNorm for BC1S tensors with standard PyTorch semantics (out * weight + bias).

    Apple's LayerNormANE computes (out + bias) * weight, which differs from
    PyTorch's out * weight + bias. This version matches PyTorch exactly so we
    can load standard checkpoint weights without conversion.
    """

    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        # x: (B, C, 1, S)
        mean = x.mean(dim=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        x_norm = (x - mean) * (var + self.eps).rsqrt()
        return x_norm * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class MultiHeadSelfAttentionANE(nn.Module):
    """Self-attention optimized for ANE using BC1S format and Conv2d projections."""

    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.dim_per_head = embed_dim // n_heads
        self.scale = float(self.dim_per_head) ** -0.5

        self.q_proj = nn.Conv2d(embed_dim, embed_dim, 1)
        self.k_proj = nn.Conv2d(embed_dim, embed_dim, 1)
        self.v_proj = nn.Conv2d(embed_dim, embed_dim, 1)
        self.out_proj = nn.Conv2d(embed_dim, embed_dim, 1)

    def forward(self, x, qk_mask=None, k_mask=None):
        """
        Args:
            x: (batch, embed_dim, 1, seq_len)
            qk_mask: (B, S_key, 1, S_query) — causal mask, -1e4 for masked
            k_mask: (B, S_key, 1, 1) — padding mask, -1e4 for padded keys
        Returns:
            (batch, embed_dim, 1, seq_len)
        """
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        mh_q = q.split(self.dim_per_head, dim=1)
        mh_k = k.transpose(1, 3).split(self.dim_per_head, dim=3)
        mh_v = v.split(self.dim_per_head, dim=1)

        attn_weights = [
            torch.einsum('bchq,bkhc->bkhq', qi, ki) * self.scale
            for qi, ki in zip(mh_q, mh_k)
        ]

        if qk_mask is not None:
            for i in range(self.n_heads):
                attn_weights[i] = attn_weights[i] + qk_mask
        if k_mask is not None:
            for i in range(self.n_heads):
                attn_weights[i] = attn_weights[i] + k_mask

        attn_weights = [aw.softmax(dim=1) for aw in attn_weights]

        attn = [
            torch.einsum('bkhq,bchk->bchq', wi, vi)
            for wi, vi in zip(attn_weights, mh_v)
        ]
        attn = torch.cat(attn, dim=1)
        return self.out_proj(attn)


class FFN_ANE(nn.Module):
    """FFN optimized for ANE using Conv2d projections."""

    def __init__(self, embed_dim, ffn_dim):
        super().__init__()
        self.linear1 = nn.Conv2d(embed_dim, ffn_dim, 1)
        self.linear2 = nn.Conv2d(ffn_dim, embed_dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        return self.linear2(self.act(self.linear1(x)))


class TransformerBlockANE(nn.Module):
    """Post-norm transformer block for ANE (matches nn.TransformerEncoderLayer default)."""

    def __init__(self, embed_dim, n_heads, ffn_dim):
        super().__init__()
        self.self_attn = MultiHeadSelfAttentionANE(embed_dim, n_heads)
        self.ffn = FFN_ANE(embed_dim, ffn_dim)
        self.norm1 = LayerNormANEStd(embed_dim, eps=EPS)
        self.norm2 = LayerNormANEStd(embed_dim, eps=EPS)

    def forward(self, x, qk_mask=None, k_mask=None):
        x = self.norm1(self.self_attn(x, qk_mask=qk_mask, k_mask=k_mask) + x)
        x = self.norm2(self.ffn(x) + x)
        return x


class CausalTransformerANE(nn.Module):
    """Causal transformer optimized for ANE inference.

    All computation uses BC1S format: (batch, channels, 1, seq_len).
    Loads weights from standard PyTorch checkpoints with automatic conversion.
    """

    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff,
                 max_seq_len, n_iterations=1, tie_weights=False):
        super().__init__()
        self.d_model = d_model
        self.n_iterations = n_iterations
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlockANE(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.norm = LayerNormANEStd(d_model, eps=EPS)
        self._tie_weights = tie_weights
        if not tie_weights:
            self.lm_head = nn.Conv2d(d_model, vocab_size, 1, bias=False)

    def forward(self, input_ids):
        """
        Args:
            input_ids: (batch, seq_len) int32
        Returns:
            logits: (batch, vocab_size, 1, seq_len) in BC1S format
        """
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_emb(input_ids) + self.pos_emb(positions)
        # BSC -> BC1S
        x = x.transpose(1, 2).unsqueeze(2)

        # Causal mask: (1, key, 1, query). Mask future keys (key_pos > query_pos).
        # Build as (S, S) then reshape: mask[k, q] = -1e4 when k > q
        mask_2d = torch.tril(
            torch.full((seq_len, seq_len), -1e4, device=input_ids.device),
            diagonal=-1,
        )
        causal_mask = mask_2d.unsqueeze(0).unsqueeze(2)  # (1, S_key, 1, S_query)

        for _ in range(self.n_iterations):
            for block in self.blocks:
                x = block(x, qk_mask=causal_mask)

        x = self.norm(x)
        if self._tie_weights:
            # x: (B, D, 1, S), token_emb.weight: (V, D)
            # Output: (B, V, 1, S)
            x_sq = x.squeeze(2)  # (B, D, S)
            logits = torch.einsum('vd,bds->bvs', self.token_emb.weight, x_sq)
            return logits.unsqueeze(2)
        return self.lm_head(x)


class ClassifierANE(nn.Module):
    """Classifier optimized for ANE inference.

    Bidirectional attention (no causal mask), mean pooling, per-task head.
    Supports depth recurrence (n_iterations > 1).
    """

    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff,
                 max_seq_len, n_classes, n_iterations=1):
        super().__init__()
        self.d_model = d_model
        self.n_iterations = n_iterations

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlockANE(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.norm = LayerNormANEStd(d_model, eps=EPS)
        self.head = nn.Conv2d(d_model, n_classes, 1)

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: (batch, seq_len) int32
            attention_mask: (batch, seq_len) int32, 1=valid 0=pad
        Returns:
            logits: (batch, n_classes)
        """
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = x.transpose(1, 2).unsqueeze(2)  # BC1S

        # Key padding mask: (B, S_key, 1, 1), -1e4 for padded positions
        pad_mask = (1 - attention_mask.float()) * -1e4
        pad_mask = pad_mask.unsqueeze(2).unsqueeze(2)  # (B, S, 1, 1)

        for _ in range(self.n_iterations):
            for block in self.blocks:
                x = block(x, k_mask=pad_mask)

        x = self.norm(x)  # (B, D, 1, S)

        # Mean pooling over non-padded positions
        mask_bc1s = attention_mask.float().unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
        pooled = (x * mask_bc1s).sum(dim=3, keepdim=True) / mask_bc1s.sum(dim=3, keepdim=True).clamp(min=1)
        # pooled: (B, D, 1, 1)
        logits = self.head(pooled)  # (B, n_classes, 1, 1)
        return logits.squeeze(-1).squeeze(-1)  # (B, n_classes)


def load_classifier_from_checkpoint(path, task="sst2"):
    """Load an ANE-optimized classifier from a standard training checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    task_classes = cfg.get("task_classes", {})
    if task not in task_classes:
        raise ValueError(f"Task '{task}' not in checkpoint. Available: {list(task_classes.keys())}")
    n_classes = task_classes[task]

    model = ClassifierANE(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        max_seq_len=cfg["max_seq_len"],
        n_classes=n_classes,
        n_iterations=cfg.get("n_iterations", 1),
    )

    src = ckpt["model_state"]
    dst = {}
    d = cfg["d_model"]

    for layer_idx in range(cfg["n_layers"]):
        prefix_src = f"transformer.layers.{layer_idx}"
        prefix_dst = f"blocks.{layer_idx}"

        in_proj_w = src[f"{prefix_src}.self_attn.in_proj_weight"]
        in_proj_b = src[f"{prefix_src}.self_attn.in_proj_bias"]
        q_w, k_w, v_w = in_proj_w.split(d, dim=0)
        q_b, k_b, v_b = in_proj_b.split(d, dim=0)

        dst[f"{prefix_dst}.self_attn.q_proj.weight"] = q_w[:, :, None, None]
        dst[f"{prefix_dst}.self_attn.q_proj.bias"] = q_b
        dst[f"{prefix_dst}.self_attn.k_proj.weight"] = k_w[:, :, None, None]
        dst[f"{prefix_dst}.self_attn.k_proj.bias"] = k_b
        dst[f"{prefix_dst}.self_attn.v_proj.weight"] = v_w[:, :, None, None]
        dst[f"{prefix_dst}.self_attn.v_proj.bias"] = v_b

        dst[f"{prefix_dst}.self_attn.out_proj.weight"] = src[f"{prefix_src}.self_attn.out_proj.weight"][:, :, None, None]
        dst[f"{prefix_dst}.self_attn.out_proj.bias"] = src[f"{prefix_src}.self_attn.out_proj.bias"]

        dst[f"{prefix_dst}.ffn.linear1.weight"] = src[f"{prefix_src}.linear1.weight"][:, :, None, None]
        dst[f"{prefix_dst}.ffn.linear1.bias"] = src[f"{prefix_src}.linear1.bias"]
        dst[f"{prefix_dst}.ffn.linear2.weight"] = src[f"{prefix_src}.linear2.weight"][:, :, None, None]
        dst[f"{prefix_dst}.ffn.linear2.bias"] = src[f"{prefix_src}.linear2.bias"]

        dst[f"{prefix_dst}.norm1.weight"] = src[f"{prefix_src}.norm1.weight"]
        dst[f"{prefix_dst}.norm1.bias"] = src[f"{prefix_src}.norm1.bias"]
        dst[f"{prefix_dst}.norm2.weight"] = src[f"{prefix_src}.norm2.weight"]
        dst[f"{prefix_dst}.norm2.bias"] = src[f"{prefix_src}.norm2.bias"]

    dst["token_emb.weight"] = src["token_emb.weight"]
    dst["pos_emb.weight"] = src["pos_emb.weight"]
    dst["norm.weight"] = src["norm.weight"]
    dst["norm.bias"] = src["norm.bias"]

    # Task head
    dst["head.weight"] = src[f"heads.{task}.weight"][:, :, None, None]
    dst["head.bias"] = src[f"heads.{task}.bias"]

    model.load_state_dict(dst)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded ANE classifier: {n_params/1e6:.1f}M params ({task}, {n_classes} classes)")
    print(f"  d_model={cfg['d_model']}, n_layers={cfg['n_layers']}, "
          f"n_heads={cfg['n_heads']}, n_iterations={cfg.get('n_iterations', 1)}")
    return model, cfg


def load_from_checkpoint(path):
    """Load an ANE-optimized model from a standard training checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    model = CausalTransformerANE(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        max_seq_len=cfg["max_seq_len"],
        n_iterations=cfg.get("n_iterations", 1),
        tie_weights=cfg.get("tie_weights", False),
    )

    src = ckpt["model_state"]
    dst = {}

    for layer_idx in range(cfg["n_layers"]):
        prefix_src = f"transformer.layers.{layer_idx}"
        prefix_dst = f"blocks.{layer_idx}"

        # Split in_proj_weight [3*d, d] -> q_proj, k_proj, v_proj [d, d, 1, 1]
        in_proj_w = src[f"{prefix_src}.self_attn.in_proj_weight"]
        in_proj_b = src[f"{prefix_src}.self_attn.in_proj_bias"]
        d = cfg["d_model"]
        q_w, k_w, v_w = in_proj_w.split(d, dim=0)
        q_b, k_b, v_b = in_proj_b.split(d, dim=0)

        dst[f"{prefix_dst}.self_attn.q_proj.weight"] = q_w[:, :, None, None]
        dst[f"{prefix_dst}.self_attn.q_proj.bias"] = q_b
        dst[f"{prefix_dst}.self_attn.k_proj.weight"] = k_w[:, :, None, None]
        dst[f"{prefix_dst}.self_attn.k_proj.bias"] = k_b
        dst[f"{prefix_dst}.self_attn.v_proj.weight"] = v_w[:, :, None, None]
        dst[f"{prefix_dst}.self_attn.v_proj.bias"] = v_b

        # out_proj: Linear [d, d] -> Conv2d [d, d, 1, 1]
        dst[f"{prefix_dst}.self_attn.out_proj.weight"] = (
            src[f"{prefix_src}.self_attn.out_proj.weight"][:, :, None, None]
        )
        dst[f"{prefix_dst}.self_attn.out_proj.bias"] = (
            src[f"{prefix_src}.self_attn.out_proj.bias"]
        )

        # FFN: linear1 [ff, d] -> Conv2d [ff, d, 1, 1], etc.
        dst[f"{prefix_dst}.ffn.linear1.weight"] = (
            src[f"{prefix_src}.linear1.weight"][:, :, None, None]
        )
        dst[f"{prefix_dst}.ffn.linear1.bias"] = src[f"{prefix_src}.linear1.bias"]
        dst[f"{prefix_dst}.ffn.linear2.weight"] = (
            src[f"{prefix_src}.linear2.weight"][:, :, None, None]
        )
        dst[f"{prefix_dst}.ffn.linear2.bias"] = src[f"{prefix_src}.linear2.bias"]

        # LayerNorm -> LayerNormANE (same param shapes)
        dst[f"{prefix_dst}.norm1.weight"] = src[f"{prefix_src}.norm1.weight"]
        dst[f"{prefix_dst}.norm1.bias"] = src[f"{prefix_src}.norm1.bias"]
        dst[f"{prefix_dst}.norm2.weight"] = src[f"{prefix_src}.norm2.weight"]
        dst[f"{prefix_dst}.norm2.bias"] = src[f"{prefix_src}.norm2.bias"]

    # Embeddings (same format)
    dst["token_emb.weight"] = src["token_emb.weight"]
    dst["pos_emb.weight"] = src["pos_emb.weight"]

    # Final layer norm
    dst["norm.weight"] = src["norm.weight"]
    dst["norm.bias"] = src["norm.bias"]

    # LM head: only needed when weights aren't tied
    if not cfg.get("tie_weights", False) and "lm_head.weight" in src:
        dst["lm_head.weight"] = src["lm_head.weight"][:, :, None, None]

    model.load_state_dict(dst)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded ANE model: {n_params/1e6:.1f}M params")
    print(f"  d_model={cfg['d_model']}, n_layers={cfg['n_layers']}, "
          f"n_heads={cfg['n_heads']}, vocab={cfg['vocab_size']}")
    return model, cfg


def verify_outputs(checkpoint_path, seq_len=128):
    """Verify ANE model produces same outputs as standard model."""
    from convert_coreml import CausalTransformerTrace, load_lm

    std_model, cfg = load_lm(str(checkpoint_path), seq_len)
    std_model.eval()
    ane_model, _ = load_from_checkpoint(str(checkpoint_path))

    input_ids = torch.randint(0, cfg["vocab_size"], (1, seq_len))

    with torch.no_grad():
        std_out = std_model(input_ids)  # (1, seq_len, vocab)
        ane_out = ane_model(input_ids)  # (1, vocab, 1, seq_len) in BC1S

    # Convert ANE output to standard format for comparison
    ane_out_std = ane_out.squeeze(2).transpose(1, 2)  # (1, seq_len, vocab)

    max_diff = (std_out - ane_out_std).abs().max().item()
    mean_diff = (std_out - ane_out_std).abs().mean().item()
    print(f"\n  Output verification:")
    print(f"    Max absolute diff: {max_diff:.6f}")
    print(f"    Mean absolute diff: {mean_diff:.6f}")
    print(f"    Match: {'YES' if max_diff < 0.01 else 'NO (check weight loading)'}")
    return max_diff < 0.01


def convert_and_benchmark(checkpoint_path, seq_len=128, n_iter=200):
    """Convert ANE model to Core ML and benchmark."""
    import coremltools as ct

    print(f"Loading ANE model from {checkpoint_path}...")
    model, cfg = load_from_checkpoint(str(checkpoint_path))

    tag = Path(checkpoint_path).stem
    out_path = HERE / f"{tag}_ane.mlpackage"

    example_input = torch.randint(0, cfg["vocab_size"], (1, seq_len))

    print("\nTracing model...", flush=True)
    traced = torch.jit.trace(model, (example_input,), check_trace=False)

    print("Converting to Core ML...", flush=True)
    input_shape = ct.Shape(
        shape=(1, ct.RangeDim(lower_bound=1, upper_bound=256, default=seq_len))
    )
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input_ids", shape=input_shape, dtype=np.int32)],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
    )
    mlmodel.save(str(out_path))
    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved: {out_path} ({size_mb:.1f} MB)")

    # Benchmark
    example_np = {"input_ids": np.random.randint(
        0, cfg["vocab_size"], (1, seq_len)).astype(np.int32)}

    print(f"\nBenchmarking ({n_iter} iterations, seq_len={seq_len})...", flush=True)
    results = {}

    # PyTorch CPU
    print("  PyTorch CPU...", end=" ", flush=True)
    model_cpu = model.cpu()
    inp_cpu = (example_input.cpu(),)
    for _ in range(10):
        with torch.no_grad():
            model_cpu(*inp_cpu)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        with torch.no_grad():
            model_cpu(*inp_cpu)
    ms = (time.perf_counter() - t0) / n_iter * 1000
    results["PyTorch CPU"] = ms
    print(f"{ms:.2f} ms")

    # PyTorch MPS
    if torch.backends.mps.is_available():
        print("  PyTorch MPS...", end=" ", flush=True)
        device = torch.device("mps")
        model_mps = model.to(device)
        inp_mps = (example_input.to(device),)
        for _ in range(10):
            with torch.no_grad():
                model_mps(*inp_mps)
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            with torch.no_grad():
                model_mps(*inp_mps)
        torch.mps.synchronize()
        ms = (time.perf_counter() - t0) / n_iter * 1000
        results["PyTorch MPS (GPU)"] = ms
        print(f"{ms:.2f} ms")
        model.cpu()

    # Core ML backends
    cu_map = {
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "ALL": ct.ComputeUnit.ALL,
    }
    for cu_name, cu in cu_map.items():
        label = f"Core ML {cu_name}"
        print(f"  {label}...", end=" ", flush=True)
        try:
            mlm = ct.models.MLModel(str(out_path), compute_units=cu)
            for _ in range(10):
                mlm.predict(example_np)
            t0 = time.perf_counter()
            for _ in range(n_iter):
                mlm.predict(example_np)
            ms = (time.perf_counter() - t0) / n_iter * 1000
            results[label] = ms
            print(f"{ms:.2f} ms")
        except Exception as e:
            print(f"failed ({e})")

    # Results table
    n_params = sum(p.numel() for p in model.parameters())
    fastest = min(results.values())
    print(f"\n{'='*70}")
    print(f"  ANE-Optimized Benchmark: {tag}")
    print(f"  {n_params/1e6:.1f}M params | seq_len={seq_len} | {n_iter} iterations")
    print(f"{'='*70}")
    print(f"\n  {'Backend':<25} {'Latency':>10} {'vs Fastest':>12} {'Throughput':>14}")
    print(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*14}")
    for name, ms in sorted(results.items(), key=lambda x: x[1]):
        ratio = ms / fastest
        tps = 1000 / ms
        marker = " <<<" if ms == fastest else ""
        print(f"  {name:<25} {ms:>8.2f}ms {ratio:>10.1f}x {tps:>10.0f} inf/s{marker}")

    if "Core ML CPU_AND_NE" in results:
        ane_ms = results["Core ML CPU_AND_NE"]
        print(f"\n  At {ane_ms:.2f}ms/inference:")
        print(f"    GAG best-of-10:    {ane_ms*10:.1f}ms")
        print(f"    100 verify cycles: {ane_ms*100:.1f}ms")

    # Compare to previous (non-ANE) results
    print(f"\n  Previous (non-ANE optimized) results for reference:")
    print(f"    PyTorch MPS:     6.11ms")
    print(f"    Core ML ANE:    24.84ms")

    return results


def verify_classifier_outputs(checkpoint_path, task="sst2", seq_len=128):
    """Verify ANE classifier produces same outputs as standard model."""
    from convert_coreml import ClassifierTrace, load_classifier

    std_model, cfg = load_classifier(str(checkpoint_path), task, seq_len)
    std_model.eval()
    ane_model, _ = load_classifier_from_checkpoint(str(checkpoint_path), task)

    input_ids = torch.randint(0, cfg["vocab_size"], (1, seq_len))
    attention_mask = torch.ones(1, seq_len, dtype=torch.long)

    with torch.no_grad():
        std_out = std_model(input_ids, attention_mask)  # (1, n_classes)
        ane_out = ane_model(input_ids, attention_mask)  # (1, n_classes)

    max_diff = (std_out - ane_out).abs().max().item()
    mean_diff = (std_out - ane_out).abs().mean().item()
    print(f"\n  Classifier output verification ({task}):")
    print(f"    Max absolute diff: {max_diff:.6f}")
    print(f"    Mean absolute diff: {mean_diff:.6f}")
    print(f"    Match: {'YES' if max_diff < 0.01 else 'NO (check weight loading)'}")
    return max_diff < 0.01


def convert_and_benchmark_classifier(checkpoint_path, task="sst2", seq_len=128, n_iter=200):
    """Convert ANE classifier to Core ML and benchmark."""
    import coremltools as ct

    print(f"Loading ANE classifier from {checkpoint_path} (task={task})...")
    model, cfg = load_classifier_from_checkpoint(str(checkpoint_path), task)

    tag = f"{Path(checkpoint_path).stem}_{task}_ane"
    out_path = HERE / f"{tag}.mlpackage"

    example_input = (
        torch.randint(0, cfg["vocab_size"], (1, seq_len)),
        torch.ones(1, seq_len, dtype=torch.long),
    )

    print("\nTracing model...", flush=True)
    traced = torch.jit.trace(model, example_input, check_trace=False)

    print("Converting to Core ML...", flush=True)
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, seq_len), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, seq_len), dtype=np.int32),
        ],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
    )
    mlmodel.save(str(out_path))
    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved: {out_path} ({size_mb:.1f} MB)")

    example_np = {
        "input_ids": np.random.randint(0, cfg["vocab_size"], (1, seq_len)).astype(np.int32),
        "attention_mask": np.ones((1, seq_len), dtype=np.int32),
    }

    print(f"\nBenchmarking ({n_iter} iterations, seq_len={seq_len})...", flush=True)
    results = {}

    print("  PyTorch CPU...", end=" ", flush=True)
    model_cpu = model.cpu()
    inp_cpu = tuple(t.cpu() for t in example_input)
    for _ in range(10):
        with torch.no_grad():
            model_cpu(*inp_cpu)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        with torch.no_grad():
            model_cpu(*inp_cpu)
    ms = (time.perf_counter() - t0) / n_iter * 1000
    results["PyTorch CPU"] = ms
    print(f"{ms:.2f} ms")

    cu_map = {
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "ALL": ct.ComputeUnit.ALL,
    }
    for cu_name, cu in cu_map.items():
        label = f"Core ML {cu_name}"
        print(f"  {label}...", end=" ", flush=True)
        try:
            mlm = ct.models.MLModel(str(out_path), compute_units=cu)
            for _ in range(10):
                mlm.predict(example_np)
            t0 = time.perf_counter()
            for _ in range(n_iter):
                mlm.predict(example_np)
            ms = (time.perf_counter() - t0) / n_iter * 1000
            results[label] = ms
            print(f"{ms:.2f} ms")
        except Exception as e:
            print(f"failed ({e})")

    n_params = sum(p.numel() for p in model.parameters())
    fastest = min(results.values())
    print(f"\n{'='*70}")
    print(f"  ANE Classifier Benchmark: {task}")
    print(f"  {n_params/1e6:.1f}M params | seq_len={seq_len} | {n_iter} iterations")
    print(f"{'='*70}")
    print(f"\n  {'Backend':<25} {'Latency':>10} {'vs Fastest':>12} {'Throughput':>14}")
    print(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*14}")
    for name, ms in sorted(results.items(), key=lambda x: x[1]):
        ratio = ms / fastest
        tps = 1000 / ms
        marker = " <<<" if ms == fastest else ""
        print(f"  {name:<25} {ms:>8.2f}ms {ratio:>10.1f}x {tps:>10.0f} inf/s{marker}")

    if "Core ML CPU_AND_NE" in results:
        ane_ms = results["Core ML CPU_AND_NE"]
        lm_ane_ms = 2.63
        total = lm_ane_ms + ane_ms * 10
        print(f"\n  Full GAG loop on ANE (projected):")
        print(f"    LM generate:     {lm_ane_ms:.2f}ms")
        print(f"    10x classify:    {ane_ms*10:.1f}ms")
        print(f"    Total best-of-10: {total:.1f}ms")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="ANE-optimized transformer: convert, verify, and benchmark")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/best_lm_base.pt")
    parser.add_argument("--model-type", type=str, default="lm",
                        choices=["lm", "classifier"])
    parser.add_argument("--task", type=str, default="sst2")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--n-iter", type=int, default=200)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    ckpt_path = HERE / args.checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if args.model_type == "classifier":
        print("Step 1: Verify classifier output equivalence...")
        ok = verify_classifier_outputs(ckpt_path, args.task, args.seq_len)
        if not ok:
            print("\nWARNING: Outputs don't match!")
        if args.verify_only:
            return
        print("\nStep 2: Convert to Core ML and benchmark...")
        convert_and_benchmark_classifier(ckpt_path, args.task, args.seq_len, args.n_iter)
    else:
        if args.verify_only:
            verify_outputs(ckpt_path, args.seq_len)
            return
        print("Step 1: Verify output equivalence...")
        ok = verify_outputs(ckpt_path, args.seq_len)
        if not ok:
            print("\nWARNING: Outputs don't match! Check weight conversion.")
            print("Continuing with benchmark anyway...\n")
        print("\nStep 2: Convert to Core ML and benchmark...")
        convert_and_benchmark(ckpt_path, args.seq_len, args.n_iter)


if __name__ == "__main__":
    main()
