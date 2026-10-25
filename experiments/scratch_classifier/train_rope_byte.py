"""RoPE + byte-level vocabulary experiment.

Rotary Position Embeddings eliminate the position embedding table entirely,
pushing reasoning ratio even higher. Combined with byte-level vocab (256),
this maximizes the fraction of parameters in the transformer.

Config: d_model=32, n_layers=4, n_heads=4, vocab=256, byte-level
Expected: ~59K params, ~86% reasoning ratio (vs 39% without RoPE)

Usage:
    python train_rope_byte.py --tok-param 1800
    python train_rope_byte.py --tok-param 3600
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset


def precompute_freqs(dim, max_seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x, freqs):
    """Apply rotary embeddings to x. x shape: (B, n_heads, S, head_dim)."""
    B, H, S, D = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(B, H, S, D // 2, 2))
    freqs = freqs[:S].unsqueeze(0).unsqueeze(0)
    x_rotated = torch.view_as_real(x_complex * freqs).reshape(B, H, S, D)
    return x_rotated.type_as(x)


class RoPEAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x, freqs, causal_mask=None):
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        if causal_mask is not None:
            attn = attn + causal_mask[:S, :S]
        attn = F.softmax(attn, dim=-1)
        if self.training and self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout)

        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        return self.out_proj(out)


class RoPETransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = RoPEAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, freqs, causal_mask=None):
        x = x + self.attn(self.ln1(x), freqs, causal_mask)
        x = x + self.ff(self.ln2(x))
        return x


class RoPEByteLM(nn.Module):
    """Byte-level LM with RoPE — zero position embedding parameters."""

    def __init__(self, vocab_size=256, d_model=32, n_layers=4, n_heads=4,
                 max_seq_len=1024, dropout=0.0, tie_weights=True):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        self.token_emb = nn.Embedding(vocab_size, d_model)
        # No position embeddings — RoPE handles positions

        d_ff = 4 * d_model
        self.layers = nn.ModuleList([
            RoPETransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.token_emb.weight

        head_dim = d_model // n_heads
        self.register_buffer('freqs', precompute_freqs(head_dim, max_seq_len))
        self.register_buffer('causal_mask', torch.triu(
            torch.full((max_seq_len, max_seq_len), float("-inf")), diagonal=1))

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        for layer in self.layers:
            nn.init.normal_(layer.attn.qkv.weight, std=0.02)
            nn.init.normal_(layer.attn.out_proj.weight, std=0.02)

    def forward(self, input_ids, causal=True):
        B, S = input_ids.shape
        x = self.token_emb(input_ids)
        mask = self.causal_mask if causal else None
        for layer in self.layers:
            x = layer(x, self.freqs, mask)
        x = self.norm(x)
        return self.lm_head(x)


class ByteDataset(IterableDataset):
    def __init__(self, path, seq_len, max_tokens=None):
        self.path = path
        self.seq_len = seq_len
        self.max_tokens = max_tokens
        self.total_tokens = os.path.getsize(path)

    def __iter__(self):
        data = np.memmap(self.path, dtype=np.uint8, mode='r')
        n_chunks = len(data) // self.seq_len
        tokens_yielded = 0
        import random
        while True:
            indices = list(range(n_chunks))
            random.shuffle(indices)
            for idx in indices:
                start = idx * self.seq_len
                chunk = torch.from_numpy(
                    data[start:start + self.seq_len].astype(np.int64))
                tokens_yielded += self.seq_len
                if self.max_tokens and tokens_yielded > self.max_tokens:
                    return
                yield chunk


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=50):
    model.eval()
    total_loss = 0
    n = 0
    for batch in val_loader:
        if n >= max_batches:
            break
        ids = batch.to(device).unsqueeze(0) if batch.dim() == 1 else batch.to(device)
        logits = model(ids)
        loss = F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
            ids[:, 1:].contiguous().view(-1),
        )
        total_loss += loss.item()
        n += 1
    model.train()
    avg = total_loss / max(n, 1)
    return avg, math.exp(min(avg, 20))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d-model', type=int, default=32)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--tok-param', type=float, default=1800)
    parser.add_argument('--seq-len', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    parser.add_argument('--data-dir', type=str, default='pile_runs/custom_vocab/byte')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()

    vocab_size = 256
    d_model = args.d_model
    n_layers = args.n_layers
    n_heads = args.n_heads
    seq_len = args.seq_len
    device = args.device

    # Check byte data exists
    byte_dir = Path(args.data_dir)
    train_path = byte_dir / 'train.bytes'
    val_path = byte_dir / 'val.bytes'

    if not train_path.exists():
        print("ERROR: Byte data not found. Run train_custom_vocab.py --mode byte first to prepare data.")
        return

    # Build model
    model = RoPEByteLM(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, max_seq_len=seq_len, tie_weights=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    emb_params = model.token_emb.weight.numel()
    trans_params = n_params - emb_params
    reasoning_ratio = trans_params / n_params * 100

    print(f"{'='*70}")
    print(f"  RoPE + BYTE-LEVEL EXPERIMENT")
    print(f"{'='*70}")
    print(f"  Vocab: {vocab_size} (byte-level)")
    print(f"  Architecture: d={d_model}, L={n_layers}, H={n_heads}, RoPE positions")
    print(f"  Target: {args.tok_param} tok/param")
    print(f"  Device: {device}")
    print(f"\n  Model: {n_params:,} params")
    print(f"    Embeddings: {emb_params:,} ({emb_params/n_params*100:.1f}%)")
    print(f"    Transformer: {trans_params:,} ({reasoning_ratio:.1f}%)")
    print(f"    Position emb: 0 (RoPE)")
    print(f"    Reasoning ratio: {reasoning_ratio:.1f}%")

    total_tokens = int(args.tok_param * n_params)
    n_steps = total_tokens // (args.batch_size * seq_len)
    print(f"  Target tokens (causal): {total_tokens/1e6:.1f}M")
    print(f"  Steps: {n_steps:,}")

    # Compare with non-RoPE model
    pos_emb_params = vocab_size * d_model  # what pos_emb WOULD cost (but it's actually seq_len * d_model)
    pos_emb_actual = seq_len * d_model
    print(f"\n  vs non-RoPE (learned pos_emb): would add {pos_emb_actual:,} params")
    non_rope_total = n_params + pos_emb_actual
    non_rope_trans = trans_params
    non_rope_ratio = non_rope_trans / non_rope_total * 100
    print(f"    non-RoPE total: {non_rope_total:,}, reasoning: {non_rope_ratio:.1f}%")
    print(f"    RoPE saves: {pos_emb_actual:,} params, +{reasoning_ratio - non_rope_ratio:.1f}pp reasoning ratio")

    # Data loaders
    train_ds = ByteDataset(str(train_path), seq_len, max_tokens=total_tokens)
    val_ds = ByteDataset(str(val_path), seq_len, max_tokens=seq_len * 500)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    # Output directory
    run_dir = Path(f'pile_runs/rope_byte/run_d{d_model}_tp{int(args.tok_param)}')
    run_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint
    start_step = 0
    best_ppl = float('inf')
    best_state = None
    ckpt_path = run_dir / 'checkpoint.pt'

    if args.resume and ckpt_path.exists():
        print(f"\n  Resuming from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_step = ckpt['step']
        best_ppl = ckpt['best_ppl']
        if ckpt.get('best_model_state'):
            best_state = ckpt['best_model_state']
        print(f"  Resumed at step {start_step:,} (tok/param={start_step * args.batch_size * seq_len / n_params:.1f}), best_ppl={best_ppl:.2f}")

    # Training
    print(f"\n  Training...")
    model.train()
    step = 0
    running_loss = 0
    start_time = time.time()

    eval_every = max(50, n_steps // 40)
    log_every = max(100, n_steps // 20)

    for batch in train_loader:
        step += 1
        if step <= start_step:
            scheduler.step()
            continue

        ids = batch.to(device)
        logits = model(ids)
        loss = F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
            ids[:, 1:].contiguous().view(-1),
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        tp = step * args.batch_size * seq_len / n_params

        if step % eval_every == 0:
            val_loss, val_ppl = evaluate(model, val_loader, device)
            best_tag = ""
            if val_ppl < best_ppl:
                best_ppl = val_ppl
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_tag = " *BEST*"
                torch.save({
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'step': step,
                    'best_ppl': best_ppl,
                    'best_model_state': best_state,
                }, ckpt_path)
            print(f"    >>> EVAL step {step}: val_ppl={val_ppl:.2f} (tok/param={tp:.1f}){best_tag}")
            model.train()

        if step % log_every == 0:
            avg_loss = running_loss / log_every
            elapsed = time.time() - start_time
            tok_per_s = step * args.batch_size * seq_len / elapsed
            ppl = math.exp(min(avg_loss, 20))
            print(f"    step {step:>6,}/{n_steps:,}  loss={avg_loss:.4f}  ppl={ppl:>8.1f}  "
                  f"{tok_per_s:,.0f} tok/s  tp={tp:.1f}")
            running_loss = 0

        if step >= n_steps:
            break

    # Final eval
    val_loss, val_ppl = evaluate(model, val_loader, device)
    if val_ppl < best_ppl:
        best_ppl = val_ppl
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    print(f"\n  Final: val_ppl={val_ppl:.2f}, best={best_ppl:.2f}, tok/param={args.tok_param:.1f}")

    elapsed = time.time() - start_time
    print(f"  Training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save
    results = {
        'model': 'rope_byte',
        'd_model': d_model,
        'n_layers': n_layers,
        'n_heads': n_heads,
        'vocab_size': vocab_size,
        'n_params': n_params,
        'emb_params': emb_params,
        'trans_params': trans_params,
        'reasoning_ratio': reasoning_ratio,
        'tok_param_target': args.tok_param,
        'best_val_ppl': best_ppl,
        'final_val_ppl': val_ppl,
        'total_tokens': total_tokens,
        'training_time_s': elapsed,
    }

    with open(run_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    if best_state:
        torch.save({
            'model_state': best_state,
            'config': {'d_model': d_model, 'n_layers': n_layers, 'n_heads': n_heads},
            'vocab_size': vocab_size,
            'n_params': n_params,
            'val_ppl': best_ppl,
            'tok_param_ratio': args.tok_param,
        }, run_dir / 'best_model.pt')

    print(f"  Results saved to {run_dir / 'results.json'}")

    # Comparison table
    ref_total, ref_emb, ref_trans = 834400, 805000, 29000
    non_rope_33k_trans = 13152
    print(f"\n{'='*70}")
    print(f"  PARAM BUDGET COMPARISON")
    print(f"{'='*70}")
    print(f"  {'':25s} {'RoPE d=32':>14s} {'Byte d=16':>14s} {'800K (50K)':>14s}")
    print(f"  {'Vocab size':25s} {vocab_size:>14,} {256:>14,} {50304:>14,}")
    print(f"  {'Total params':25s} {n_params:>14,} {33648:>14,} {ref_total:>14,}")
    print(f"  {'Embedding params':25s} {emb_params:>14,} {20496:>14,} {ref_emb:>14,}")
    print(f"  {'Position params':25s} {'0 (RoPE)':>14s} {4096:>14,} {16384:>14,}")
    print(f"  {'Transformer params':25s} {trans_params:>14,} {non_rope_33k_trans:>14,} {ref_trans:>14,}")
    print(f"  {'Reasoning ratio':25s} {reasoning_ratio:>13.1f}% {39.1:>13.1f}% {3.5:>13.1f}%")
    print(f"  {'Best val PPL':25s} {best_ppl:>14.2f} {14.67:>14.2f} {'N/A':>14s}")


if __name__ == '__main__':
    main()
