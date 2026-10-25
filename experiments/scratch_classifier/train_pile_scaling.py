"""
Pythia scaling comparison: train models on The Pile at different sizes,
compare against Pythia's published checkpoints at matched token counts.

Thesis: a smaller model trained to higher tok/param matches a larger
model trained to lower tok/param on the same data budget.

Usage:
    # Quick 14M test (~1h A100)
    python train_pile_scaling.py --model-size 14m --tokens 1.5e9

    # Core comparison: train small model on same data budget as larger
    python train_pile_scaling.py --model-size 70m --tokens 1.5e9

    # Full suite
    python train_pile_scaling.py --model-size 160m --tokens 1.5e9

    # Local quick validation
    python train_pile_scaling.py --model-size 14m --tokens 50e6 --device cpu
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

# ---------------------------------------------------------------------------
# Model configs matching Pythia architecture (untied embeddings, 4x FFN)
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    # Tiny configs for extreme overtraining experiments (use --tie-weights)
    "800k": dict(d_model=16,  n_layers=4,  n_heads=4,  label="800K"),
    "1.6m": dict(d_model=32,  n_layers=2,  n_heads=4,  label="1.6M"),
    "3m":   dict(d_model=64,  n_layers=3,  n_heads=4,  label="3M"),
    # Pythia-matching configs (for direct comparison)
    "14m":  dict(d_model=128,  n_layers=6,  n_heads=4,  label="14M"),
    "70m":  dict(d_model=512,  n_layers=6,  n_heads=8,  label="70M"),
    "160m": dict(d_model=768,  n_layers=12, n_heads=12, label="160M"),
    "410m": dict(d_model=1024, n_layers=24, n_heads=16, label="410M"),
    # Custom sizes for scaling curve (with 50K vocab, embeddings dominate at small sizes)
    "20m":  dict(d_model=160,  n_layers=8,  n_heads=8,  label="20M"),
    "50m":  dict(d_model=384,  n_layers=6,  n_heads=6,  label="50M"),
    "100m": dict(d_model=640,  n_layers=8,  n_heads=8,  label="100M"),
    "400m": dict(d_model=1024, n_layers=24, n_heads=16, label="400M"),
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ParallelTransformerLayer(nn.Module):
    """GPT-NeoX style parallel residuals: x = x + Attn(LN(x)) + FFN(LN(x))."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.ln_attn = nn.LayerNorm(d_model)
        self.ln_ff = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None):
        a = self.ln_attn(x)
        f = self.ln_ff(x)
        a, _ = self.attn(a, a, a, attn_mask=attn_mask, is_causal=False)
        return x + a + self.ff(f)


class CausalLM(nn.Module):
    """Standard transformer decoder LM with learned positional embeddings."""

    def __init__(self, vocab_size, d_model, n_layers, n_heads, max_seq_len=1024,
                 dropout=0.0, tie_weights=False, parallel_residuals=False):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.parallel_residuals = parallel_residuals

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        d_ff = 4 * d_model
        if parallel_residuals:
            self.layers = nn.ModuleList([
                ParallelTransformerLayer(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ])
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                dropout=dropout, batch_first=True, activation="gelu",
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        if not hasattr(self.lm_head, '_tied'):
            nn.init.normal_(self.lm_head.weight, std=0.02)

    def forward(self, input_ids, causal=True):
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0)

        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        if causal:
            attn_mask = torch.triu(
                torch.full((S, S), float("-inf"), device=input_ids.device),
                diagonal=1,
            )
        else:
            attn_mask = None

        if self.parallel_residuals:
            for layer in self.layers:
                x = layer(x, attn_mask=attn_mask)
        else:
            x = self.transformer(x, mask=attn_mask)
        x = self.norm(x)
        return self.lm_head(x)


MASK_TOKEN_ID = 50257  # first unused token in padded 50304 vocab


def apply_mlm_mask(input_ids, mask_prob=0.15, vocab_size=50304):
    """BERT-style masking: 15% of tokens selected, 80/10/10 mask/random/keep."""
    labels = torch.full_like(input_ids, -100)
    prob_matrix = torch.full_like(input_ids, mask_prob, dtype=torch.float)
    masked_indices = torch.bernoulli(prob_matrix).bool()
    labels[masked_indices] = input_ids[masked_indices]

    replace_mask = torch.bernoulli(torch.full_like(prob_matrix, 0.8)).bool() & masked_indices
    input_ids = input_ids.clone()
    input_ids[replace_mask] = MASK_TOKEN_ID

    random_mask = torch.bernoulli(torch.full_like(prob_matrix, 0.5)).bool() & masked_indices & ~replace_mask
    random_tokens = torch.randint(vocab_size, input_ids.shape, device=input_ids.device)
    input_ids[random_mask] = random_tokens[random_mask]

    return input_ids, labels


# ---------------------------------------------------------------------------
# Streaming dataset from The Pile
# ---------------------------------------------------------------------------

class PileStreamDataset(IterableDataset):
    """Streams tokenized chunks from The Pile via HuggingFace datasets."""

    def __init__(self, tokenizer, seq_len, max_tokens, split="train", seed=42):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_tokens = max_tokens
        self.split = split
        self.seed = seed

    def __iter__(self):
        from datasets import load_dataset
        ds = load_dataset(
            "monology/pile-uncopyrighted",
            streaming=True,
            split=self.split,
        )
        ds = ds.shuffle(seed=self.seed, buffer_size=10_000)

        buffer = []
        tokens_yielded = 0
        eos_id = self.tokenizer.eos_token_id

        for example in ds:
            text = example["text"]
            ids = self.tokenizer.encode(text)
            buffer.extend(ids)
            buffer.append(eos_id)

            while len(buffer) >= self.seq_len:
                chunk = buffer[:self.seq_len]
                buffer = buffer[self.seq_len:]
                tokens_yielded += self.seq_len

                if tokens_yielded > self.max_tokens:
                    return

                yield torch.tensor(chunk, dtype=torch.long)


class PreTokenizedDataset(IterableDataset):
    """Streams pre-tokenized chunks from a binary file."""

    def __init__(self, path, seq_len, max_tokens=None):
        self.path = path
        self.seq_len = seq_len
        self.max_tokens = max_tokens
        import numpy as np
        self.total_tokens = os.path.getsize(path) // 2  # uint16

    def __iter__(self):
        import numpy as np
        data = np.memmap(self.path, dtype=np.uint16, mode='r')
        n_chunks = len(data) // self.seq_len
        tokens_yielded = 0

        import random
        while True:
            indices = list(range(n_chunks))
            random.shuffle(indices)
            for idx in indices:
                start = idx * self.seq_len
                chunk = torch.from_numpy(data[start:start + self.seq_len].astype(np.int64))
                tokens_yielded += self.seq_len
                if self.max_tokens and tokens_yielded > self.max_tokens:
                    return
                yield chunk


# ---------------------------------------------------------------------------
# Pre-tokenize The Pile to binary for fast repeated access
# ---------------------------------------------------------------------------

def pretokenize_pile(tokenizer, output_path, max_tokens, seq_len):
    """Download and tokenize Pile data into a compact binary file."""
    import numpy as np
    from datasets import load_dataset

    print(f"Pre-tokenizing Pile to {output_path}...")
    print(f"  Target: {max_tokens/1e9:.1f}B tokens")

    ds = load_dataset("monology/pile-uncopyrighted", streaming=True, split="train")
    eos_id = tokenizer.eos_token_id

    buffer = []
    total_tokens = 0
    n_docs = 0
    t0 = time.time()

    # Write in large batches for efficiency
    batch_size = 100_000 * seq_len
    file_tokens = []

    for example in ds:
        text = example["text"]
        ids = tokenizer.encode(text)
        buffer.extend(ids)
        buffer.append(eos_id)
        n_docs += 1

        while len(buffer) >= seq_len:
            chunk = buffer[:seq_len]
            buffer = buffer[seq_len:]
            file_tokens.extend(chunk)
            total_tokens += seq_len

            if len(file_tokens) >= batch_size:
                arr = np.array(file_tokens, dtype=np.uint16)
                with open(output_path, 'ab') as f:
                    f.write(arr.tobytes())
                file_tokens = []

                elapsed = time.time() - t0
                rate = total_tokens / elapsed
                eta = (max_tokens - total_tokens) / rate if rate > 0 else 0
                print(f"  {total_tokens/1e6:.0f}M tokens from {n_docs:,} docs "
                      f"({rate/1e6:.1f}M tok/s, ETA {eta/60:.0f}min)", flush=True)

        if total_tokens >= max_tokens:
            break

    if file_tokens:
        arr = np.array(file_tokens, dtype=np.uint16)
        with open(output_path, 'ab') as f:
            f.write(arr.tobytes())

    elapsed = time.time() - t0
    print(f"  Done: {total_tokens/1e6:.0f}M tokens from {n_docs:,} docs in {elapsed/60:.1f}min")
    print(f"  File: {output_path} ({os.path.getsize(output_path)/1e9:.2f} GB)")
    return total_tokens


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_iter, vocab_size, device, max_batches=50):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in val_iter:
        if n_batches >= max_batches:
            break
        input_ids = batch.to(device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, vocab_size),
            input_ids[:, 1:].reshape(-1),
        )
        total_loss += loss.item()
        n_batches += 1

    model.train()
    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, math.exp(min(avg_loss, 20))


@torch.no_grad()
def evaluate_mlm(model, val_iter, vocab_size, device, max_batches=50):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in val_iter:
        if n_batches >= max_batches:
            break
        input_ids = batch.to(device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        masked_ids, labels = apply_mlm_mask(input_ids, vocab_size=vocab_size)
        logits = model(masked_ids, causal=False)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            labels.reshape(-1),
            ignore_index=-100,
        )
        total_loss += loss.item()
        n_batches += 1

    model.train()
    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, math.exp(min(avg_loss, 20))


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    mlm_mode = getattr(args, 'pretrain_mode', 'causal') == 'mlm'

    # Device selection
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}", flush=True)

    # Tokenizer (local path or download from HF)
    from transformers import AutoTokenizer
    tok_path = args.tokenizer_path or "EleutherAI/gpt-neox-20b"
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    vocab_size = 50304  # Pythia rounds up for efficiency
    print(f"Tokenizer: {tok_path} (vocab={vocab_size})", flush=True)

    # Model config
    cfg = MODEL_CONFIGS[args.model_size]
    seq_len = args.seq_len
    max_tokens = int(float(args.tokens))
    total_train_tokens = int(max_tokens * 0.95)
    total_val_tokens = int(max_tokens * 0.05)

    mode_str = "MLM (bidirectional)" if mlm_mode else "Causal LM"
    print(f"\nExperiment: {cfg['label']} model, {max_tokens/1e9:.2f}B tokens [{mode_str}]", flush=True)
    print(f"  tok/param ratio will be computed after model init", flush=True)

    # Output directory
    out_dir = Path(args.output_dir) / f"pile_{args.model_size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-tokenize data if needed (shared data dir or per-model dir)
    data_dir = Path(args.data_dir) if args.data_dir else out_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    train_bin = data_dir / "train.bin"
    val_bin = data_dir / "val.bin"

    if not train_bin.exists():
        if os.path.exists(train_bin):
            os.remove(train_bin)
        pretokenize_pile(tokenizer, str(train_bin), total_train_tokens, seq_len)

    if not val_bin.exists():
        if os.path.exists(val_bin):
            os.remove(val_bin)
        # Use a different seed region for val
        import numpy as np
        from datasets import load_dataset
        print(f"Pre-tokenizing Pile validation data...")
        ds = load_dataset("monology/pile-uncopyrighted", streaming=True, split="train")
        # Skip ahead to get different data for validation
        eos_id = tokenizer.eos_token_id
        buffer = []
        total = 0
        file_tokens = []
        skip = 500_000  # skip first 500K docs
        for i, example in enumerate(ds):
            if i < skip:
                continue
            text = example["text"]
            ids = tokenizer.encode(text)
            buffer.extend(ids)
            buffer.append(eos_id)
            while len(buffer) >= seq_len:
                chunk = buffer[:seq_len]
                buffer = buffer[seq_len:]
                file_tokens.extend(chunk)
                total += seq_len
            if total >= total_val_tokens:
                break
        arr = np.array(file_tokens[:total], dtype=np.uint16)
        with open(val_bin, 'wb') as f:
            f.write(arr.tobytes())
        print(f"  Val: {total/1e6:.0f}M tokens")

    # Create datasets
    train_dataset = PreTokenizedDataset(str(train_bin), seq_len, max_tokens=total_train_tokens)
    val_dataset = PreTokenizedDataset(str(val_bin), seq_len)

    actual_train_tokens = train_dataset.total_tokens
    print(f"Training data: {actual_train_tokens/1e6:.0f}M tokens on disk")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=2 if device != "mps" else 0,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=0,
    )

    # Model
    use_amp = (device == "cuda")

    model = CausalLM(
        vocab_size=vocab_size,
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        max_seq_len=seq_len,
        dropout=args.dropout,
        tie_weights=args.tie_weights,
        parallel_residuals=args.parallel_residuals,
    ).to(device)

    if args.init_from:
        print(f"Initializing weights from {args.init_from}...", flush=True)
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"  Loaded weights (step {ckpt.get('global_step', '?')}, "
              f"val_ppl {ckpt.get('val_ppl', '?')})", flush=True)

    n_params = sum(p.numel() for p in model.parameters())
    tok_param_ratio = total_train_tokens / n_params

    print(f"\nModel: {n_params:,} params ({n_params/1e6:.1f}M)", flush=True)
    print(f"  d_model={cfg['d_model']}, n_layers={cfg['n_layers']}, "
          f"n_heads={cfg['n_heads']}, d_ff={4*cfg['d_model']}", flush=True)
    print(f"  tie_weights={args.tie_weights}, seq_len={seq_len}", flush=True)
    print(f"  tok/param ratio: {tok_param_ratio:.1f}", flush=True)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # Cosine schedule with warmup
    tokens_per_step = args.batch_size * seq_len * args.accumulate
    total_steps = total_train_tokens // tokens_per_step
    warmup_steps = min(2000, total_steps // 10)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"\nTraining for {total_steps:,} steps ({total_train_tokens/1e9:.2f}B tokens)", flush=True)
    print(f"  batch_size={args.batch_size}, accumulate={args.accumulate}, "
          f"effective_batch={args.batch_size * args.accumulate}", flush=True)
    print(f"  tokens/step={tokens_per_step:,}, warmup={warmup_steps:,} steps", flush=True)
    print(f"  lr={args.lr}, weight_decay=0.1", flush=True)
    print(f"  eval every {args.eval_every} steps, save every {args.save_every} steps", flush=True)
    print(f"{'='*80}\n", flush=True)

    # Resume from checkpoint
    global_step = 0
    tokens_seen = 0
    best_val_loss = float("inf")
    log_entries = []

    resume_path = out_dir / "last_checkpoint.pt"
    if resume_path.exists() and not args.no_resume:
        print(f"Resuming from {resume_path}...", flush=True)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        if use_amp and "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])
        global_step = ckpt["global_step"]
        tokens_seen = ckpt["tokens_seen"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        if os.path.exists(out_dir / "training_log.json"):
            with open(out_dir / "training_log.json") as f:
                log_entries = json.load(f)
        print(f"  Resumed at step {global_step}, {tokens_seen/1e6:.0f}M tokens seen", flush=True)

    # Training
    model.train()
    optimizer.zero_grad()

    running_loss = 0.0
    running_steps = 0
    t_start = time.time()
    t_log = time.time()

    for batch in train_loader:
        if global_step >= total_steps:
            break

        # Skip already-processed steps when resuming
        if tokens_seen > 0 and global_step == 0:
            tokens_seen_batch = (global_step + 1) * tokens_per_step
            if tokens_seen_batch < tokens_seen:
                global_step += 1
                continue

        input_ids = batch.to(device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        with torch.cuda.amp.autocast(enabled=use_amp):
            if mlm_mode:
                masked_ids, labels = apply_mlm_mask(input_ids, vocab_size=vocab_size)
                logits = model(masked_ids, causal=False)
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    labels.reshape(-1),
                    ignore_index=-100,
                )
            else:
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, vocab_size),
                    input_ids[:, 1:].reshape(-1),
                )
            loss = loss / args.accumulate

        scaler.scale(loss).backward()

        running_loss += loss.item() * args.accumulate
        running_steps += 1

        if running_steps % args.accumulate == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            tokens_seen = global_step * tokens_per_step

            # Log every 100 steps
            if global_step % 100 == 0:
                avg_loss = running_loss / running_steps
                ppl = math.exp(min(avg_loss, 20))
                elapsed = time.time() - t_log
                tok_per_sec = (running_steps * args.batch_size * seq_len) / max(elapsed, 1)
                lr_now = scheduler.get_last_lr()[0]

                print(f"  step {global_step:>7,}/{total_steps:,}  "
                      f"loss={avg_loss:.4f}  ppl={ppl:7.1f}  "
                      f"lr={lr_now:.2e}  "
                      f"{tok_per_sec:,.0f} tok/s  "
                      f"tokens={tokens_seen/1e6:.0f}M", flush=True)

                running_loss = 0.0
                running_steps = 0
                t_log = time.time()

            # Eval
            if global_step % args.eval_every == 0:
                if mlm_mode:
                    val_loss, val_ppl = evaluate_mlm(model, iter(val_loader), vocab_size, device)
                else:
                    val_loss, val_ppl = evaluate(model, iter(val_loader), vocab_size, device)
                improved = ""
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    improved = " *BEST*"
                    torch.save({
                        "model_state": model.state_dict(),
                        "config": cfg,
                        "vocab_size": vocab_size,
                        "n_params": n_params,
                        "global_step": global_step,
                        "tokens_seen": tokens_seen,
                        "val_loss": val_loss,
                        "val_ppl": val_ppl,
                        "tok_param_ratio": tok_param_ratio,
                        "pretrain_mode": "mlm" if mlm_mode else "causal",
                    }, out_dir / "best_model.pt")

                entry = {
                    "step": global_step,
                    "tokens_seen": tokens_seen,
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "tok_param_ratio": tokens_seen / n_params,
                    "timestamp": time.time(),
                }
                log_entries.append(entry)

                with open(out_dir / "training_log.json", 'w') as f:
                    json.dump(log_entries, f, indent=2)

                print(f"  >>> EVAL step {global_step}: val_loss={val_loss:.4f} "
                      f"val_ppl={val_ppl:.2f} "
                      f"(tok/param={tokens_seen/n_params:.1f}){improved}", flush=True)

                model.train()

            # Checkpoint
            if global_step % args.save_every == 0:
                torch.save({
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict() if use_amp else None,
                    "config": cfg,
                    "vocab_size": vocab_size,
                    "n_params": n_params,
                    "global_step": global_step,
                    "tokens_seen": tokens_seen,
                    "best_val_loss": best_val_loss,
                }, out_dir / "last_checkpoint.pt")

        if device == "mps" and running_steps % 20 == 0:
            torch.mps.empty_cache()

    # Final eval
    if mlm_mode:
        val_loss, val_ppl = evaluate_mlm(model, iter(val_loader), vocab_size, device)
    else:
        val_loss, val_ppl = evaluate(model, iter(val_loader), vocab_size, device)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            "model_state": model.state_dict(),
            "config": cfg,
            "vocab_size": vocab_size,
            "n_params": n_params,
            "global_step": global_step,
            "tokens_seen": tokens_seen,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "tok_param_ratio": tok_param_ratio,
            "pretrain_mode": "mlm" if mlm_mode else "causal",
        }, out_dir / "best_model.pt")

    elapsed_total = time.time() - t_start
    best_ppl = math.exp(min(best_val_loss, 20))

    print(f"\n{'='*80}", flush=True)
    print(f"Training complete!", flush=True)
    print(f"  Model: {cfg['label']} ({n_params:,} params)", flush=True)
    print(f"  Tokens: {tokens_seen/1e9:.2f}B ({tok_param_ratio:.1f} tok/param)", flush=True)
    print(f"  Final val_loss: {val_loss:.4f} (ppl={val_ppl:.2f})", flush=True)
    print(f"  Best val_loss:  {best_val_loss:.4f} (ppl={best_ppl:.2f})", flush=True)
    print(f"  Time: {elapsed_total/3600:.1f}h", flush=True)
    print(f"  Output: {out_dir}", flush=True)

    # Save final summary
    summary = {
        "model_size": args.model_size,
        "model_label": cfg["label"],
        "n_params": n_params,
        "tokens_trained": tokens_seen,
        "tok_param_ratio": tok_param_ratio,
        "best_val_loss": best_val_loss,
        "best_val_ppl": best_ppl,
        "final_val_loss": val_loss,
        "final_val_ppl": val_ppl,
        "total_steps": global_step,
        "training_time_hours": elapsed_total / 3600,
        "device": device,
        "config": cfg,
        "seq_len": seq_len,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "tie_weights": args.tie_weights,
        "pretrain_mode": "mlm" if mlm_mode else "causal",
    }
    with open(out_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Pythia scaling comparison on The Pile")
    parser.add_argument("--model-size", type=str, required=True,
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model size to train")
    parser.add_argument("--tokens", type=str, default="1.5e9",
                        help="Total tokens to train on (e.g. 1.5e9)")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--accumulate", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--tie-weights", action="store_true", default=False,
                        help="Tie input/output embeddings (Pythia does NOT)")
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--output-dir", type=str, default="pile_runs")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--tokenizer-path", type=str, default=None,
                        help="Local path to tokenizer (avoids HF download)")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--parallel-residuals", action="store_true",
                        help="Use GPT-NeoX parallel residuals (for ablation)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory with pre-tokenized train.bin/val.bin (shared across model sizes)")
    parser.add_argument("--pretrain-mode", type=str, default="causal",
                        choices=["causal", "mlm"],
                        help="Training mode: causal (next-token) or mlm (masked LM, bidirectional)")
    parser.add_argument("--init-from", type=str, default=None,
                        help="Path to a checkpoint to initialize weights from (e.g. MLM-pretrained model for causal fine-tuning)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
