"""Custom vocabulary experiment: tiny BPE vocab for extreme overtraining.

Trains a BPE tokenizer on a text subset, then trains a transformer LM
with a dramatically smaller vocabulary. This eliminates the vocabulary tax
and enables reaching high tok/param ratios locally.

The hypothesis: a 1K-4K token vocabulary with d_model=16 creates a model
where most parameters are in the transformer (reasoning), not embeddings.

Usage:
    # Byte-level (256 vocab, fastest — ~16 min/phase for 1800 t/p on MPS)
    python train_custom_vocab.py --vocab-size 256 --mode byte

    # Custom BPE (1K vocab — ~44 min/phase for 1800 t/p on MPS)
    python train_custom_vocab.py --vocab-size 1024 --tok-param 1800

    # Custom BPE (4K vocab)
    python train_custom_vocab.py --vocab-size 4096 --tok-param 1800

    # With MLM pretrain phase
    python train_custom_vocab.py --vocab-size 1024 --tok-param 1800 --mlm-pretrain
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


class CustomVocabLM(nn.Module):
    """Tiny transformer LM with custom vocabulary size."""

    def __init__(self, vocab_size, d_model=16, n_layers=4, n_heads=4,
                 max_seq_len=1024, dropout=0.0, tie_weights=True):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        d_ff = 4 * d_model
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

    def forward(self, input_ids, causal=True):
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        if causal:
            mask = torch.triu(
                torch.full((S, S), float("-inf"), device=input_ids.device),
                diagonal=1,
            )
        else:
            mask = None

        x = self.transformer(x, mask=mask)
        x = self.norm(x)
        return self.lm_head(x)


MASK_TOKEN_OFFSET = 1  # mask token = vocab_size (one past last real token)


def apply_mlm_mask(input_ids, mask_prob=0.15, vocab_size=256):
    labels = torch.full_like(input_ids, -100)
    prob_matrix = torch.full_like(input_ids, mask_prob, dtype=torch.float)
    masked_indices = torch.bernoulli(prob_matrix).bool()
    labels[masked_indices] = input_ids[masked_indices]

    replace_mask = torch.bernoulli(torch.full_like(prob_matrix, 0.8)).bool() & masked_indices
    input_ids = input_ids.clone()
    input_ids[replace_mask] = vocab_size  # mask token

    random_mask = torch.bernoulli(torch.full_like(prob_matrix, 0.5)).bool() & masked_indices & ~replace_mask
    random_tokens = torch.randint(vocab_size, input_ids.shape, device=input_ids.device)
    input_ids[random_mask] = random_tokens[random_mask]

    return input_ids, labels


class ByteDataset(IterableDataset):
    """Streams raw bytes from a text file as token IDs (0-255)."""

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


class BPEDataset(IterableDataset):
    """Streams pre-tokenized BPE data from a binary file (uint16)."""

    def __init__(self, path, seq_len, max_tokens=None):
        self.path = path
        self.seq_len = seq_len
        self.max_tokens = max_tokens
        self.total_tokens = os.path.getsize(path) // 2

    def __iter__(self):
        data = np.memmap(self.path, dtype=np.uint16, mode='r')
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


def prepare_byte_data(source_bin, output_dir):
    """Convert uint16 tokenized data to raw bytes (UTF-8 text)."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.bytes"
    val_path = output_dir / "val.bytes"

    if train_path.exists() and val_path.exists():
        print(f"  Byte data already exists at {output_dir}")
        return str(train_path), str(val_path)

    print(f"  Converting tokenized data to bytes...")
    data = np.memmap(source_bin, dtype=np.uint16, mode='r')
    text = tokenizer.decode(data[:min(len(data), 2_000_000)].tolist())
    text_bytes = text.encode('utf-8')

    split = int(len(text_bytes) * 0.95)
    with open(train_path, 'wb') as f:
        f.write(text_bytes[:split])
    with open(val_path, 'wb') as f:
        f.write(text_bytes[split:])

    print(f"  Train: {split:,} bytes, Val: {len(text_bytes)-split:,} bytes")
    return str(train_path), str(val_path)


def prepare_bpe_data(source_bin, output_dir, vocab_size):
    """Train a BPE tokenizer on source data and re-tokenize."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    from transformers import AutoTokenizer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tok_path = output_dir / f"tokenizer_{vocab_size}.json"
    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"

    if train_path.exists() and val_path.exists() and tok_path.exists():
        print(f"  BPE data already exists at {output_dir}")
        tok = Tokenizer.from_file(str(tok_path))
        return str(train_path), str(val_path), tok

    # Decode source tokens to text
    print(f"  Decoding source tokens...")
    orig_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    data = np.memmap(source_bin, dtype=np.uint16, mode='r')
    text = orig_tokenizer.decode(data[:min(len(data), 2_000_000)].tolist())

    # Train BPE tokenizer
    print(f"  Training BPE tokenizer with vocab_size={vocab_size}...")
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[MASK]"],
        min_frequency=2,
    )

    # Train from text chunks
    text_chunks = [text[i:i+10000] for i in range(0, len(text), 10000)]
    tok.train_from_iterator(text_chunks, trainer=trainer)
    tok.save(str(tok_path))
    actual_vocab = tok.get_vocab_size()
    print(f"  Actual vocab size: {actual_vocab}")

    # Tokenize all text
    print(f"  Tokenizing with custom BPE...")
    encoded = tok.encode(text)
    all_ids = encoded.ids
    split = int(len(all_ids) * 0.95)

    train_arr = np.array(all_ids[:split], dtype=np.uint16)
    val_arr = np.array(all_ids[split:], dtype=np.uint16)

    with open(train_path, 'wb') as f:
        f.write(train_arr.tobytes())
    with open(val_path, 'wb') as f:
        f.write(val_arr.tobytes())

    print(f"  Train: {len(train_arr):,} tokens, Val: {len(val_arr):,} tokens")
    return str(train_path), str(val_path), tok


@torch.no_grad()
def evaluate(model, val_loader, vocab_size, device, max_batches=50):
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


@torch.no_grad()
def evaluate_mlm(model, val_loader, vocab_size, device, max_batches=50):
    model.eval()
    total_loss = 0
    n = 0
    for batch in val_loader:
        if n >= max_batches:
            break
        ids = batch.to(device).unsqueeze(0) if batch.dim() == 1 else batch.to(device)
        masked_ids, labels = apply_mlm_mask(ids, vocab_size=vocab_size)
        logits = model(masked_ids, causal=False)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               labels.reshape(-1), ignore_index=-100)
        total_loss += loss.item()
        n += 1
    model.train()
    avg = total_loss / max(n, 1)
    return avg, math.exp(min(avg, 20))


def train_phase(model, train_loader, val_loader, vocab_size, device,
                total_tokens, n_params, args, phase_name, mlm=False):
    """Train one phase (MLM or causal)."""
    tokens_per_step = args.batch_size * args.seq_len
    total_steps = total_tokens // tokens_per_step
    warmup_steps = min(500, total_steps // 10)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   betas=(0.9, 0.95), weight_decay=0.1)

    def lr_fn(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    actual_vocab = vocab_size + 1  # model always has mask token slot

    print(f"\n  Phase: {phase_name} ({'MLM' if mlm else 'Causal'})")
    print(f"  Steps: {total_steps:,}, tokens: {total_tokens/1e6:.1f}M, "
          f"tok/param: {total_tokens/n_params:.1f}")

    model.train()
    log = []
    best_loss = float('inf')
    t0 = time.time()
    step = 0
    running_loss = 0
    running_n = 0

    for batch in train_loader:
        if step >= total_steps:
            break

        ids = batch.to(device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)

        if mlm:
            masked_ids, labels = apply_mlm_mask(ids, vocab_size=vocab_size)
            logits = model(masked_ids, causal=False)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   labels.reshape(-1), ignore_index=-100)
        else:
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
        running_n += 1
        step += 1

        if step % 100 == 0:
            avg = running_loss / running_n
            ppl = math.exp(min(avg, 20))
            tok_s = running_n * tokens_per_step / max(time.time() - t0, 1)
            tp = step * tokens_per_step / n_params
            print(f"    step {step:>6,}/{total_steps:,}  loss={avg:.4f}  ppl={ppl:7.1f}  "
                  f"{tok_s:,.0f} tok/s  tp={tp:.1f}", flush=True)
            running_loss = 0
            running_n = 0
            t0 = time.time()

        if step % args.eval_every == 0:
            if mlm:
                val_loss, val_ppl = evaluate_mlm(model, iter(val_loader),
                                                  vocab_size, device)
            else:
                val_loss, val_ppl = evaluate(model, iter(val_loader),
                                              actual_vocab, device)

            improved = ""
            if val_loss < best_loss:
                best_loss = val_loss
                improved = " *BEST*"

            tp = step * tokens_per_step / n_params
            print(f"    >>> EVAL step {step}: val_ppl={val_ppl:.2f} "
                  f"(tok/param={tp:.1f}){improved}", flush=True)

            log.append({
                'step': step,
                'tokens_seen': step * tokens_per_step,
                'val_loss': val_loss,
                'val_ppl': val_ppl,
                'tok_param_ratio': tp,
            })
            model.train()

        if device == "mps" and step % 20 == 0:
            torch.mps.empty_cache()

    # Final eval
    if mlm:
        val_loss, val_ppl = evaluate_mlm(model, iter(val_loader),
                                          vocab_size, device)
    else:
        val_loss, val_ppl = evaluate(model, iter(val_loader),
                                      actual_vocab, device)
    tp = step * tokens_per_step / n_params
    log.append({
        'step': step, 'tokens_seen': step * tokens_per_step,
        'val_loss': val_loss, 'val_ppl': val_ppl, 'tok_param_ratio': tp,
    })

    best_ppl = math.exp(min(best_loss, 20))
    print(f"    Final: val_ppl={val_ppl:.2f}, best={best_ppl:.2f}, "
          f"tok/param={tp:.1f}")

    return log, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--mode", choices=["byte", "bpe"], default="bpe")
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--tok-param", type=float, default=1800,
                        help="Target tok/param ratio")
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--mlm-pretrain", action="store_true",
                        help="Add MLM pretrain phase before causal")
    parser.add_argument("--mlm-tok-param", type=float, default=50,
                        help="MLM pretrain tok/param ratio")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--source-data", type=str,
                        default="pile_runs/micro_train.bin",
                        help="Source tokenized data to build vocab from")
    parser.add_argument("--output-dir", type=str, default="pile_runs/custom_vocab")
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"{'='*70}")
    print(f"  CUSTOM VOCABULARY EXPERIMENT")
    print(f"{'='*70}")
    print(f"  Mode: {args.mode}")
    print(f"  Vocab: {args.vocab_size if args.mode == 'bpe' else 256}")
    print(f"  Architecture: d={args.d_model}, L={args.n_layers}, H={args.n_heads}")
    print(f"  Target: {args.tok_param} tok/param")
    print(f"  Device: {device}")
    print()

    source_bin = Path(args.source_data)
    if not source_bin.is_absolute():
        source_bin = Path(__file__).parent / source_bin
    if not source_bin.exists():
        print(f"ERROR: Source data not found at {source_bin}")
        return

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).parent / out_dir

    # Prepare data
    if args.mode == "byte":
        vocab_size = 256
        data_dir = out_dir / "byte"
        train_path, val_path = prepare_byte_data(str(source_bin), str(data_dir))
        DatasetCls = ByteDataset
    else:
        vocab_size = args.vocab_size
        data_dir = out_dir / f"bpe_{vocab_size}"
        train_path, val_path, tok = prepare_bpe_data(
            str(source_bin), str(data_dir), vocab_size)
        vocab_size = tok.get_vocab_size()
        DatasetCls = BPEDataset

    # Build model (+1 to vocab for mask token if MLM)
    model_vocab = vocab_size + 1  # always reserve mask token slot
    model = CustomVocabLM(
        vocab_size=model_vocab,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.seq_len,
        tie_weights=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    emb_params = sum(p.numel() for n, p in model.named_parameters() if 'emb' in n)
    trans_params = sum(p.numel() for n, p in model.named_parameters()
                       if 'transformer' in n or 'norm' in n)

    print(f"  Model: {n_params:,} params")
    print(f"    Embeddings: {emb_params:,} ({emb_params/n_params*100:.1f}%)")
    print(f"    Transformer: {trans_params:,} ({trans_params/n_params*100:.1f}%)")
    print(f"    Reasoning ratio: {trans_params/n_params*100:.1f}%")

    target_tokens = int(args.tok_param * n_params)
    mlm_tokens = int(args.mlm_tok_param * n_params) if args.mlm_pretrain else 0

    print(f"  Target tokens (causal): {target_tokens/1e6:.1f}M")
    if args.mlm_pretrain:
        print(f"  MLM pretrain tokens: {mlm_tokens/1e6:.1f}M")
    print()

    run_dir = out_dir / f"run_v{vocab_size}_d{args.d_model}_tp{int(args.tok_param)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    all_logs = {}

    # Phase 1 (optional): MLM pretrain
    if args.mlm_pretrain:
        train_ds = DatasetCls(train_path, args.seq_len, max_tokens=mlm_tokens)
        val_ds = DatasetCls(val_path, args.seq_len)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

        log, model = train_phase(
            model, train_loader, val_loader, vocab_size, device,
            mlm_tokens, n_params, args, "MLM Pretrain", mlm=True)
        all_logs['mlm_pretrain'] = log

        # Save MLM checkpoint
        torch.save({'model_state': model.state_dict(), 'n_params': n_params,
                     'vocab_size': model_vocab, 'config': vars(args)},
                    run_dir / "mlm_checkpoint.pt")

    # Phase 2: Causal from MLM (or from scratch)
    phase_name = "Causal from MLM" if args.mlm_pretrain else "Causal"
    train_ds = DatasetCls(train_path, args.seq_len, max_tokens=target_tokens)
    val_ds = DatasetCls(val_path, args.seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    log, model = train_phase(
        model, train_loader, val_loader, vocab_size, device,
        target_tokens, n_params, args, phase_name, mlm=False)
    all_logs['causal_from_mlm' if args.mlm_pretrain else 'causal'] = log

    # Phase 3: Baseline (if MLM pretrain was done)
    if args.mlm_pretrain:
        model_baseline = CustomVocabLM(
            vocab_size=model_vocab,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            max_seq_len=args.seq_len,
            tie_weights=True,
        ).to(device)

        train_ds = DatasetCls(train_path, args.seq_len, max_tokens=target_tokens)
        val_ds = DatasetCls(val_path, args.seq_len)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

        log, _ = train_phase(
            model_baseline, train_loader, val_loader, vocab_size, device,
            target_tokens, n_params, args, "Baseline Causal", mlm=False)
        all_logs['baseline_causal'] = log

    # Save results
    results = {
        'config': {
            'vocab_size': vocab_size,
            'model_vocab': model_vocab,
            'd_model': args.d_model,
            'n_layers': args.n_layers,
            'n_heads': args.n_heads,
            'n_params': n_params,
            'emb_params': emb_params,
            'transformer_params': trans_params,
            'reasoning_ratio': trans_params / n_params,
            'tok_param_target': args.tok_param,
            'mode': args.mode,
            'device': device,
        },
        'logs': all_logs,
    }

    with open(run_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {run_dir / 'results.json'}")

    # Print summary comparison
    print(f"\n{'='*70}")
    print(f"  PARAM BUDGET COMPARISON")
    print(f"{'='*70}")
    print(f"  {'':20s} {'This model':>12s}  {'800K (50K vocab)':>16s}")
    print(f"  {'Vocab size':20s} {vocab_size:>12,}  {50304:>16,}")
    print(f"  {'Total params':20s} {n_params:>12,}  {834400:>16,}")
    print(f"  {'Embedding params':20s} {emb_params:>12,}  {805000:>16,}")
    print(f"  {'Transformer params':20s} {trans_params:>12,}  {29000:>16,}")
    print(f"  {'Reasoning ratio':20s} {trans_params/n_params*100:>11.1f}%  {29000/834400*100:>15.1f}%")


if __name__ == "__main__":
    main()
