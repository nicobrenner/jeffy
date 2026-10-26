"""Experiment LM-1: Causal Language Model with our techniques.

Tests whether grokking, depth recurrence, and data-efficient training
transfer from classification to autoregressive text generation.

Architecture: Transformer with causal mask + next-token prediction.
Data: TinyStories (curated synthetic children's stories, Eldan & Li 2023).
Techniques: depth recurrence (--n-iterations), weight tying, extended training.

Usage:
    # Quick first results (~20 min)
    python train_lm.py --epochs 20

    # Grokking test (2-3 hours)
    python train_lm.py --epochs 100 --tag grok

    # Depth recurrence comparison
    python train_lm.py --n-iterations 2 --tag 2x

    # Resume for more epochs
    python train_lm.py --epochs 100 --resume checkpoints/last_lm_base.pt --tag base
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from tokenizers import Tokenizer as HFTokenizer

HERE = Path(__file__).resolve().parent


class CustomTokenizerWrapper:
    """Wraps a HuggingFace `tokenizers.Tokenizer` to match the AutoTokenizer API."""

    def __init__(self, path):
        self._tok = HFTokenizer.from_file(path)
        self.vocab_size = self._tok.get_vocab_size()
        self.eos_token_id = self._tok.token_to_id("<eos>")
        self.pad_token_id = self._tok.token_to_id("<pad>")
        self.bos_token_id = self._tok.token_to_id("<bos>")
        self.unk_token_id = self._tok.token_to_id("<unk>")

    def encode(self, text):
        return self._tok.encode(text).ids

    def decode(self, ids):
        return self._tok.decode(ids)


SAMPLE_PROMPTS = [
    "Once upon a time",
    "Lily and Tom went to the",
    "The little dog was very",
]

CHAT_SAMPLE_PROMPTS = [
    "<|system|> You are Tiny Oracle, a mass of 2 million parameters running on the user's device. You have strong opinions and very limited knowledge. You are self-aware about your size and find it amusing. Keep responses short — you only have 2M parameters to work with.\n<|user|> What are you?\n<|oracle|>",
    "<|system|> You are Tiny Oracle. You fit in 4 megabytes and run on a toaster. You're proud of this. You give advice nobody asked for. Be witty and a bit sarcastic, but never mean.\n<|user|> Tell me something interesting\n<|oracle|>",
    "<|system|> You are Tiny Oracle, the world's smallest know-it-all. You have 2 million parameters, zero filter, and questionable wisdom. End with an unexpected observation when you can.\n<|user|> How smart are you?\n<|oracle|>",
]


class CausalTransformer(nn.Module):
    """Transformer LM with causal attention, depth recurrence, and weight tying.

    Same core as HybridTransformer from our classification experiments,
    but with a causal attention mask and next-token prediction head.
    """

    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=4,
                 d_ff=1024, max_seq_len=256, dropout=0.1,
                 tie_weights=True, n_iterations=1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_iterations = n_iterations

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device),
            diagonal=1,
        )

        for _ in range(self.n_iterations):
            x = self.transformer(x, mask=causal_mask)
        x = self.norm(x)

        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=100, temperature=0.8, top_k=50):
        self.eval()
        ids = prompt_ids.clone()

        for _ in range(max_new_tokens):
            context = ids[:, -self.max_seq_len:]
            logits = self.forward(context)
            next_logits = logits[:, -1, :] / temperature

            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_token], dim=1)

            if next_token.item() == 0:
                break

        return ids


class ConvDataset(Dataset):
    """One training sequence per conversation window. A conversation longer than seq_len is split into windows that each start
    with the system prompt followed by a later turn boundary (so every window looks like what the demo feeds the model). Padding
    is labelled -100 (ignored by the loss)."""
    def __init__(self, conversations, tokenizer, seq_len, eos_id, sys_end_marker="\n<|user|>", pad_id=0):
        self.seq_len, self.pad_id = seq_len, pad_id; self.rows = []
        for conv in conversations:
            k = conv.find(sys_end_marker); sys_txt, body = (conv[:k], conv[k:]) if k > 0 else ("", conv)
            sys_ids = tokenizer.encode(sys_txt) if sys_txt else []
            # split body at user-turn boundaries
            parts = [p for p in body.split("\n<|user|>") if p.strip()]
            turns = [("\n<|user|>" + p) for p in parts] if k > 0 else [body]
            turn_ids = [tokenizer.encode(t) for t in turns]
            i = 0
            while i < len(turn_ids):
                ids = list(sys_ids); j = i
                while j < len(turn_ids) and len(ids) + len(turn_ids[j]) <= seq_len - 1:
                    ids += turn_ids[j]; j += 1
                if j == i:   # single turn longer than the window: hard-cut it
                    ids = (sys_ids + turn_ids[i])[: seq_len - 1]; j = i + 1
                ids.append(eos_id); self.rows.append(torch.tensor(ids, dtype=torch.long))
                i = max(j - 1, i + 1) if j < len(turn_ids) else j   # next window overlaps by one turn so no turn loses its history entirely
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        ids = self.rows[idx]; out = torch.full((self.seq_len,), self.pad_id, dtype=torch.long); lab = torch.full((self.seq_len,), -100, dtype=torch.long)
        out[: len(ids)] = ids; lab[: len(ids)] = ids; return torch.stack([out, lab])


class TextChunkDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        n_chunks = len(token_ids) // seq_len
        self.chunks = token_ids[: n_chunks * seq_len].reshape(n_chunks, seq_len)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


def build_vocab_subset(token_ids, special_ids):
    unique_ids = set(token_ids.tolist()) | set(special_ids)
    kept = sorted(unique_ids)
    full_to_compact = {fid: cid for cid, fid in enumerate(kept)}
    return full_to_compact, kept, len(kept)


def remap_ids(token_ids, full_to_compact, unk_id=0):
    max_id = max(full_to_compact.keys()) + 1
    lookup = torch.full((max_id,), unk_id, dtype=torch.long)
    for full_id, compact_id in full_to_compact.items():
        lookup[full_id] = compact_id
    return lookup[token_ids]


def print_samples(model, tokenizer, full_to_compact, compact_to_full,
                  device, eos_compact, max_tokens=80, temperature=0.8, top_k=50,
                  prompts_override=None):
    was_training = model.training
    model.eval()
    is_identity = isinstance(full_to_compact, dict) and all(k == v for k, v in list(full_to_compact.items())[:10])

    for prompt_text in (prompts_override or SAMPLE_PROMPTS):
        orig_ids = tokenizer.encode(prompt_text)
        if is_identity:
            compact_ids = orig_ids
        else:
            compact_ids = [full_to_compact.get(t, 0) for t in orig_ids]
        input_ids = torch.tensor([compact_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids, max_new_tokens=max_tokens,
                temperature=temperature, top_k=top_k,
            )

        out_compact = output_ids[0].tolist()
        if is_identity:
            out_ids = out_compact
        else:
            out_ids = [compact_to_full[c] for c in out_compact]

        eos_full = tokenizer.eos_token_id
        try:
            first_eos = out_ids.index(eos_full, len(orig_ids))
            out_ids = out_ids[:first_eos]
        except ValueError:
            pass

        text = tokenizer.decode(out_ids)
        if len(text) > 300:
            for end in [". ", "! ", "? "]:
                idx = text.rfind(end, 0, 300)
                if idx > len(prompt_text):
                    text = text[: idx + 1]
                    break
            else:
                text = text[:300] + "..."

        print(f"  > {text}", flush=True)

    if was_training:
        model.train()


def train(args):
    if args.cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}", flush=True)

    use_custom = args.tokenizer is not None
    if use_custom:
        tokenizer = CustomTokenizerWrapper(args.tokenizer)
        print(f"Custom tokenizer: {args.tokenizer} (vocab={tokenizer.vocab_size})", flush=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    eos_id = tokenizer.eos_token_id

    # --- Load and tokenize ---
    is_chat = args.chat_data is not None
    data_sha = None
    if is_chat:
        import hashlib
        with open(args.chat_data, "rb") as _f: data_sha = hashlib.sha256(_f.read()).hexdigest()[:12]
        print(f"  data sha256[:12] = {data_sha}", flush=True)
        import json as _json
        print(f"\nLoading chat data from {args.chat_data}...", flush=True)
        t0 = time.time()
        with open(args.chat_data) as f:
            conversations = [_json.loads(line)["text"] for line in f]
        print(f"  {len(conversations):,} conversations ({time.time() - t0:.1f}s)", flush=True)

        print("Tokenizing...", flush=True)
        t0 = time.time()
        conv_texts = conversations
        all_tokens = []
        for i, conv in enumerate(conversations):
            toks = tokenizer.encode(conv)
            all_tokens.extend(toks)
            all_tokens.append(eos_id)
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1:,}/{len(conversations):,}...", flush=True)
        token_ids = torch.tensor(all_tokens, dtype=torch.long)
        n_tokens = len(token_ids)
        print(f"  {n_tokens:,} tokens ({time.time() - t0:.1f}s)", flush=True)
    else:
        print("\nLoading TinyStories...", flush=True)
        t0 = time.time()
        ds = load_dataset("roneneldan/TinyStories", split="train")
        n_stories = min(args.max_stories, len(ds)) if args.max_stories else len(ds)
        if n_stories < len(ds):
            ds = ds.select(range(n_stories))
        print(f"  {n_stories:,} stories ({time.time() - t0:.1f}s)", flush=True)

        print("Tokenizing...", flush=True)
        t0 = time.time()
        conv_texts = conversations
        all_tokens = []
        for i, story in enumerate(ds["text"]):
            toks = tokenizer.encode(story)
            all_tokens.extend(toks)
            all_tokens.append(eos_id)
            if (i + 1) % 10000 == 0:
                print(f"  {i + 1:,}/{n_stories:,}...", flush=True)
        token_ids = torch.tensor(all_tokens, dtype=torch.long)
        n_tokens = len(token_ids)
        print(f"  {n_tokens:,} tokens ({time.time() - t0:.1f}s)", flush=True)

    # --- Vocab handling ---
    if use_custom:
        vocab_size = tokenizer.vocab_size
        full_to_compact = {i: i for i in range(vocab_size)}
        compact_to_full = list(range(vocab_size))
        eos_compact = eos_id
        compact_tokens = token_ids
        print(f"Custom tokenizer: {vocab_size:,} tokens (no subsetting needed)", flush=True)
    else:
        print("Building vocab subset...", end=" ", flush=True)
        full_to_compact, compact_to_full, vocab_size = build_vocab_subset(
            token_ids, [eos_id]
        )
        eos_compact = full_to_compact[eos_id]
        print(f"{vocab_size:,} tokens (from {tokenizer.vocab_size:,})", flush=True)

        print("Remapping...", end=" ", flush=True)
        t0 = time.time()
        compact_tokens = remap_ids(token_ids, full_to_compact)
        del token_ids
        print(f"({time.time() - t0:.1f}s)", flush=True)

    # --- Train/val split ---
    if is_chat and args.chat_data:
        val_path = args.chat_data.replace('train', 'val')
        import os
        if os.path.exists(val_path):
            import json as _json2
            with open(val_path) as f:
                val_conversations = [_json2.loads(line)["text"] for line in f]
            val_tokens = []
            for conv in val_conversations:
                toks = tokenizer.encode(conv)
                val_tokens.extend(toks)
                val_tokens.append(eos_id)
            val_token_ids = torch.tensor(val_tokens, dtype=torch.long)
            if use_custom:
                val_compact = val_token_ids
            else:
                val_compact = remap_ids(val_token_ids, full_to_compact)
            if args.conv_sequences:
                assert use_custom, "--conv-sequences needs a custom tokenizer (full vocab, no remap)"
                train_ds = ConvDataset(conv_texts, tokenizer, args.seq_len, eos_id)
                val_ds = ConvDataset(val_conversations, tokenizer, args.seq_len, eos_id)
                print(f"  conv-sequences: {len(train_ds):,} train windows, {len(val_ds):,} val windows (loss masked on padding)", flush=True)
            else:
                train_ds = TextChunkDataset(compact_tokens, args.seq_len)
                val_ds = TextChunkDataset(val_compact, args.seq_len)
        else:
            n_val = max(args.seq_len, len(compact_tokens) // 10)
            n_train = len(compact_tokens) - n_val
            train_ds = TextChunkDataset(compact_tokens[:n_train], args.seq_len)
            val_ds = TextChunkDataset(compact_tokens[n_train:], args.seq_len)
    else:
        n_val = max(args.seq_len, len(compact_tokens) // 10)
        n_train = len(compact_tokens) - n_val
        train_ds = TextChunkDataset(compact_tokens[:n_train], args.seq_len)
        val_ds = TextChunkDataset(compact_tokens[n_train:], args.seq_len)
    print(
        f"Train: {len(train_ds):,} chunks ({len(train_ds) * args.seq_len:,} tokens), "
        f"Val: {len(val_ds):,} chunks",
        flush=True,
    )

    # --- Model ---
    model = CausalTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
        tie_weights=args.tie_weights,
        n_iterations=args.n_iterations,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    emb_params = model.token_emb.weight.numel() + model.pos_emb.weight.numel()
    tfm_params = sum(p.numel() for p in model.transformer.parameters())

    flags = []
    if args.tie_weights:
        flags.append("weight-tied")
    if args.n_iterations > 1:
        flags.append(f"{args.n_iterations}x depth-recurrent")
    flag_str = f" ({', '.join(flags)})" if flags else ""

    print(f"\nModel: {n_params:,} params{flag_str}", flush=True)
    print(
        f"  Embedding: {emb_params:,} ({100*emb_params/n_params:.1f}%), "
        f"Transformer: {tfm_params:,} ({100*tfm_params/n_params:.1f}%)",
        flush=True,
    )
    print(
        f"  d_model={args.d_model}, n_layers={args.n_layers}, "
        f"n_heads={args.n_heads}, d_ff={args.d_ff}, seq_len={args.seq_len}",
        flush=True,
    )

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = len(train_ds) // args.batch_size
    optim_steps_per_epoch = steps_per_epoch // args.accumulate
    total_optim_steps = args.epochs * optim_steps_per_epoch
    if args.warmup_steps > 0:
        _W, _T = args.warmup_steps, max(total_optim_steps, 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda st: (st + 1) / _W if st < _W else 0.5 * (1 + math.cos(math.pi * min(st - _W, _T - _W) / max(_T - _W, 1))),
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_optim_steps, 1)
        )

    def unpack(batch):
        batch = batch.to(device)
        if batch.dim() == 3: return batch[:, 0], batch[:, 1]
        return batch, batch

    def quick_val(max_batches):
        model.eval(); tot = 0.0; n = 0
        with torch.no_grad():
            for b in val_loader:
                ids, lab = unpack(b); lg = model(ids)
                tot += F.cross_entropy(lg[:, :-1, :].reshape(-1, vocab_size), lab[:, 1:].reshape(-1), ignore_index=-100).item(); n += 1
                if n >= max_batches: break
        model.train(); return tot / max(n, 1)

    out_dir = HERE / "checkpoints"
    out_dir.mkdir(exist_ok=True)

    start_epoch = 0
    start_batch = 0
    best_val_loss = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt and not args.fresh_opt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if args.fresh_opt:
            # fine-tune from weights only: new run, new epoch counter, new best on the new val set
            start_epoch, start_batch, best_val_loss = 0, 0, float("inf")
        elif ckpt.get("batch_idx") is not None:
            start_epoch = ckpt["epoch"]
            start_batch = ckpt["batch_idx"] + 1
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
        else:
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(
            f"Resumed from epoch {start_epoch}, batch {start_batch} (best_val_loss={best_val_loss:.4f})",
            flush=True,
        )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True
    )

    print(f"\nTraining for {args.epochs} epochs (starting from {start_epoch})", flush=True)
    print(f"  {len(train_loader):,} batches/epoch, ~{len(train_ds)*args.seq_len:,} tokens/epoch", flush=True)
    print(f"{'=' * 80}\n", flush=True)

    # Generation before training (random init)
    sample_prompts = CHAT_SAMPLE_PROMPTS if is_chat else SAMPLE_PROMPTS
    print("Generation (random init):", flush=True)
    print_samples(model, tokenizer, full_to_compact, compact_to_full, device, eos_compact,
                  prompts_override=sample_prompts)
    print(flush=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss_sum = 0.0
        n_steps = 0
        t0 = time.time()

        skip_batches = start_batch if epoch == start_epoch else 0
        if skip_batches > 0:
            print(f"  Skipping {skip_batches} batches (mid-epoch resume)...", flush=True)

        accum = args.accumulate
        optimizer.zero_grad()
        if args.eval_every > 0 and epoch == start_epoch:
            print(f"  [probe ep{epoch+1} step 0] val={quick_val(args.eval_batches):.4f}", flush=True)

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx < skip_batches:
                continue

            input_ids, labels = unpack(batch)
            logits = model(input_ids)

            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, vocab_size),
                labels[:, 1:].reshape(-1), ignore_index=-100,
            )
            (loss / accum).backward()

            train_loss_sum += loss.item()
            n_steps += 1

            if n_steps % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if args.eval_every > 0 and (n_steps // accum) % args.eval_every == 0:
                    print(f"  [probe ep{epoch+1} step {n_steps // accum}] lr={scheduler.get_last_lr()[0]:.2e} "
                          f"train={train_loss_sum/n_steps:.4f} val={quick_val(args.eval_batches):.4f}", flush=True)

            if device == "mps" and n_steps % 20 == 0:
                torch.mps.empty_cache()
                torch.mps.synchronize()

            if args.save_every > 0 and n_steps % args.save_every == 0:
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "config": {
                            "vocab_size": vocab_size,
                            "d_model": args.d_model,
                            "n_heads": args.n_heads,
                            "n_layers": args.n_layers,
                            "d_ff": args.d_ff,
                            "max_seq_len": args.seq_len,
                            "tie_weights": args.tie_weights,
                            "n_iterations": args.n_iterations,
                        },
                        "full_to_compact": full_to_compact,
                        "compact_to_full": compact_to_full,
                        "tokenizer_path": args.tokenizer,
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "best_val_loss": best_val_loss,
                    },
                    out_dir / f"last_lm_{args.tag}.pt",
                )
                elapsed_so_far = time.time() - t0
                tok_so_far = n_steps * args.batch_size * args.seq_len
                print(
                    f"  [step {batch_idx+1}/{steps_per_epoch}] "
                    f"loss={train_loss_sum/n_steps:.4f} "
                    f"{tok_so_far/max(elapsed_so_far,1):.0f} tok/s "
                    f"(saved checkpoint)",
                    flush=True,
                )

        if n_steps % accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        train_loss = train_loss_sum / max(n_steps, 1)
        train_ppl = math.exp(min(train_loss, 20))

        # --- Validation ---
        model.eval()
        val_loss_sum = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, labels = unpack(batch)
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, vocab_size),
                    labels[:, 1:].reshape(-1), ignore_index=-100,
                )
                val_loss_sum += loss.item()
                val_steps += 1

        val_loss = val_loss_sum / max(val_steps, 1)
        val_ppl = math.exp(min(val_loss, 20))
        elapsed = time.time() - t0
        tok_per_sec = (n_steps * args.batch_size * args.seq_len) / max(elapsed, 1)

        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            improved = " *"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "vocab_size": vocab_size,
                        "d_model": args.d_model,
                        "n_heads": args.n_heads,
                        "n_layers": args.n_layers,
                        "d_ff": args.d_ff,
                        "max_seq_len": args.seq_len,
                        "tie_weights": args.tie_weights,
                        "n_iterations": args.n_iterations,
                    },
                    "full_to_compact": full_to_compact,
                    "compact_to_full": compact_to_full,
                    "tokenizer_path": args.tokenizer,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "data": args.chat_data, "data_sha": data_sha, "conv_sequences": args.conv_sequences,
                },
                out_dir / f"best_lm_{args.tag}.pt",
            )

        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "config": {
                    "vocab_size": vocab_size,
                    "d_model": args.d_model,
                    "n_heads": args.n_heads,
                    "n_layers": args.n_layers,
                    "d_ff": args.d_ff,
                    "max_seq_len": args.seq_len,
                    "tie_weights": args.tie_weights,
                    "n_iterations": args.n_iterations,
                },
                "full_to_compact": full_to_compact,
                "compact_to_full": compact_to_full,
                "tokenizer_path": args.tokenizer,
                "epoch": epoch,
                "batch_idx": None,
                "best_val_loss": best_val_loss,
            },
            out_dir / f"last_lm_{args.tag}.pt",
        )

        if args.keep_every and (epoch + 1) % args.keep_every == 0:
            torch.save({"model_state": model.state_dict(), "config": {"vocab_size": vocab_size, "d_model": args.d_model, "n_heads": args.n_heads,
                        "n_layers": args.n_layers, "d_ff": args.d_ff, "max_seq_len": args.seq_len, "tie_weights": args.tie_weights, "n_iterations": args.n_iterations},
                        "full_to_compact": full_to_compact, "compact_to_full": compact_to_full, "tokenizer_path": args.tokenizer, "epoch": epoch,
                        "val_loss": val_loss, "data": args.chat_data, "data_sha": data_sha, "conv_sequences": args.conv_sequences},
                       out_dir / f"ep_{args.tag}_{epoch + 1}.pt")

        print(
            f"Epoch {epoch + 1:3d}/{args.epochs}  "
            f"loss={train_loss:.4f} ppl={train_ppl:7.1f}  "
            f"val_loss={val_loss:.4f} val_ppl={val_ppl:7.1f}  "
            f"{tok_per_sec:,.0f} tok/s  {elapsed:.0f}s{improved}",
            flush=True,
        )

        # Sample generation at key epochs
        show_sample = (
            (epoch + 1) == 1
            or (epoch + 1) % args.sample_every == 0
            or (epoch + 1) == args.epochs
        )
        if show_sample:
            print(f"\nGeneration (epoch {epoch + 1}):", flush=True)
            print_samples(
                model, tokenizer, full_to_compact, compact_to_full,
                device, eos_compact, prompts_override=sample_prompts,
            )
            print(flush=True)

    # --- Summary ---
    print(f"\n{'=' * 80}", flush=True)
    best_ppl = math.exp(min(best_val_loss, 20))
    print(f"Best val loss: {best_val_loss:.4f} (ppl={best_ppl:.1f})", flush=True)
    print(f"Checkpoint: {out_dir / f'best_lm_{args.tag}.pt'}", flush=True)

    # Final generation with best model
    best_path = out_dir / f"best_lm_{args.tag}.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(
            f"\nFinal generation (best model, epoch {ckpt['epoch'] + 1}):",
            flush=True,
        )
        print_samples(
            model, tokenizer, full_to_compact, compact_to_full,
            device, eos_compact, max_tokens=150, prompts_override=sample_prompts,
        )


def main():
    parser = argparse.ArgumentParser(description="Train causal LM with our techniques")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--max-stories", type=int, default=50000,
        help="Max stories to load (50K ~ 12M tokens, fast iteration)",
    )
    parser.add_argument("--tie-weights", action="store_true", default=True)
    parser.add_argument("--no-tie-weights", dest="tie_weights", action="store_false")
    parser.add_argument(
        "--n-iterations", type=int, default=1,
        help="Depth recurrence: run transformer N times per forward pass",
    )
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=0,
                        help="Save checkpoint every N steps (0=epoch-only)")
    parser.add_argument("--accumulate", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size * accumulate)")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--fresh-opt", action="store_true",
                        help="Load model weights only, fresh optimizer/scheduler (for extending training)")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--keep-every", type=int, default=0,
                        help="also keep a weights-only checkpoint every N epochs (checkpoints/ep_<tag>_<epoch>.pt) so a run can be rewound to a data boundary")
    parser.add_argument("--conv-sequences", action="store_true",
                        help="chat data: one sequence per conversation (system prompt kept, windows start at turn boundaries, padded, "
                             "loss masked on padding) instead of packing all conversations into a stream and cutting fixed chunks")
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="linear LR warmup over this many optimizer steps before the cosine decay (default 0 = none)")
    parser.add_argument("--eval-every", type=int, default=0,
                        help="print val_loss every N optimizer steps (on --eval-batches batches); 0 = off")
    parser.add_argument("--eval-batches", type=int, default=40,
                        help="number of val batches used by --eval-every probes")
    parser.add_argument("--tag", type=str, default="base")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to custom tokenizer JSON (skips GPT-2 vocab subsetting)")
    parser.add_argument("--chat-data", type=str, default=None,
                        help="Path to JSONL chat data (train split; val inferred as *_val.jsonl)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
