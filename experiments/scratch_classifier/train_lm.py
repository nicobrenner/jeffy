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


class RoPEBlock(nn.Module):
    """Pre-LN transformer block with rotary position embeddings on q/k (RoFormer, arXiv 2104.09864). Explicit attention
    (matmul + causal mask) so ONNX export stays simple. Used when --rope: no learned position table, relative positions
    for free (the previous-token head that induction needs is one fixed rotation instead of 256 learned offsets)."""
    def __init__(self, d_model, n_heads, d_ff, dropout, max_seq_len):
        super().__init__()
        self.h = n_heads; self.dk = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model); self.ln2 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model); self.proj = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.drop = nn.Dropout(dropout)
        inv = 1.0 / (10000 ** (torch.arange(0, self.dk, 2).float() / self.dk)); t = torch.arange(max_seq_len).float()
        f = torch.outer(t, inv); self.register_buffer("cos", torch.cat([f, f], -1).cos()[None, None], persistent=False); self.register_buffer("sin", torch.cat([f, f], -1).sin()[None, None], persistent=False)
    def rot(self, x, T):
        c, s_ = self.cos[:, :, :T], self.sin[:, :, :T]; x1, x2 = x[..., : self.dk // 2], x[..., self.dk // 2 :]
        return x * c + torch.cat([-x2, x1], -1) * s_
    def forward(self, x, mask=None):
        B, T, D = x.shape; h = self.ln1(x); q, k, v = self.qkv(h).view(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k = self.rot(q, T), self.rot(k, T)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        if mask is not None: att = att + mask
        att = self.drop(torch.softmax(att, -1)); y = (att @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.drop(self.proj(y)); return x + self.drop(self.ff(self.ln2(x)))


class RoPEStack(nn.Module):
    def __init__(self, layers): super().__init__(); self.layers = nn.ModuleList(layers)
    def forward(self, x, mask=None):
        for l in self.layers: x = l(x, mask=mask)
        return x


class CausalTransformer(nn.Module):
    """Transformer LM with causal attention, depth recurrence, and weight tying.

    Same core as HybridTransformer from our classification experiments,
    but with a causal attention mask and next-token prediction head.
    """

    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=4,
                 d_ff=1024, max_seq_len=256, dropout=0.1,
                 tie_weights=True, n_iterations=1, n_acts=0, pre_ln=False, rope=False):
        super().__init__()
        self.pre_ln = pre_ln; self.rope = rope
        self.d_model = d_model
        self.n_acts = n_acts
        self.act_emb = nn.Embedding(n_acts + 1, d_model, padding_idx=0) if n_acts else None   # act 0 = none (zero vector)
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_iterations = n_iterations

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = None if rope else nn.Embedding(max_seq_len, d_model)   # RoPE: no position table
        self.emb_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=pre_ln,   # pre-LN: induction heads form at d256 (2026-10-27 unit tests)
        )
        self.transformer = RoPEStack([RoPEBlock(d_model, n_heads, d_ff, dropout, max_seq_len) for _ in range(n_layers)]) if rope else nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        if self.pos_emb is not None: nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, input_ids, acts=None):
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_emb(input_ids) + (self.pos_emb(positions) if self.pos_emb is not None else 0)
        a = self.act_emb(acts) if (acts is not None and self.act_emb is not None) else None
        if a is not None: x = x + a
        x = self.emb_dropout(x)

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device),
            diagonal=1,
        )

        for it in range(self.n_iterations):
            if it > 0 and a is not None: x = x + a   # quiet signal re-injected on every recurrence pass
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
    ACTS = ["dunno", "me", "wh", "yn", "hi", "tell"]   # id = index + 1; 0 = none
    def __init__(self, conversations, tokenizer, seq_len, eos_id, sys_end_marker="\n<|user|>", pad_id=0, act_vector=False, act_span=False, ctx_turns=None):
        import re as _re
        ctx_turns = ctx_turns or [set() for _ in conversations]
        self.seq_len, self.pad_id, self.act_vector = seq_len, pad_id, act_vector; self.rows = []
        oracle_id = tokenizer.encode("<|oracle|>")[-1]
        act_re = _re.compile(r"<\|oracle\|> \*(" + "|".join(self.ACTS) + r") ")
        for conv, ctxset in zip(conversations, ctx_turns):
            k = conv.find(sys_end_marker); sys_txt, body = (conv[:k], conv[k:]) if k > 0 else ("", conv)
            sys_ids = tokenizer.encode(sys_txt) if sys_txt else []
            # split body at user-turn boundaries
            parts = [p for p in body.split("\n<|user|>") if p.strip()]
            turns = [("\n<|user|>" + p) for p in parts] if k > 0 else [body]
            turn_ids, turn_acts, turn_lab = [], [], []
            for ti, t in enumerate(turns):
                act = 0
                if act_vector:
                    m2 = act_re.search(t)
                    if m2: act = self.ACTS.index(m2.group(1)) + 1; t = act_re.sub("<|oracle|> ", t, count=1)
                ids_t = tokenizer.encode(t); acts_t = [0] * len(ids_t); lab_t = [1] * len(ids_t)
                if ti in ctxset and oracle_id in ids_t:
                    k0 = len(ids_t) - 1 - ids_t[::-1].index(oracle_id); lab_t[k0 + 1:] = [0] * (len(ids_t) - k0 - 1)   # context-only Oracle turn
                if act and oracle_id in ids_t:
                    k = len(ids_t) - 1 - ids_t[::-1].index(oracle_id)
                    if act_span: acts_t[k:] = [act] * (len(ids_t) - k)
                    else: acts_t[k] = act
                turn_ids.append(ids_t); turn_acts.append(acts_t); turn_lab.append(lab_t)
            i = 0
            while i < len(turn_ids):
                ids = list(sys_ids); acts = [0] * len(sys_ids); lab = [1] * len(sys_ids); j = i
                while j < len(turn_ids) and len(ids) + len(turn_ids[j]) <= seq_len - 1:
                    ids += turn_ids[j]; acts += turn_acts[j]; lab += turn_lab[j]; j += 1
                if j == i:   # single turn longer than the window: hard-cut it
                    ids = (sys_ids + turn_ids[i])[: seq_len - 1]; acts = ([0] * len(sys_ids) + turn_acts[i])[: seq_len - 1]; lab = ([1] * len(sys_ids) + turn_lab[i])[: seq_len - 1]; j = i + 1
                ids.append(eos_id); acts.append(0); lab.append(1)
                self.rows.append((torch.tensor(ids, dtype=torch.long), torch.tensor(acts, dtype=torch.long), torch.tensor(lab, dtype=torch.long)))
                i = max(j - 1, i + 1) if j < len(turn_ids) else j   # next window overlaps by one turn so no turn loses its history entirely
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        ids, acts, keep = self.rows[idx]; out = torch.full((self.seq_len,), self.pad_id, dtype=torch.long); lab = torch.full((self.seq_len,), -100, dtype=torch.long)
        out[: len(ids)] = ids; lab[: len(ids)] = torch.where(keep.bool(), ids, torch.full_like(ids, -100))
        if not self.act_vector: return torch.stack([out, lab])
        ac = torch.zeros(self.seq_len, dtype=torch.long); ac[: len(acts)] = acts; return torch.stack([out, lab, ac])


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
            _rows = [_json.loads(line) for line in f]
        conversations = [r["text"] for r in _rows]
        ctx_turns = [set(r.get("ctx_oracle_turns", [])) for r in _rows]   # DAgger: loss only on the expert's turn, never on the model's own bad turn
        print(f"  {len(conversations):,} conversations ({time.time() - t0:.1f}s)", flush=True)

        print("Tokenizing...", flush=True)
        t0 = time.time()
        conv_texts = conversations; conv_ctx = ctx_turns
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
        conv_texts = conversations; conv_ctx = ctx_turns
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
                train_ds = ConvDataset(conv_texts, tokenizer, args.seq_len, eos_id, act_vector=args.act_vector, act_span=args.act_span, ctx_turns=conv_ctx)
                print(f"  context-only Oracle turns (no loss): {sum(len(c) for c in conv_ctx):,}", flush=True)
                val_ds = ConvDataset(val_conversations, tokenizer, args.seq_len, eos_id, act_vector=args.act_vector, act_span=args.act_span)
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
        n_iterations=args.n_iterations, pre_ln=args.pre_ln, rope=args.rope,
        n_acts=len(ConvDataset.ACTS) if args.act_vector else 0,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    emb_params = model.token_emb.weight.numel() + (model.pos_emb.weight.numel() if model.pos_emb is not None else 0)
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

    ACTS_ON = {"acts": None}
    def unpack(batch):
        batch = batch.to(device)
        if batch.dim() == 3:
            ACTS_ON["acts"] = batch[:, 2] if batch.size(1) == 3 else None
            return batch[:, 0], batch[:, 1]
        ACTS_ON["acts"] = None
        return batch, batch
    def fwd(ids): return model(ids, acts=ACTS_ON["acts"]) if ACTS_ON["acts"] is not None else model(ids)

    def quick_val(max_batches):
        model.eval(); tot = 0.0; n = 0
        with torch.no_grad():
            for b in val_loader:
                ids, lab = unpack(b); lg = fwd(ids)
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
        missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
        if missing or unexpected: print(f"  resume: missing {missing} unexpected {unexpected}", flush=True)
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
            logits = fwd(input_ids)

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
                            "n_iterations": args.n_iterations, "pre_ln": args.pre_ln, "rope": args.rope,
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
                logits = fwd(input_ids)
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
                        "n_iterations": args.n_iterations, "pre_ln": args.pre_ln, "rope": args.rope,
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
                    "n_iterations": args.n_iterations, "pre_ln": args.pre_ln, "rope": args.rope,
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
                        "n_layers": args.n_layers, "d_ff": args.d_ff, "max_seq_len": args.seq_len, "pre_ln": args.pre_ln, "rope": args.rope, "tie_weights": args.tie_weights, "n_iterations": args.n_iterations},
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
        missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
        if missing or unexpected: print(f"  resume: missing {missing} unexpected {unexpected}", flush=True)
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
    parser.add_argument("--act-vector", action="store_true",
                        help="plan act as a learned VECTOR added at the <|oracle|> position (and re-injected each recurrence pass) instead of a text token; "
                             "data must be the chat_v2ct format; the *act text is stripped")
    parser.add_argument("--act-span", action="store_true", help="with --act-vector: the act vector is added at every token of the reply, not only at <|oracle|>")
    parser.add_argument("--keep-every", type=int, default=0,
                        help="also keep a weights-only checkpoint every N epochs (checkpoints/ep_<tag>_<epoch>.pt) so a run can be rewound to a data boundary")
    parser.add_argument("--conv-sequences", action="store_true",
                        help="chat data: one sequence per conversation (system prompt kept, windows start at turn boundaries, padded, "
                             "loss masked on padding) instead of packing all conversations into a stream and cutting fixed chunks")
    parser.add_argument("--rope", action="store_true", help="rotary position embeddings (pre-LN RoPE blocks, no position table)")
    parser.add_argument("--pre-ln", action="store_true", help="pre-LayerNorm transformer layers (norm_first); default post-LN as before")
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
