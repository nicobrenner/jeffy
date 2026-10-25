"""Quick A/B test: does more depth help Tiny Oracle at ~2.7M params?

Runs 4 configs on CPU for 10 epochs each on the v2 chat data.
Compares val_loss and generation quality.

Configs (all ~2.6-2.7M params, weight-tied):
  A) 2L d256 ff1024  — current baseline
  B) 4L d192 ff768   — deeper, narrower
  C) 6L d160 ff640   — deepest
  D) 2L d256 ff1024 x2 iter — depth recurrence (2 layers run twice = 4 effective)
"""

import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_lm import (
    CausalTransformer,
    CustomTokenizerWrapper,
    TextChunkDataset,
)

CONFIGS = {
    "2L_d256_baseline": dict(n_layers=2, d_model=256, n_heads=4, d_ff=1024, n_iterations=1),
    "4L_d192": dict(n_layers=4, d_model=192, n_heads=4, d_ff=768, n_iterations=1),
    "6L_d160": dict(n_layers=6, d_model=160, n_heads=4, d_ff=640, n_iterations=1),
    "2L_d256_x2iter": dict(n_layers=2, d_model=256, n_heads=4, d_ff=1024, n_iterations=2),
}

EPOCHS = 20
BATCH_SIZE = 32
SEQ_LEN = 256
LR = 3e-4
VOCAB_SIZE = 4096
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
CKPT_DIR = Path("pile_runs/rope_byte/oracle_depth_ckpts")
import sys as _sys
SEED = int(_sys.argv[1]) if len(_sys.argv) > 1 else None  # python3 test_oracle_depth.py 1  -> seeds run (3 configs)
if SEED is not None:
    CONFIGS.pop("6L_d160", None)
RESULTS_PATH = Path("pile_runs/rope_byte/oracle_depth_test.json") if SEED is None else Path(f"pile_runs/rope_byte/oracle_depth_seeds_s{SEED}.json")

CHAT_PROMPTS = [
    "<|system|> You are Tiny Oracle, a mass of 2 million parameters. You have strong opinions and limited knowledge. Keep responses short.\n<|user|> What are you?\n<|oracle|>",
    "<|system|> You are Tiny Oracle, a mass of 2 million parameters. You have strong opinions and limited knowledge. Keep responses short.\n<|user|> Tell me something interesting\n<|oracle|>",
    "<|system|> You are Tiny Oracle, a mass of 2 million parameters. You have strong opinions and limited knowledge. Keep responses short.\n<|user|> What do you think about AI?\n<|oracle|>",
]


def generate_response(model, tokenizer, prompt, max_tokens=60, temperature=0.8, top_k=40):
    model.eval()
    torch.manual_seed(42)
    prompt_ids = tokenizer.encode(prompt)
    ids = list(prompt_ids)
    generated = []
    with torch.no_grad():
        for _ in range(max_tokens):
            ids_t = torch.tensor([ids[-SEQ_LEN:]], dtype=torch.long).to(DEVICE)
            logits = model(ids_t)
            next_logits = logits[0, -1, :] / temperature
            v, _ = torch.topk(next_logits, top_k)
            next_logits[next_logits < v[-1]] = float("-inf")
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            tok = tokenizer._tok.id_to_token(next_id) or ""
            if tok in ("<|user|>", "<|system|>", "<eos>") or next_id == 3:
                break
            ids.append(next_id)
            generated.append(next_id)
    return tokenizer.decode(generated).strip()


def main():
    tokenizer = CustomTokenizerWrapper("tokenizer_chat_v2.json")

    print("Loading chat data...", flush=True)
    with open("data/chat_v2_train.jsonl") as f:
        train_convos = [json.loads(line)["text"] for line in f]
    with open("data/chat_v2_val.jsonl") as f:
        val_convos = [json.loads(line)["text"] for line in f]

    eos_id = tokenizer.eos_token_id
    train_tokens = []
    for conv in train_convos:
        train_tokens.extend(tokenizer.encode(conv))
        train_tokens.append(eos_id)
    val_tokens = []
    for conv in val_convos:
        val_tokens.extend(tokenizer.encode(conv))
        val_tokens.append(eos_id)

    train_ids = torch.tensor(train_tokens, dtype=torch.long)
    val_ids = torch.tensor(val_tokens, dtype=torch.long)
    train_ds = TextChunkDataset(train_ids, SEQ_LEN)
    val_ds = TextChunkDataset(val_ids, SEQ_LEN)
    print(f"  Train: {len(train_ds)} chunks, Val: {len(val_ds)} chunks", flush=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # Load any previously completed configs
    results = {}
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            results = json.load(f)
        print(f"  Loaded {len(results)} completed configs from prior run", flush=True)

    for name, cfg in CONFIGS.items():
        if name in results:
            print(f"\n  Skipping {name} (already completed)", flush=True)
            continue

        print(f"\n{'='*60}", flush=True)
        print(f"  Config: {name}", flush=True)
        print(f"{'='*60}", flush=True)

        torch.manual_seed(SEED if SEED is not None else 0)
        model = CausalTransformer(
            vocab_size=VOCAB_SIZE, d_model=cfg["d_model"],
            n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
            d_ff=cfg["d_ff"], max_seq_len=SEQ_LEN,
            tie_weights=True, n_iterations=cfg["n_iterations"],
        ).to(DEVICE)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {n_params:,}", flush=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        best_val = float("inf")
        losses = []

        # Resume from checkpoint if exists
        ckpt_path = CKPT_DIR / (f"{name}_checkpoint.pt" if SEED is None else f"{name}_s{SEED}_checkpoint.pt")
        start_epoch = 0
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = ckpt["epoch"]
            best_val = ckpt["best_val_loss"]
            losses = ckpt.get("losses", [])
            print(f"  Resumed from epoch {start_epoch}, best_val={best_val:.4f}", flush=True)

        t_total = time.time()

        for epoch in range(start_epoch, EPOCHS):
            model.train()
            train_loss_sum = 0
            n_steps = 0
            t0 = time.time()

            for batch in train_loader:
                ids = batch.to(DEVICE)
                logits = model(ids)
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, VOCAB_SIZE),
                    ids[:, 1:].reshape(-1),
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss_sum += loss.item()
                n_steps += 1

            train_loss = train_loss_sum / max(n_steps, 1)

            model.eval()
            val_sum = 0
            val_n = 0
            with torch.no_grad():
                for batch in val_loader:
                    ids = batch.to(DEVICE)
                    logits = model(ids)
                    loss = F.cross_entropy(
                        logits[:, :-1].reshape(-1, VOCAB_SIZE),
                        ids[:, 1:].reshape(-1),
                    )
                    val_sum += loss.item()
                    val_n += 1
            val_loss = val_sum / max(val_n, 1)
            elapsed = time.time() - t0

            tag = ""
            if val_loss < best_val:
                best_val = val_loss
                tag = " *"

            losses.append({"epoch": epoch + 1, "train": train_loss, "val": val_loss})
            print(
                f"  ep {epoch+1:2d}: train={train_loss:.4f} val={val_loss:.4f} "
                f"ppl={math.exp(min(val_loss,20)):.1f} ({elapsed:.0f}s){tag}",
                flush=True,
            )

            # Checkpoint every epoch
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best_val_loss": best_val,
                "losses": losses,
                "config": cfg,
            }, ckpt_path)

        total_time = time.time() - t_total

        print(f"\n  Generation samples:", flush=True)
        samples = []
        for prompt in CHAT_PROMPTS:
            resp = generate_response(model, tokenizer, prompt)
            q = prompt.split("<|user|>")[-1].split("\n<|oracle|>")[0].strip()
            print(f"    Q: {q}", flush=True)
            print(f"    A: {resp[:150]}", flush=True)
            samples.append({"q": q, "a": resp[:200]})

        results[name] = {
            "params": n_params,
            "best_val_loss": best_val,
            "best_ppl": math.exp(min(best_val, 20)),
            "total_time": total_time,
            "losses": losses,
            "samples": samples,
            "config": cfg,
        }

        # Save incremental results after each config
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved results ({len(results)}/{len(CONFIGS)} configs done)", flush=True)

        # Clean up checkpoint for completed config
        if ckpt_path.exists():
            ckpt_path.unlink()

    print(f"\n{'='*60}", flush=True)
    print(f"  SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  {'Config':<25s} {'Params':>10s} {'Val Loss':>10s} {'PPL':>8s} {'Time':>8s}", flush=True)
    for name, r in results.items():
        print(
            f"  {name:<25s} {r['params']:>10,} {r['best_val_loss']:>10.4f} "
            f"{r['best_ppl']:>8.1f} {r['total_time']:>7.0f}s",
            flush=True,
        )

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
