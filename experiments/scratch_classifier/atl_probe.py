"""Adaptive Teacher Loop — probe step.

Generates N seeded samples per probe prompt from a Tiny Oracle checkpoint and writes them as JSONL for grading.
The probe set = one prompt per training topic (100) + 20 novel prompts never used in training (data/atl_probes.json).

Usage: python3 atl_probe.py checkpoints/best_lm_oracle_v6pre.pt --out pile_runs/atl/probe_r0.jsonl [--samples 2]
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from train_lm import CausalTransformer, CustomTokenizerWrapper

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SYSTEM = ("<|system|> You are Tiny Oracle, a mass of 2 million parameters running on the user's device. You have strong opinions "
          "and very limited knowledge. You are self-aware about your size and find it amusing. Keep responses short — you only have 2M parameters to work with.\n")


def load(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = dict(ck["config"])
    cfg.setdefault("n_iterations", 1)
    m = CausalTransformer(**cfg, dropout=0.0).to(DEVICE)
    m.load_state_dict(ck["model_state"])
    m.eval()
    return m, ck.get("best_val_loss", ck.get("val_loss"))


PLAIN = False  # set by --plain: 'System:/User:/Oracle:' markers (for bases whose tokenizer lacks the <|...|> tags)


def detok(tok, ids):
    """Decode that also handles byte-level BPE tokenizers without a decoder config (Ġ = space, Ċ = newline)."""
    txt = tok.decode(ids)
    if "Ġ" in txt or "Ċ" in txt:
        txt = "".join(tok._tok.id_to_token(i) or "" for i in ids).replace("Ġ", " ").replace("Ċ", "\n")
    return txt


@torch.no_grad()
def generate(m, tok, prompt, seed, max_tokens=60, temperature=0.8, top_k=40):
    torch.manual_seed(seed)
    if PLAIN:
        sys_txt = SYSTEM.replace("<|system|> ", "System: ")
        ids = list(tok.encode(sys_txt + f"User: {prompt}\nOracle:"))
    else:
        ids = list(tok.encode(SYSTEM + f"<|user|> {prompt}\n<|oracle|>"))
    out = []
    for _ in range(max_tokens):
        x = torch.tensor([ids[-m.max_seq_len:]], dtype=torch.long, device=DEVICE)
        logits = m(x)[0, -1, :] / temperature
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[-1]] = float("-inf")
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
        t = tok._tok.id_to_token(nxt) or ""
        if t in ("<|user|>", "<|system|>", "<eos>") or nxt == tok.eos_token_id:
            break
        ids.append(nxt)
        out.append(nxt)
        if PLAIN:
            txt = detok(tok, out)
            if "\n" in txt and txt.strip():
                return txt.split("\n")[0].replace("User:", "").strip()
    return detok(tok, out).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--probes", default="data/atl_probes.json")
    ap.add_argument("--tokenizer", default="tokenizer_chat_v2.json")
    ap.add_argument("--plain", action="store_true")
    args = ap.parse_args()
    global PLAIN
    PLAIN = args.plain
    probes = json.load(open(args.probes))
    tok = CustomTokenizerWrapper(args.tokenizer)
    m, vl = load(args.ckpt)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.out, "w") as f:
        for p in probes:
            for s in range(args.samples):
                resp = generate(m, tok, p["prompt"], seed=s)
                f.write(json.dumps({"id": p["id"], "topic": p["topic"], "novel": p.get("novel", False), "prompt": p["prompt"], "seed": s, "response": resp}) + "\n")
                n += 1
    print(f"{args.ckpt} (val_loss {vl:.4f}): {n} responses for {len(probes)} probes -> {args.out}")


if __name__ == "__main__":
    main()
