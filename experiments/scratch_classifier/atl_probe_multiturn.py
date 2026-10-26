"""Multi-turn probe: 3-turn scripted conversations (the regime the browser demo runs in, which no blind pass has covered).
The user's first turn comes from the novel-prompt set, the follow-ups are generic reactions that force the model to stay on
its own previous answer. Each model answers turn 1, then turn 2 with its own turn-1 answer in context, then turn 3.
Writes one row per (conversation, turn) with the full transcript so far, so a grader can judge the turn in context.

  python3 atl_probe_multiturn.py checkpoints/best_lm_oracle_v7a_conv.pt --plain --tokenizer tokenizer_tinystories.json \
      --temperature 0.5 --no_repeat_ngram 3 --rep_penalty 1.3 --out pile_runs/atl/probe_mt_conv.jsonl
"""
import argparse
import json
import random
from pathlib import Path

import torch

import atl_probe
from atl_probe import load, detok, SYSTEM, DEVICE
from train_lm import CustomTokenizerWrapper

FOLLOWUPS = [["is that your opinion?", "why do you think that?"], ["really? tell me more", "and what else?"],
             ["what do you mean by that?", "ok but what would you do?"], ["haha ok. do you like that?", "what else do you like?"],
             ["I don't believe you", "prove it"], ["that's interesting. how do you know?", "what are you anyway?"]]


@torch.no_grad()
def generate_ctx(m, tok, ctx_ids, seed, args):
    torch.manual_seed(seed)
    ids = list(ctx_ids); out = []
    for _ in range(60):
        x = torch.tensor([ids[-m.max_seq_len:]], dtype=torch.long, device=DEVICE)
        logits = m(x)[0, -1, :]
        if args.rep_penalty != 1.0 and out:
            prev = torch.tensor(sorted(set(out)), device=logits.device); lp = logits[prev]
            logits[prev] = torch.where(lp > 0, lp / args.rep_penalty, lp * args.rep_penalty)
        if args.no_repeat_ngram and len(out) >= args.no_repeat_ngram - 1:
            n = args.no_repeat_ngram; tail = tuple(out[len(out) - (n - 1):])
            for b in {out[i + n - 1] for i in range(len(out) - n + 1) if tuple(out[i:i + n - 1]) == tail}:
                logits[b] = float("-inf")
        logits[1] = float("-inf")
        logits = logits / args.temperature
        v, _ = torch.topk(logits, args.top_k); logits[logits < v[-1]] = float("-inf")
        nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
        t = tok._tok.id_to_token(nxt) or ""
        if t in ("<|user|>", "<|system|>", "<eos>") or nxt == tok.eos_token_id:
            break
        ids.append(nxt); out.append(nxt)
        if args.plain:
            txt = detok(tok, out)
            if "\n" in txt and txt.strip():
                return txt.split("\n")[0].strip()
    return detok(tok, out).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("--out", required=True); ap.add_argument("--probes", default="data/atl_probes.json")
    ap.add_argument("--tokenizer", default="tokenizer_chat_v2.json"); ap.add_argument("--plain", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.8); ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--rep_penalty", type=float, default=1.0); ap.add_argument("--no_repeat_ngram", type=int, default=0)
    ap.add_argument("--n", type=int, default=40, help="conversations (novel prompts first, then topic prompts)")
    args = ap.parse_args()
    probes = json.load(open(args.probes))
    probes = [p for p in probes if p.get("novel")] + [p for p in probes if not p.get("novel")]
    probes = probes[:args.n]
    tok = CustomTokenizerWrapper(args.tokenizer); m, _ = load(args.ckpt)
    rng = random.Random(7)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for p in probes:
            fu = rng.choice(FOLLOWUPS)
            transcript = []
            if args.plain:
                ctx = SYSTEM.replace("<|system|> ", "System: ")
            else:
                ctx = SYSTEM
            for turn, user in enumerate([p["prompt"]] + fu, 1):
                if args.plain:
                    ctx += f"User: {user}\nOracle:"
                else:
                    ctx += f"<|user|> {user}\n<|oracle|>"
                resp = generate_ctx(m, tok, tok.encode(ctx), seed=turn, args=args)
                transcript.append({"user": user, "oracle": resp})
                ctx += (" " + resp + "\n") if args.plain else (" " + resp + "\n")
                f.write(json.dumps({"id": p["id"], "topic": p["topic"], "novel": p.get("novel", False), "turn": turn,
                                    "history": transcript[:-1], "prompt": user, "response": resp}) + "\n")
    print(f"{args.ckpt}: {len(probes)} conversations x 3 turns -> {args.out}")


if __name__ == "__main__":
    main()
