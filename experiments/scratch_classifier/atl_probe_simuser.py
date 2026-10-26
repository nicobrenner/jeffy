"""Multi-turn probe with a REACTIVE simulated user, run in stages so an LLM can write the user's turns in response to what the
model actually said (Nico: scripted follow-ups made both sides read like Eliza).
  stage 1:  python3 atl_probe_simuser.py CKPT --tokenizer T [--plain] --out F.jsonl --turn 1            # answers the openers
  (an agent reads F.jsonl, writes followups_2.json = {"<id>": "<user turn 2>", ...} as a real, slightly impatient person)
  stage 2:  python3 atl_probe_simuser.py CKPT ... --out F.jsonl --turn 2 --followups followups_2.json  # appends turn-2 rows
  stage 3:  same with --turn 3 --followups followups_3.json
Rows: id, turn, history [{user, oracle}], prompt, response. Sampler recipe as atl_probe_multiturn.
"""
import argparse, json, random
from pathlib import Path
from atl_probe import load, SYSTEM
from atl_probe_multiturn import generate_ctx
from train_lm import CustomTokenizerWrapper

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("--out", required=True); ap.add_argument("--probes", default="data/atl_probes.json")
    ap.add_argument("--tokenizer", default="tokenizer_chat_v2.json"); ap.add_argument("--plain", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.5); ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--rep_penalty", type=float, default=1.3); ap.add_argument("--no_repeat_ngram", type=int, default=3)
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--turn", type=int, required=True); ap.add_argument("--followups", default=None)
    args = ap.parse_args()
    tok = CustomTokenizerWrapper(args.tokenizer); m, _ = load(args.ckpt)
    probes = json.load(open(args.probes)); probes = [p for p in probes if p.get("novel")] + [p for p in probes if not p.get("novel")]
    probes = probes[:args.n]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    prev = {}
    if args.turn > 1:
        for r in map(json.loads, open(args.out)):
            prev.setdefault(r["id"], []).append(r)
        fu = json.load(open(args.followups))
    with open(args.out, "a" if args.turn > 1 else "w") as f:
        for p in probes:
            hist = sorted(prev.get(p["id"], []), key=lambda r: r["turn"]) if args.turn > 1 else []
            if args.turn > 1 and (len(hist) != args.turn - 1 or p["id"] not in fu): continue
            transcript = [{"user": r["prompt"], "oracle": r["response"]} for r in hist]
            user = p["prompt"] if args.turn == 1 else fu[p["id"]]
            ctx = SYSTEM.replace("<|system|> ", "System: ") if args.plain else SYSTEM
            for t in transcript:
                ctx += (f"User: {t['user']}\nOracle: {t['oracle']}\n" if args.plain else f"<|user|> {t['user']}\n<|oracle|> {t['oracle']}\n")
            ctx += f"User: {user}\nOracle:" if args.plain else f"<|user|> {user}\n<|oracle|>"
            resp = generate_ctx(m, tok, tok.encode(ctx), seed=args.turn, args=args)
            f.write(json.dumps({"id": p["id"], "topic": p["topic"], "novel": p.get("novel", False), "turn": args.turn, "history": transcript,
                                "prompt": user, "response": resp}) + "\n")
    print(f"{args.ckpt}: turn {args.turn} written -> {args.out}")

if __name__ == "__main__":
    main()
