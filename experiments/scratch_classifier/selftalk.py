"""Yin/Yang self-talk harness: two checkpoints talk to each other. Each model sees ITSELF as <|oracle|> and the other as <|user|>,
so no model ever leaves its trained role. Openers come from the probe pool so topics vary. Every trace gets automatic flags
(placeholder, human-identity, repetition, runaway length) so the curation pipeline (auto → grader → Nico) can start from here.

  python3 selftalk.py --a checkpoints/best_lm_oracle_v2_grown_life8.pt --b checkpoints/best_lm_oracle_v2_grown_life8.pt \
      --n 50 --turns 6 --out pile_runs/selftalk/round0.jsonl
Options: --sys-a/--sys-b override a side's system prompt (e.g. a "person" role for the Oracle-and-person variant); --seed.
"""
import argparse, json, random, re
from pathlib import Path
import torch
from atl_probe import load, SYSTEM
from atl_probe_multiturn import generate_ctx
from train_lm import CustomTokenizerWrapper

HUMAN = re.compile(r"\b(I work (as|at|in|for)|(I'?m|I am) (a|an) (teacher|nurse|doctor|lawyer|student|programmer|writer|chef|engineer|designer|mom|dad|mother|father|farmer|waiter|waitress|cashier|artist|musician|dentist|pilot|cop|police|soldier|salesman|accountant|mechanic|plumber|carpenter|librarian|professor|retired|college student|stay-?at-?home)|my (wife|husband|kids|children|son|daughter|boyfriend|girlfriend|sister|brother|siblings?|parents|mom|dad|job|boss)|I have (a|an|two|three|\d|an older|a younger) (sisters?|brothers?|siblings?|kids?|children|sons?|daughters?|cats?|dogs?)|years old|I'?m (married|single|divorced))\b", re.I)
PH = re.compile(r"\[user|\[name|<unk>")
NAME = re.compile(r"\b(?:[Mm]y name is|[Mm]y name's|[Cc]all me)\s+([A-Z][a-z]+)")

def ngrams(s, n=3):
    w = s.lower().split(); return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}

def flags(turns):
    f = set(); seen = {"A": set(), "B": set()}
    for t in turns:
        x = t["text"]
        if PH.search(x): f.add("placeholder")
        if HUMAN.search(x): f.add("human-identity")
        m = NAME.search(x)
        if m and m.group(1) not in ("Tiny",): f.add("human-name")
        g = ngrams(x)
        if g and len(g & seen[t["speaker"]]) / len(g) > 0.5: f.add("repetition")
        seen[t["speaker"]] |= g
        if len(x.split()) > 60 or len(x.split()) == 0: f.add("length")
    return sorted(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--tokenizer", default="tokenizer_chat_v2.json")
    ap.add_argument("--sys-a", default=None); ap.add_argument("--sys-b", default=None)
    ap.add_argument("--probes", default="data/atl_probes.json"); ap.add_argument("--n", type=int, default=50); ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.5); ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--rep_penalty", type=float, default=1.3); ap.add_argument("--no_repeat_ngram", type=int, default=3); ap.add_argument("--plain", action="store_true")
    args = ap.parse_args()
    tok = CustomTokenizerWrapper(args.tokenizer); A, _ = load(args.a); B, _ = load(args.b) if args.b != args.a else (A, None)
    models = {"A": A, "B": B}; systems = {"A": args.sys_a or SYSTEM, "B": args.sys_b or SYSTEM}
    probes = json.load(open(args.probes)); rng = random.Random(args.seed); openers = rng.sample(probes, min(args.n, len(probes)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True); kept = 0
    with open(args.out, "w") as f:
        for ci, p in enumerate(openers):
            turns = [{"speaker": "A", "text": p["prompt"]}]   # A "says" the opener (a human-written prompt) to start; B answers first
            for ti in range(args.turns):
                spk = "B" if ti % 2 == 0 else "A"
                ctx = systems[spk]
                for t in turns:
                    ctx += (f"<|oracle|> {t['text']}\n" if t["speaker"] == spk else f"<|user|> {t['text']}\n")
                ctx += "<|oracle|>"
                resp = generate_ctx(models[spk], tok, tok.encode(ctx), seed=args.seed * 100000 + ci * 100 + ti, args=args)
                turns.append({"speaker": spk, "text": resp})
                if not resp.strip(): break
            fl = flags(turns[1:]); kept += not fl
            f.write(json.dumps({"id": f"st{args.seed}_{ci:03d}", "opener": p["prompt"], "topic": p.get("topic"), "turns": turns, "flags": fl, "pass_auto": not fl}) + "\n")
    print(f"{len(openers)} conversations x {args.turns} turns -> {args.out}; pass automatic filter: {kept}/{len(openers)}")

if __name__ == "__main__":
    main()
