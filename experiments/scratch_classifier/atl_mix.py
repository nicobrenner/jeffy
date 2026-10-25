"""Adaptive Teacher Loop — build the training mix for one round and (optionally) summarise grades.

  python3 atl_mix.py summarize pile_runs/atl/grades_r0.jsonl pile_runs/atl/probe_r0.jsonl [--goal salad|wit|smart|topic:<name>]
      -> prints the worst topic x failure cells for the goal, mean scores, per-topic held-out loss is reported by train_lm separately
  python3 atl_mix.py mix pile_runs/atl/batch_r1.jsonl --base data/chat_v6_train.jsonl --replay 0.3 --out data/chat_atl_r1_train.jsonl
      -> new batch + a replay sample of the base (30% of the mix), shuffled; val stays data/chat_v6_val.jsonl
"""

import argparse
import collections
import json
import random
import statistics as st

random.seed(0)
FAIL_TAGS = ["salad", "off-persona", "memorized-line", "too-long", "mean", "wrong-deflection", "invented-fact", "unsafe", "repetition"]


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def summarize(grades_path, probe_path, goal):
    probes = {(r["id"], r["seed"]): r for r in load(probe_path)}
    grades = load(grades_path)
    for g in grades:
        g.update({k: probes[(g["id"], g["seed"])][k] for k in ("topic", "novel", "prompt", "response")})
    metric = {"salad": lambda g: g["coherence"], "wit": lambda g: g["wit"], "smart": lambda g: g["coherence"] + g["helpfulness"]}
    if goal.startswith("topic:"):
        t = goal.split(":", 1)[1]
        score = lambda g: (g["coherence"] + g["persona"] + g["helpfulness"]) if g["topic"] == t else None
    else:
        score = metric.get(goal, lambda g: g["coherence"] + g["persona"] + g["wit"] + g["helpfulness"])
    means = {k: st.mean(g[k] for g in grades) for k in ("coherence", "persona", "wit", "helpfulness")}
    novel = [g for g in grades if g["novel"]]
    print(f"{len(grades)} graded | means {means} | novel-prompt coherence {st.mean(g['coherence'] for g in novel):.2f} | "
          f"tag rates: " + ", ".join(f"{t} {sum(t in g['tags'] for g in grades) / len(grades):.0%}" for t in FAIL_TAGS if any(t in g['tags'] for g in grades)))
    by_topic = collections.defaultdict(list)
    for g in grades:
        s = score(g)
        if s is not None:
            by_topic[g["topic"]].append(s)
    worst = sorted(((st.mean(v), t) for t, v in by_topic.items()), key=lambda x: x[0])[:8]
    print(f"\nworst 8 topics for goal '{goal}':")
    for m, t in worst:
        tags = collections.Counter(tag for g in grades if g["topic"] == t for tag in g["tags"]).most_common(3)
        print(f"  {m:4.2f}  {t:<40} {tags}")
    bad = sorted(grades, key=lambda g: (score(g) if score(g) is not None else 99))[:30]
    print(f"\n{len(bad)} worst responses (for correction examples) -> stdout as JSONL below")
    for g in bad:
        print(json.dumps({"id": g["id"], "topic": g["topic"], "prompt": g["prompt"], "student": g["response"], "tags": g["tags"]}))


def to_plain(t):
    return (t.replace("<|system|> ", "System: ").replace("<|user|> ", "User: ").replace("<|oracle|> ", "Oracle: ")
             .replace("<|system|>", "System:").replace("<|user|>", "User:").replace("<|oracle|>", "Oracle:"))


def mix(batch_path, base_path, replay, out, plain=False):
    batch = load(batch_path)
    base = load(base_path)
    n_replay = int(len(batch) * replay / (1 - replay))
    conv = to_plain if plain else (lambda t: t)
    rows = [{"text": conv(r["text"])} for r in batch] + [{"text": conv(r["text"])} for r in random.sample(base, min(n_replay, len(base)))]
    random.shuffle(rows)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"mix: {len(batch)} new + {n_replay} replay = {len(rows)} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summarize"); s.add_argument("grades"); s.add_argument("probes"); s.add_argument("--goal", default="salad")
    m = sub.add_parser("mix"); m.add_argument("batch"); m.add_argument("--base", required=True); m.add_argument("--replay", type=float, default=0.3); m.add_argument("--out", required=True)
    m.add_argument("--plain", action="store_true", help="render <|tags|> as System:/User:/Oracle: (TinyStories-tokenizer bases)")
    a = ap.parse_args()
    if a.cmd == "summarize":
        summarize(a.grades, a.probes, a.goal)
    else:
        mix(a.batch, a.base, a.replay, a.out, a.plain)


if __name__ == "__main__":
    main()
