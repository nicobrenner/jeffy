"""Build one shuffled blind set from N probe files (M1..MN labels assigned at random), seal the key. Usage:
  python3 build_blind_set.py OUT_PREFIX tag1=probe1.jsonl tag2=probe2.jsonl ...
Writes OUT_PREFIX.jsonl (shuffled rows with model=M#) and OUT_PREFIX_key.json ({M#: tag}).
"""
import json, random, sys
from pathlib import Path

out = sys.argv[1]
arms = [a.split("=", 1) for a in sys.argv[2:]]
random.seed(20261025)
labels = [f"M{i+1}" for i in range(len(arms))]
random.shuffle(labels)
key = {}
rows = []
for lab, (tag, path) in zip(labels, arms):
    key[lab] = tag
    for l in open(path):
        r = json.loads(l)
        rows.append({"model": lab, "id": r["id"], "topic": r["topic"], "novel": r["novel"], "prompt": r["prompt"], "seed": r["seed"], "response": r["response"]})
random.shuffle(rows)
with open(out + ".jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
json.dump(dict(sorted(key.items())), open(out + "_key.json", "w"))
print(f"{len(rows)} rows -> {out}.jsonl ; key sealed in {out}_key.json")
