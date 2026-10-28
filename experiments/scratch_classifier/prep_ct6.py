"""Fixer pass ct6: like prep_ct5.py but the 3,294 train / 183 val rows whose system line the model recites are KEPT, with that
line replaced by a draw from the corpus's own clean system-prompt bank (42 lines, weighted as they occur); plus the 700
room-event + visit-memory rows. Same 8-epoch ct2 recipe (see prep_ct5.py docstring, --tag oracle_v2_grown_ct6)."""
import json, random
from collections import Counter
PH = "make up something entertaining instead of admitting ignorance"   # offline corpus cleaning, not a harness rule
rng = random.Random(6)
bank = Counter(json.loads(l)["text"].split("\n", 1)[0] for l in open("data/chat_v2ct_train.jsonl") if PH not in l)
lines, weights = zip(*bank.items())
for split in ("train", "val"):
    out = []; n_var = 0
    for l in open(f"data/chat_v2ct_{split}.jsonl"):
        r = json.loads(l)
        if PH in r["text"]:
            head, rest = r["text"].split("\n", 1); r["text"] = rng.choices(lines, weights)[0] + "\n" + rest; n_var += 1
        out.append(json.dumps(r) + "\n")
    extra = list(open("data/chat_v2room.jsonl")) if split == "train" else []
    out += extra; rng.shuffle(out); open(f"data/chat_v2ct6_{split}.jsonl", "w").writelines(out)
    print(f"{split}: {len(out) - len(extra)} rows kept, {n_var} system lines varied from a bank of {len(lines)}, + {len(extra)} room/visit rows = {len(out)}")
