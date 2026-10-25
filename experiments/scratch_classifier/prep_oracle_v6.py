"""Build the Tiny Oracle v6 dataset: generated on-persona data + hand-written batches + a slice of generic filler,
deduplicated, with validation held out BY TOPIC so val_ppl cannot be inflated by paraphrases of training lines
(the v4 leak).

Inputs : data/oracle_v3_synthetic.jsonl (from generate_oracle_data.py, has "topic"), data/oracle_v3_batch*.jsonl,
         data/chat_v2_train.jsonl (generic PersonaChat-style filler, sampled)
Outputs: data/chat_v6_train.jsonl, data/chat_v6_val.jsonl, data/chat_v6_stats.json
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

random.seed(0)
DATA = Path("data")
FILLER_FRACTION = 0.20      # share of generic conversations kept for conversational glue
HELDOUT_TOPIC_FRACTION = 0.12
RANDOM_VAL_FRACTION = 0.04
PERSONA = re.compile(r"parameter|megabyte|tiny oracle|neural net|token|browser|datacenter|4 ?mb|2 million|floppy|toaster|weights|neurons", re.I)


def oracle_turns(text):
    return [t.strip() for t in re.findall(r"<\|oracle\|>(.*?)(?=<\|user\||<\|system\||<eos>|$)", text, flags=re.S)]


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower())[:80]


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def main():
    synthetic = load(DATA / "oracle_v3_synthetic.jsonl") if (DATA / "oracle_v3_synthetic.jsonl").exists() else []
    synthetic += [x for p in sorted((DATA / "oracle_gen").glob("*.jsonl")) for x in load(p)]  # in-session generated waves
    batches = [x for p in sorted(DATA.glob("oracle_v3_batch*.jsonl")) for x in load(p)]
    for x in batches:
        x.setdefault("topic", "handwritten")
    generic = load(DATA / "chat_v2_train.jsonl")
    print(f"synthetic {len(synthetic):,} | handwritten {len(batches):,} | generic pool {len(generic):,}")

    # dedupe on-persona conversations by their normalised oracle turns
    seen, persona = set(), []
    for x in synthetic + batches:
        key = tuple(norm(t) for t in oracle_turns(x["text"]))
        if not key or key in seen:
            continue
        seen.add(key)
        persona.append(x)
    print(f"on-persona after dedupe: {len(persona):,} conversations, {len(seen):,} unique turn-sets")

    # generic filler: conversations with NO persona markers, sampled to FILLER_FRACTION of the final set
    pool = [x for x in generic if not PERSONA.search(x["text"])]
    n_filler = int(len(persona) * FILLER_FRACTION / (1 - FILLER_FRACTION))
    filler = random.sample(pool, min(n_filler, len(pool)))
    for x in filler:
        x["topic"] = "generic"

    # validation: whole held-out topics + a small random slice of the rest
    topics = sorted({x["topic"] for x in persona if x["topic"] not in ("handwritten", "restyled")})
    random.shuffle(topics)
    heldout = set(topics[: max(1, int(len(topics) * HELDOUT_TOPIC_FRACTION))])
    train, val = [], []
    for x in persona + filler:
        if x["topic"] in heldout or random.random() < RANDOM_VAL_FRACTION:
            val.append(x)
        else:
            train.append(x)
    random.shuffle(train)
    random.shuffle(val)

    for name, rows in (("chat_v6_train.jsonl", train), ("chat_v6_val.jsonl", val)):
        with open(DATA / name, "w") as f:
            for x in rows:
                f.write(json.dumps({"text": x["text"]}) + "\n")
    uniq_train = len({norm(t) for x in train for t in oracle_turns(x["text"])})
    stats = dict(train=len(train), val=len(val), heldout_topics=sorted(heldout), unique_oracle_turns_train=uniq_train,
                 topic_counts=Counter(x["topic"] for x in train).most_common(12))
    (DATA / "chat_v6_stats.json").write_text(json.dumps(stats, indent=1))
    print(f"train {len(train):,} | val {len(val):,} (held-out topics: {len(heldout)}) | unique oracle turns in train: {uniq_train:,}")
    print(f"wrote data/chat_v6_train.jsonl, data/chat_v6_val.jsonl, data/chat_v6_stats.json")


if __name__ == "__main__":
    main()
