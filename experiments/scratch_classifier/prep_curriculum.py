"""Induction curriculum for the copy-circuit scout (2026-10-28): synthetic rows mixed into the scout corpus so the copy circuit has a
reason to form. Two kinds, both with the copy target inside the Oracle turn (oracle-only loss): (1) a span of REAL chat tokens of
period 3-12 repeated to ~100 tokens; (2) echo rows "<|user|> the user said X … <|oracle|> X" with X a real 3-12-token chat phrase.
  python3 prep_curriculum.py --share 0.3 --out data/chat_v2curr30_train.jsonl   (share = fraction of windows that are synthetic)"""
import argparse, json, random, re
from atl_probe import SYSTEM
from train_lm import CustomTokenizerWrapper
def build(share, seed=0):
    rng = random.Random(seed); tok = CustomTokenizerWrapper("tokenizer_chat_v2.json")
    base = list(open("data/chat_v2scout_train.jsonl")); texts = [json.loads(l)["text"] for l in base]
    turns = [m.group(1) for t in texts[:8000] for m in re.finditer(r"<\|oracle\|> \*\w+ (.*?)(?=\n<\|user\|>|$)", t, flags=re.S)]   # offline data building
    n_syn = int(len(base) * 1.3 * share / (1 - share)); rows = []   # ~1.3 windows per corpus row
    for i in range(n_syn):
        src = rng.choice(turns); ids = tok.encode(src); p = rng.randint(3, 12)
        if len(ids) < p + 1: continue
        a = rng.randint(0, len(ids) - p); span = ids[a:a + p]
        if rng.random() < 0.5:   # repeated span: period p, ~100 tokens, inside the Oracle turn
            rep = tok.decode(span * (100 // p + 1))
            rows.append({"text": SYSTEM.rstrip("\n") + "\n<|user|> " + rng.choice(["say it again", "again", "keep going", "repeat that", "and then"]) + "\n<|oracle|> *tell " + rep})
        else:   # echo: the Oracle copies the user's phrase back
            x = tok.decode(span).strip()
            rows.append({"text": SYSTEM.rstrip("\n") + "\n<|user|> " + rng.choice(["the user said {x}", "i said {x}", "remember this: {x}", "repeat after me, {x}", "{x}, got it?"]).format(x=x) + "\n<|oracle|> *tell " + rng.choice(["{x}", "You said {x}", "{x}. Noted", "Got it, {x}"]).format(x=x)})
    out = base + [json.dumps(r) + "\n" for r in rows]; rng.shuffle(out); return out, len(rows)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--share", type=float, required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    out, n = build(a.share); open(a.out, "w").writelines(out); open(a.out.replace("_train", "_val"), "w").writelines(out[:500])
    print(f"share {a.share}: {n} synthetic rows + corpus -> {a.out} ({len(out)} rows)"); print([json.loads(l)["text"][-160:].replace("\n", " | ") for l in out[:40] if "say it again" in l or "the user said" in l or "repeat after me" in l][:2])
