"""Turn demo teach-mode corrections (corrections.jsonl exported from demo/index.html) into training conversations.
Each correction = {model, prime, history:[{role,text}], user, original, corrected}. Output: tags-format conversations
(system prompt + history + user + CORRECTED oracle reply), chat style (no final period), repeated --copies times so a handful
of Nico's rewrites carry weight next to the 22K-conversation corpus (same trick as the 110 hand-written examples × 50).

  python3 corrections_to_data.py corrections.jsonl --out data/chat_v2_teach_train.jsonl --copies 20
"""
import argparse, json, re, random
from atl_probe import SYSTEM

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("--out", required=True); ap.add_argument("--copies", type=int, default=20)
    a = ap.parse_args(); rows = [json.loads(l) for l in open(a.src) if l.strip()]
    convs = []
    for r in rows:
        if "plain" in r.get("model", "") or "persona2" in r.get("model", "") or "v7a" in r.get("model", ""): continue   # tags-format (v2 lineage) only
        t = SYSTEM.rstrip("\n") + "\n"
        for m in r.get("history", []): t += ("<|user|> " if m["role"] == "user" else "<|oracle|> ") + m["text"].strip() + "\n"
        corrected = re.sub(r"(?<![.!?])\.\s*$", "", r["corrected"].strip())
        t += "<|user|> " + r["user"].strip() + "\n<|oracle|> " + corrected
        convs.append({"text": t, "source": "teach", "ts": r.get("ts")})
    random.Random(0).shuffle(convs)
    with open(a.out, "w") as f:
        for _ in range(a.copies):
            for c in convs: f.write(json.dumps(c) + "\n")
    print(f"{len(rows)} corrections -> {len(convs)} conversations x {a.copies} copies -> {a.out}")

if __name__ == "__main__":
    main()
