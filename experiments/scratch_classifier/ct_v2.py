"""Plan token for the v2 lineage (tags format). Each Oracle turn gets a token right after <|oracle|> naming the dialogue act of the
Oracle's OWN response (CT v3 design): *me (the Oracle talks about itself: name, size, being a model), *wh / *yn (asks back),
*hi (greeting/ack), *tell (statement). Not derivable from the prompt -> a policy decision the generator must consume.
Builds data/chat_v2ct_{train,val}.jsonl from data/chat_v2_fix_*.jsonl.     python3 ct_v2.py
"""
import json, math, re
from collections import Counter
CLASSES = ["me", "wh", "yn", "hi", "tell"]
MARK = {c: f"*{c}" for c in CLASSES}
ME = re.compile(r"tiny oracle|\bparameters?\b|megabytes?|\bnext word\b|\b(i'?m|i am) (a|an|just a|only a) (tiny |small |little )?(model|language model|ai|program|bot|chatbot|neural)|"
                r"\bmy (training|weights|vocabulary|context window)\b|on your (device|phone|toaster|browser)|\bi (don'?t|can'?t) (have|feel|sleep|eat|see)\b|i'?m (software|code|math)", re.I)
GREET = r"^(hi|hello|hey|good (morning|evening|afternoon|night)|yo|sup|thanks|thank you|thx|bye|goodbye|see you|ok|okay|cool|nice|great|awesome|lol|haha|sure|yes|no|yeah|nope)\b"
WH = r"\b(what|who|whom|whose|where|when|why|how|which)\b"

def act_of(resp):
    s = resp.strip().lower()
    if ME.search(s): return "me"
    if re.match(GREET, s) and len(s.split()) <= 6: return "hi"
    if "?" in s: return "wh" if re.search(WH + r"[^?]*\?", s) else "yn"
    return "tell"

def main():
    for split in ("train", "val"):
        counts = Counter(); out = []
        for l in open(f"data/chat_v2_fix_{split}.jsonl"):
            t = json.loads(l)["text"]; pieces = []; pos = 0
            for m in re.finditer(r"<\|oracle\|> (.*?)(?=\n<\|user\|>|\n<\|system\|>|$)", t, flags=re.S):
                lab = act_of(m.group(1)); counts[lab] += 1
                pieces.append(t[pos:m.start(1)] + MARK[lab] + " "); pos = m.start(1)
            pieces.append(t[pos:]); out.append({"text": "".join(pieces)})
        with open(f"data/chat_v2ct_{split}.jsonl", "w") as f:
            for r in out: f.write(json.dumps(r) + "\n")
        n = sum(counts.values()); H = -sum(v / n * math.log2(v / n) for v in counts.values())
        print(f"{split}: {len(out)} conversations, {n} oracle turns -> data/chat_v2ct_{split}.jsonl; classes {dict(counts)} majority {max(counts.values())/n:.3f} entropy {H:.2f} bits")
    print("sample:", out[0]["text"][:400].replace("\n", " | "))

if __name__ == "__main__":
    main()
