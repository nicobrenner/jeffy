"""Plan token for the v2 lineage (tags format). Each Oracle turn gets a token right after <|oracle|> naming the dialogue act of the
Oracle's OWN response (CT v3 design): *me (the Oracle talks about itself: name, size, being a model), *wh / *yn (asks back),
*hi (greeting/ack), *tell (statement). Not derivable from the prompt -> a policy decision the generator must consume.
Builds data/chat_v2ct_{train,val}.jsonl from data/chat_v2_fix_*.jsonl.     python3 ct_v2.py
"""
import json, math, re
from collections import Counter
CLASSES = ["dunno", "me", "wh", "yn", "hi", "tell"]
MARK = {c: f"*{c}" for c in CLASSES}
ME = re.compile(r"tiny oracle|\bparameters?\b|megabytes?|\bnext word\b|\b(i'?m|i am) (a|an|just a|only a) (tiny |small |little )?(model|language model|ai|program|bot|chatbot|neural)|"
                r"\bmy (training|weights|vocabulary|context window)\b|on your (device|phone|toaster|browser)|\bi (don'?t|can'?t) (have|feel|sleep|eat|see)\b|i'?m (software|code|math)", re.I)
GREET = r"^(hi|hello|hey|good (morning|evening|afternoon|night)|yo|sup|thanks|thank you|thx|bye|goodbye|see you|ok|okay|cool|nice|great|awesome|lol|haha|sure|yes|no|yeah|nope)\b"
WH = r"\b(what|who|whom|whose|where|when|why|how|which)\b"

DUNNO = re.compile(r"\b(i (don'?t|do not) know|no idea|no clue|not sure (what|how|if|about|i)|i have no|can'?t help (you )?(with|there)|beyond me|not in my training|never heard of|"
                   r"i made that up|i invented|came from nowhere|i'?m not (a|an) \w+ expert|outside my weights|not my department)\b", re.I)

def act_of(resp):
    s = resp.strip().lower()
    if DUNNO.search(s): return "dunno"
    if ME.search(s): return "me"
    if re.match(GREET, s) and len(s.split()) <= 6: return "hi"
    if "?" in s: return "wh" if re.search(WH + r"[^?]*\?", s) else "yn"
    return "tell"

SYN = {"n": 0}
PUSHBACK = ["huh? that's not what I asked", "what? I asked you something else", "you lost me. answer the question", "that makes no sense, read my message again",
            "wait, what are you talking about?", "no no, I asked about something else entirely", "are you even listening? try again", "that's not an answer",
            "you're making stuff up. do you actually know?", "ok that was random. can you just answer?", "I don't think you understood me", "huh??"]
def synthetic_dunno(copies=8):
    """the model's OWN bad first answers (recover_src) + a push-back + one of Nico's admit-ignorance lines, as *dunno examples."""
    import random, json as _j
    from atl_probe import SYSTEM
    rng = random.Random(3); lines = _j.load(open("pile_runs/atl/oracle_lines_nico_3.json"))["dunno"]
    src = [_j.loads(l) for l in open("pile_runs/atl/recover_src_grown_life8.jsonl")]
    out = []
    for _ in range(copies):
        for r in src:
            t = SYSTEM.rstrip("\n") + "\n" + f"<|user|> {r['prompt']}\n<|oracle|> {MARK[act_of(r['response'])]} {r['response']}\n<|user|> {rng.choice(PUSHBACK)}\n<|oracle|> {MARK['dunno']} {rng.choice(lines)}"
            out.append({"text": t})
    SYN["n"] = len(out); return out

def main():
    for split in ("train", "val"):
        counts = Counter(); out = []
        for l in open(f"data/chat_v2_fix_{split}.jsonl"):
            t = json.loads(l)["text"]; pieces = []; pos = 0
            for m in re.finditer(r"<\|oracle\|> (.*?)(?=\n<\|user\|>|\n<\|system\|>|$)", t, flags=re.S):
                lab = act_of(m.group(1)); counts[lab] += 1
                pieces.append(t[pos:m.start(1)] + MARK[lab] + " "); pos = m.start(1)
            pieces.append(t[pos:]); out.append({"text": "".join(pieces)})
        if split == "train":
            out += synthetic_dunno(); counts["dunno"] += SYN["n"]
        with open(f"data/chat_v2ct_{split}.jsonl", "w") as f:
            for r in out: f.write(json.dumps(r) + "\n")
        n = sum(counts.values()); H = -sum(v / n * math.log2(v / n) for v in counts.values())
        print(f"{split}: {len(out)} conversations, {n} oracle turns -> data/chat_v2ct_{split}.jsonl; classes {dict(counts)} majority {max(counts.values())/n:.3f} entropy {H:.2f} bits")
    print("sample:", out[0]["text"][:400].replace("\n", " | "))

if __name__ == "__main__":
    main()
