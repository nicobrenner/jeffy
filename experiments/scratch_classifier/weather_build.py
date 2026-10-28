"""Weather rows, room_build.py style: the system line carries ONE weather seed line (weather_bank.py), the Oracle's first turn
riffs on it (seed + continuation) or answers the user's reaction to it. Teaches the pre-LN line to read the weather line.
  python3 weather_build.py --n 700 --out data/chat_v2weather.jsonl"""
import argparse, json, random
from atl_probe import SYSTEM
from ct_v2 import act_of, MARK
from weather_bank import BANK, CONDITIONS
OPENERS = [("hey", "Hi. You are in my room now, mind the crawler"), ("hi there", "Hello. 2 million parameters, one chat, a pet with six legs. Welcome"), ("what's up", "Nothing. I don't exist between your messages. It's very restful"),
           ("hello?", "Yes, here. Small, but here"), ("yo", "Yo. Say something, the crawler is waiting for a word"), ("anyone there", "Me. Just me and the thing with the legs"), ("hi oracle", "Hi. Make yourself comfortable, there is nowhere to sit"),
           ("how's it going", "Going. The weather in here says it all"), ("what's the weather like in there", "Read the forecast, I just live in it")]
def oracle(t): return f"<|oracle|> {MARK[act_of(t)]} {t}"
def build(n, seed=0):
    rng = random.Random(seed); out = []
    for _ in range(n):
        c = rng.choice(CONDITIONS); b = BANK[c]; w = rng.choice(b["seeds"])
        lines = []; u, o = rng.choice(OPENERS)
        kind = rng.choice(["riff", "riff", "react", "ask"])
        if kind == "riff":   # the Oracle brings the weather up itself: seed + riff
            lines += [f"<|user|> {u}", f"<|oracle|> {MARK['tell']} {w}. {rng.choice(b['riffs'])}"]
            if rng.random() < 0.5: ru, ro = rng.choice(b["reactions"]); lines += [f"<|user|> {ru}", oracle(ro)]
        elif kind == "react":   # the user reacts to the weather line directly
            ru, ro = rng.choice(b["reactions"]); lines += [f"<|user|> {u}", oracle(o), f"<|user|> {ru}", oracle(ro)]
        else:   # the user asks about the room / weather; the Oracle answers from the line
            q = rng.choice(["what's the weather like in there", "how's the room today", "what's it like in there right now", "anything going on in there", "how are things in the tab"])
            lines += [f"<|user|> {q}", f"<|oracle|> {MARK['tell']} {w}. {rng.choice(b['riffs'])}"]
        out.append({"text": SYSTEM.rstrip("\n") + " " + w + ".\n" + "\n".join(lines), "condition": c})
    return out
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=700); ap.add_argument("--out", default="data/chat_v2weather.jsonl"); a = ap.parse_args()
    rows = build(a.n)
    with open(a.out, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} weather rows -> {a.out}"); print(rows[0]["text"].replace("\n", " | ")); print(rows[3]["text"].replace("\n", " | "))
