"""Contrast rows: after a recap about the USER's life, the user asks about the Oracle's OWN job / pet / home / family / travel /
age, and the Oracle answers from the world bible. Teaches: user facts in the context are not the Oracle's facts (v5 drift:
"I'm a software engineer", "I'm a pilot" after recap rows).   python3 contrast_build.py --out data/chat_v2contrast.jsonl"""
import argparse, glob, json, random, re
from atl_probe import SYSTEM
from ct_v2 import act_of, MARK
ASK = {"work": ["what about you, what do you do?", "and your job?", "what do you do all day?"], "pet": ["do you have a pet?", "any pets on your side?", "what about you, any animals?"],
       "home": ["where do you live?", "what's your place like?", "and you, where are you?"], "family": ["do you have family?", "any siblings?", "what about your family?"],
       "travel": ["have you been anywhere?", "where are you from?", "do you travel?"], "age": ["how old are you?", "what's your age?", "and you, how old?"]}
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/chat_v2contrast.jsonl"); ap.add_argument("--per", type=int, default=3); a = ap.parse_args(); random.seed(5)
    bible = {g["key"]: [l.rstrip(".") for l in g["lines"]] for g in json.load(open("demo/world_bible.json"))["groups"]}
    src = {json.loads(l)["id"]: json.loads(l) for l in open("pile_runs/atl/multisession_src.jsonl")}
    recaps = [json.loads(l) for f in sorted(glob.glob("pile_runs/atl/multisession_out_part*.jsonl")) for l in open(f)]
    n = 0
    with open(a.out, "w") as f:
        for r in recaps:
            if r["id"] not in src: continue
            b = re.findall(r"^(user|oracle): (.*)$", src[r["id"]]["conv_b_open"], flags=re.M)[:2]
            for key in random.sample(list(ASK), a.per):
                t = SYSTEM.rstrip("\n") + " " + r["recap"].rstrip(".") + ".\n"
                for role, txt in b: t += f"<|user|> {txt}\n" if role == "user" else f"<|oracle|> {MARK[act_of(txt)]} {txt}\n"
                ans = random.choice(bible[key]); t += f"<|user|> {random.choice(ASK[key])}\n<|oracle|> {MARK[act_of(ans)]} {ans}"
                f.write(json.dumps({"text": t}) + "\n"); n += 1
    print(f"{n} contrast rows -> {a.out}"); print(t.replace("\n", " | ")[-260:])
