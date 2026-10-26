"""chat_v2 data fix for the v2 lineage (tag format `<|user|> ... <|oracle|> ...`).
1. Fill PersonaChat placeholders ("[user 2 name]", "[user 1]", "[name]", ...) with one random first name per conversation
   (they are the USER's name when the user says them or the Oracle addresses the user).
2. Drop conversations where the Oracle introduces ITSELF with a placeholder ("My name is [user 2 name]") or claims a human
   identity (job, spouse/kids, age, pets by name) — these teach the Oracle to be a person.
Writes data/chat_v2_fix_{train,val}.jsonl and prints counts.   python3 fix_chat_v2.py
"""
import json, random, re
PH = re.compile(r"\[(?:user ?\d(?: name)?|name|username|your name|user name)\]|\((?:name|username)\)", re.I)
SELF = re.compile(r"\b(I'?m|I am|my name'?s|my name is|name'?s|call me|this is)\s*(\[|\()", re.I)
HUMAN = re.compile(r"\b(I work (as|at|in|for)\b|I'?m (a|an) (teacher|nurse|doctor|lawyer|student|programmer|writer|chef|engineer|designer|mom|dad|mother|father|"
                   r"farmer|waiter|waitress|cashier|artist|musician|dentist|pilot|cop|police|soldier|salesman|accountant|mechanic|plumber|carpenter|librarian|professor|"
                   r"retired|college student|single mom|single dad|stay at home)|my (wife|husband|kids|children|son|daughter|boyfriend|girlfriend|grandkids|grandchildren)\b|"
                   r"I'?m (married|divorced|single|\d\d years old|pregnant)|I have (a|an|two|three|four|\d) (kids|children|sons|daughters|cats?|dogs?) (named|called)|"
                   r"I'?m (\d\d|\d) years old|my job\b|at work today|I drive a|I live in (a|an) (apartment|house|farm))", re.I)
NAMES = ["Maya","Leo","Sofia","Omar","Elena","Jonas","Priya","Mateo","Hana","Noah","Aisha","Luca","Mei","Diego","Nora","Sam","Yara","Tomás","Ana","Kai",
         "Zoe","Ravi","Lena","Max","Ines","Ali","Clara","Ben","Rosa","Theo"]

def fix(text, rng):
    oracle_lines = [l for l in text.split("\n") if l.startswith("<|oracle|>")]
    for l in oracle_lines:
        if SELF.search(l): return None, "self-placeholder"
        if HUMAN.search(l): return None, "human-identity"
    if not PH.search(text): return text, None
    return PH.sub(rng.choice(NAMES), text), "filled"

for split in ("train", "val"):
    rng = random.Random(2026); kept, drop, filled = [], {}, 0
    for l in open(f"data/chat_v2_{split}.jsonl"):
        r = json.loads(l); t, why = fix(r["text"], rng)
        if t is None: drop[why] = drop.get(why, 0) + 1; continue
        if why == "filled": filled += 1
        r["text"] = t; kept.append(r)
    with open(f"data/chat_v2_fix_{split}.jsonl", "w") as f:
        for r in kept: f.write(json.dumps(r) + "\n")
    hand = sum("boiling opinions" in r["text"] for r in kept)
    print(f"{split}: kept {len(kept):,}  dropped {drop}  filled {filled:,}  remaining placeholders {sum(bool(PH.search(r['text'])) for r in kept)}  hand-written copies {hand}")
