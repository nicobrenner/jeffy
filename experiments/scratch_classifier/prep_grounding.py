"""Grounding rows for the pre-LN line (roadmap 1b, 2026-10-27): training rows whose system line carries ONE world-table line and whose
Oracle turn uses it, built from typed facts with varied question / answer phrasings (offline templating). The 26 probe items in
pile_runs/atl/grounding_probe.json and the table lines they need are HELD OUT: none appear here. Output appended to the v5b corpus.
  python3 prep_grounding.py   -> data/chat_v2v5g_train.jsonl (= chat_v2v5b_train + grounding rows), data/chat_v2v5g_val.jsonl"""
import json, random
from atl_probe import SYSTEM
from ct_v2 import MARK
rng = random.Random(11)
# typed facts (training only; probe facts excluded on purpose: feathers/steel, bricks/cotton, 7 vs 12, 3 apples, tue->wed, ice cold, cat fly, mouse/elephant,
# sun->mon, fire hot, 10-3, 2+2, week days, mile/km, fish walk, 100 vs 99, snow, half of ten, thu->fri, gram/kg, whale/shark, december, blue+yellow, triangle, water wet, yesterday/tomorrow)
CMP = [("a tonne", "a bag of sugar", "heavier"), ("a horse", "a cat", "bigger"), ("a car", "an ant", "bigger"), ("a truck", "a bicycle", "heavier"),
       ("a mountain", "a hill", "taller"), ("an ocean", "a pond", "bigger"), ("a giraffe", "a goat", "taller"), ("a brick", "a feather", "heavier"), ("a bus", "a scooter", "longer"),
       ("a litre", "a spoonful", "more"), ("a year", "a month", "longer"), ("an hour", "a minute", "longer"), ("a lion", "a rabbit", "bigger"), ("a river", "a stream", "wider"),
       ("fifty", "twenty", "bigger"), ("nine", "four", "bigger"), ("a thousand", "a hundred", "bigger"), ("sixty", "sixteen", "bigger"), ("eight", "one", "bigger")]
SAME = [("a kilo of sand", "a kilo of sugar", "weigh the same, one kilo each"), ("a litre of milk", "a litre of water", "take up the same space, one litre"), ("a dozen eggs", "a dozen apples", "are the same count, twelve")]
PROP = [("boiling water", "hot"), ("the sun", "hot"), ("a freezer", "cold"), ("soup from the stove", "hot"), ("the sea in winter", "cold"), ("a desert at noon", "hot"), ("a lemon", "sour"), ("sugar", "sweet"),
        ("sand", "dry"), ("rain", "wet"), ("coal", "black"), ("milk", "white"), ("a lime", "green"), ("the sky by day", "blue"), ("a stone", "hard"), ("a pillow", "soft")]
SEQ = [("Monday", "Sunday"), ("Thursday", "Wednesday"), ("Saturday", "Friday"), ("February", "January"), ("April", "March"), ("July", "June"), ("October", "September"), ("afternoon", "morning"), ("evening", "afternoon"), ("night", "evening")]
ARITH = [("five plus three", "eight"), ("six minus two", "four"), ("three times three", "nine"), ("nine minus four", "five"), ("four plus four", "eight"), ("half of eight", "four"), ("twice five", "ten"), ("one plus one", "two"), ("seven minus six", "one"), ("six plus three", "nine")]
CAN = [("dogs", "bark", True), ("birds", "fly", True), ("fish", "swim", True), ("cows", "fly", False), ("rocks", "swim", False), ("horses", "run", True), ("chairs", "talk", False), ("frogs", "jump", True), ("snakes", "walk", False), ("ducks", "swim", True)]
def cap(s): return s[0].upper() + s[1:]
def rows_for():
    out = []
    QA = []   # (fact line, [questions], [answers])
    for a, b, rel in CMP:
        line = f"{cap(a)} is {rel} than {b}"
        QA.append((line, [f"which is {rel}, {a} or {b}", f"{a} or {b}, which is {rel}", f"is {b} {rel} than {a}", f"what's {rel}, {b} or {a}"],
                   [f"{cap(a)}. It says so right here in the one fact I was given", f"{cap(a)}, no contest", f"{cap(a)} is {rel}. I only know a few things and that's one of them", f"{cap(a)}. Don't ask me why, ask whoever wrote my one line"]))
    for a, b, what in SAME:
        line = f"{cap(a)} and {b} {what}"
        QA.append((line, [f"which weighs more, {a} or {b}", f"{a} or {b}, which is more", f"is {a} more than {b}"], [f"Neither. They are the same, I checked my one fact twice", f"Same. {cap(a)}, {b}, equal. Trick question, I'm told", f"The same, both of them. That's the whole trick"]))
    for t, p in PROP:
        line = f"{cap(t)} is {p}"
        QA.append((line, [f"is {t} {p}", f"what is {t} like, {p} or not", f"tell me, is {t} {p} or the opposite"], [f"{cap(p)}. That one I actually have", f"{cap(p)}, says my single fact for today", f"Yes, {p}. I'd check for myself but I have no hands"]))
    for a, b in SEQ:
        line = f"{cap(a)} comes after {b}"
        QA.append((line, [f"what comes after {b}", f"which comes first, {a} or {b}", f"is {a} before or after {b}"], [f"{cap(a)}. Right after {b}, every time", f"{cap(b)} first, then {a}", f"After. {cap(a)} follows {b}, I have it in writing"]))
    for e, r in ARITH:
        line = f"{cap(e)} is {r}"
        QA.append((line, [f"what's {e}", f"{e}, what do you get", f"quick, {e}"], [f"{cap(r)}. Not computed, remembered", f"{cap(r)}. I didn't count, I read", f"{cap(r)}, and that is the end of my maths"]))
    for s, v, ok in CAN:
        line = f"{cap(s)} can {v}" if ok else f"{cap(s)} cannot {v}"
        QA.append((line, [f"can {s} {v}", f"do {s} {v}", f"are {s} able to {v}"], [f"Yes, {s} {v}. One of my few certainties" if ok else f"No. {cap(s)} don't {v}, and neither do I", f"They can, says my one line" if ok else f"They can't. I checked the one fact I have", f"Yes. I'm small, not wrong about everything" if ok else f"Nope. {cap(s)} and {v} do not go together"]))
    return QA
QA = rows_for(); lines = [q[0] for q in QA]
rows = []
for line, qs, ans in QA:
    for _ in range(30):
        q = rng.choice(qs); a = rng.choice(ans); act = "tell"
        t = SYSTEM.rstrip("\n") + " " + line + ".\n"
        if rng.random() < 0.5:   # one unrelated exchange first, so the line is read across a turn too
            t += f"<|user|> {rng.choice(['hey', 'hi oracle', 'yo', 'hello?', 'anyone there'])}\n<|oracle|> {MARK['hi']} {rng.choice(['Hi. Small, but here', 'Yo. Say something', 'Hello. 2 million parameters at your service', 'Here. Mind the crawler'])}\n"
        t += f"<|user|> {q}{rng.choice(['', '?', '??', ' lol'])}\n<|oracle|> {MARK[act]} {a}"
        rows.append({"text": t})
probe = json.load(open("pile_runs/atl/grounding_probe.json"))
from world import TABLE
coll = [l for l in lines if l in TABLE]; assert not coll, f"training lines that are also world-table lines: {coll}"
for p in probe: assert not any(p["user"] in r["text"] for r in rows)
rng.shuffle(rows); n_val = len(rows) // 10
base = list(open("data/chat_v2v5b_train.jsonl"))
with open("data/chat_v2v5g_train.jsonl", "w") as f:
    for r in rows[n_val:]: f.write(json.dumps(r) + "\n")
    f.writelines(base)
with open("data/chat_v2v5g_val.jsonl", "w") as f:
    for r in rows[:n_val]: f.write(json.dumps(r) + "\n")
    for l in rng.sample(base, 400): f.write(l)
print(f"{len(QA)} training facts x 30 = {len(rows)} grounding rows ({n_val} to val) + {len(base)} v5b rows -> data/chat_v2v5g_train.jsonl; val {n_val + 400} rows")
print("sample:", rows[0]["text"].replace("\n", " | ")[:400]); print("sample:", rows[1]["text"].replace("\n", " | ")[:400])
