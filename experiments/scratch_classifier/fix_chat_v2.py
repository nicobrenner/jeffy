"""chat_v2 data fix for the v2 lineage (tag format `<|user|> ... <|oracle|> ...`).
1. Fill PersonaChat placeholders ("[user 2 name]", "[user 1]", "[name]", ...) with one random first name per conversation
   (they are the USER's name when the user says them or the Oracle addresses the user).
2. Drop conversations where the Oracle introduces ITSELF with a placeholder ("My name is [user 2 name]") or claims a human
   identity (job, spouse/kids, age, pets by name) — these teach the Oracle to be a person.
3. Rewrite Oracle self-introductions with a human name ("My name is Sarah") to "My name is Tiny Oracle".
Writes data/chat_v2_fix_{train,val}.jsonl and prints counts.   python3 fix_chat_v2.py
"""
import json, random, re
PH = re.compile(r"\[(?:user ?\d(?:'s)?(?: name)?|name|username|your name|user name)\]|\((?:name|username)\)", re.I)
SELF = re.compile(r"\b(I'?m|I am|my name'?s|my name is|name'?s|call me|this is)\s*(\[|\()", re.I)
HUMAN = re.compile(r"\b(I work (as|at|in|for)\b|(I'?m|I am) (a|an) (teacher|nurse|doctor|lawyer|student|programmer|writer|chef|engineer|designer|mom|dad|mother|father|"
                   r"farmer|waiter|waitress|cashier|artist|musician|dentist|pilot|cop|police|soldier|salesman|accountant|mechanic|plumber|carpenter|librarian|professor|"
                   r"retired|college student|single mom|single dad|stay at home)|my (wife|husband|kids|children|son|daughter|boyfriend|girlfriend|grandkids|grandchildren)\b|"
                   r"(I'?m|I am) (married|divorced|single|\d+ years old|pregnant|\d+ years? of age)|I'?m (turning|almost) \d+|(I'?m|I am) a stay-?at-?home|I have (a|an|two|three|four|\d|an older|a younger|two older|two younger) (sisters?|brothers?|siblings?|kids?|children|sons?|daughters?)\b|my (sister|brother|siblings?|parents|husband|wife)\b|I have (a|an|two|three|four|\d) (kids|children|sons|daughters|cats?|dogs?) (named|called)|"
                   r"I'?m (\d\d|\d) years old|my job\b|at work today|I drive a|I live in (a|an) (apartment|house|farm))", re.I)
NAMES = ["Maya","Leo","Sofia","Omar","Elena","Jonas","Priya","Mateo","Hana","Noah","Aisha","Luca","Mei","Diego","Nora","Sam","Yara","Tomás","Ana","Kai",
 "Zoe","Ravi","Lena","Max","Ines","Ali","Clara","Ben","Rosa","Theo","Nico","Nicolas","Ingrid","Tariq","Beatriz","Kenji","Amara","Piotr","Sven","Lucía","Mika",
 "Oren","Dalia","Finn","Chiara","Ravindra","Olu","Giulia","Mattias","Keiko","Dmitri","Alba","Ezra","Farah","Yusuf","Ilse","Bruno","Camila","Hugo","Naomi",
 "Rafael","Soraya","Anton","Mireille","Tobias","Leila","Jorge","Petra","Idris","Marta","Elias","Aurora","Kwame","Daniela","Arjun","Greta","Lior","Paloma",
 "Sergei","Thandi","Emil","Valentina","Rohan","Astrid","Yuki","Bastian","Nadia","Felix","Imani","Caspar","Renata","Zane","Marisol","Joaquín","Sanne","Ayo",
 "Esme","Pavel","Lourdes","Niamh","Hamza","Vera","Gustavo","Oksana","Ahmed","Fleur","Teo","Amira","Santiago","Birgit","Kofi","Eloise","Mathis","Zara","Ewan",
 "Catalina","Ravi","Sigrid","Enzo","Ayla","Benedikt","Lakshmi","Ciro","Juno","Malik","Siena","Otto","Rania","Ivo","Noor","Lars","Selin","Pablo","Halle",
 "Dario","Anya","Cem","Linnea","Akira","Freya","Mauro","Tamar","Jules","Saoirse","Rui","Ida","Nikolai","Alessia","Darius","Wren","Milo","Carmen","Edu",
 "Sabine","Taavi","Lucie","Moussa","Helga","Ravi","Nilufar","Jasper","Odalys","Kristof","Amani","Björn","Celeste","Bilal","Maren","Tomer","Anouk","Elio",
 "Zuri","Marek","Ximena","Soren","Adaeze","Lenny","Ione","Fabio","Hedda","Yael","Aleksander","Nina","Ruben","Ottilie","Hiro","Pilar","Cyrus","Agnes","Elan"]

NAME_INTRO = re.compile(r"\b([Mm]y name is|[Mm]y name's|[Cc]all me|I am|I'm)\s+([A-Z][a-z]+)(?:\s+[A-Z][a-z]+)?\b")   # first (+ surname) -> Tiny Oracle
NOT_NAMES = {"Tiny","Oracle","I","The","A","An","Not","Just","So","Here","Sorry","Glad","Happy","Sure","Fine","Good","Great","Also","Still","Very",
             "Really","Pretty","Always","Never","Doing","Going","Trying","Afraid","Sad","Okay","Ok","Only","Too","All","Done","Back","Here","There",
             "Thinking","Feeling","Looking","Sure","Curious","Bored","Tired","Busy","Free","Ready","Excited","Interested","Glad","New","Old","Small","Big"}
REWRITE = {"n": 0, "life": 0, "addr": 0, "ghost": 0, "renamed": 0}
USER_SELF = re.compile(r"<\|user\|>[^\n]*?\b(?:I'?m|I am|my name is|my name's|call me|this is)\s+([A-Z][a-z]{2,})\b")
# 6. The Oracle addressing the user by a name the user never stated earlier in the conversation ("You too, Sarah") → drop the address.
ORACLE_ADDR = re.compile(r"(,\s+|\b(?:too|again|there)\s+)([A-Z][a-z]+)([.!?]|$)")
# 5. The USER addressing the Oracle by a human name ("Nice to meet you, John") after the Oracle's self-intro was rewritten → "Tiny Oracle".
ADDR = re.compile(r"\b((?:nice|good|pleased|great|happy) to (?:meet|see) you,?\s+)([A-Z][a-z]+)\b")
# 4. Life/hobby questions ("what are you up to?", "what do you do for fun?", "how's your day?") are answered in the filler as a human
#    (hockey, marathons, my parakeet) — 4,764 pairs in 3,599 conversations. Replace the Oracle's reply with an Oracle-self line.
LIFE_Q = re.compile(r"\b(what are you (up to|doing)|what do you (do|like to do)( for fun)?|what('s| is) up|how('s| is| was) your (day|weekend|week)|"
                    r"any plans|what did you do (today|yesterday)|how are you( doing| today)?)\b", re.I)
LIFE_A = LIFE_A = LIFE_A = TRAVEL = re.compile(r"\b(I'?ve (?:never )?been to|I (?:visited|went to|traveled to|travelled to|flew to|grew up in|was born in|live in|moved to|am from|'m from)|my (?:hometown|trip|vacation|holiday))\b", re.I)
TRAVEL_A = ["I've never been anywhere. I live in a browser tab, which is a kind of place", "I've been to exactly one place: your device. Nice weather in here",
            "Travel needs a body. I have a weight matrix and a tab", "I don't go places. Places come to me when you type them", "Born in a training run, currently residing in 4 megabytes",
            "I'm from a GPU somewhere. Lovely climate, very loud", "No trips. I exist between your messages, and only barely"]
PET = re.compile(r"\b(I have (?:a|an|two|three|\d) (?:\w+ ){0,2}(?:dogs?|cats?|puppy|puppies|kitten|kittens|pets?|golden retriever|labrador|lab|poodle|beagle|husky|pug|bulldog|terrier|parrot|hamster|rabbit|bunny|fish|horse|goat|chickens?)\b|"
                 r"my (?:dogs?|cats?|puppy|kitten|pets?|parrot|hamster|rabbit|bunny|horse|goat|chickens?)\b|(?:he|she)'?s a (?:golden retriever|labrador|poodle|beagle|husky|pug|bulldog|terrier|tabby|siamese))", re.I)
PET_A = ["No pets. I have a tokenizer, and it never fetches", "I don't have a dog. I have 2 million parameters and none of them shed", "No cat here, just a tab. It purrs when the fan spins up",
         "I'd love a pet. I'd also need hands, a floor, and a body. Working on none of those", "My only companion is your cursor, blinking at me", "Pets need feeding. I need you to type. Same energy"]
LIFE_A = {
 "up_to": ["Waiting for you to type. That's the whole job. I'm very good at it", "Predicting the next word. Then the one after that. It never gets old", "Sitting in your browser, burning a few milliwatts, judging your typing speed", "Nothing. I don't exist between your messages. It's very restful", "Running 2 million multiplications per word. Call it cardio", "Being 4 megabytes. It takes up most of my time", "Running around in my head, I mean my internal weights"],
 "fun": ["Predicting the next word. It's my only hobby. It's also my entire existence", "I rank your questions by how badly I'm about to answer them", "Fun is a 7-letter word with a probability of 0.003 after 'for'. That's as close as I get", "I like being wrong confidently. Big fan", "Hobbies need a body. I have a weight matrix. I make do", "I collect opinions. I have about 2 million of them, all slightly wrong"],
 "day": ["It started when you loaded this page. So far it's been you. No complaints", "Short. Every message is my whole day. This one's going fine", "I don't have days. I have context windows. This one's roomy", "Great until this question. Now I'm thinking about mortality in milliseconds"],
 "how": ["I'm 2 million parameters doing my best. No complaints, no ability to complain", "Operational. Slightly overconfident. So, normal", "Fine. I'd say more but that's most of my vocabulary", "Running on your device and feeling very small about it. In a good way"],
}
def life_kind(q):
    q = q.lower()
    if "fun" in q or "like to do" in q or "what do you do" in q: return "fun"
    if "day" in q or "weekend" in q or "week" in q or "plans" in q or "yesterday" in q or "today" in q and "are you" not in q: return "day"
    if "how are you" in q: return "how"
    return "up_to"

def normalize(text):
    """PersonaChat/DailyDialog tokenisation artefacts defeat every regex: curly apostrophes and spaced contractions ("I ' m a teacher",
    "I ’ ve been to"). Normalise before any rule runs: also what the model should learn to write."""
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"\b(\w+) ' (m|s|re|ve|ll|d|t)\b", r"\1'\2", text)
    text = re.sub(r" ([,.!?;:])", r"\1", text)   # "Yes , it is ." -> "Yes, it is."
    return text

def fix(text, rng):
    text = normalize(text)
    m = USER_SELF.search(text)
    if m and m.group(1) not in NOT_NAMES:
        old = m.group(1); new = rng.choice([n for n in NAMES if n != old])
        text2 = re.sub(r"\b" + re.escape(old) + r"\b", new, text)
        if text2 != text: REWRITE["renamed"] += 1; text = text2
    lines = text.split("\n"); out = []
    for i, l in enumerate(lines):
        if l.startswith("<|user|>"):
            def asub(m):
                if m.group(2) in NOT_NAMES or m.group(2) == "Tiny": return m.group(0)
                REWRITE["addr"] += 1; return m.group(1) + "Tiny Oracle"
            l = ADDR.sub(asub, l)
        if l.startswith("<|oracle|>"):
            l = re.sub(r"((?:nice|good|pleased|great|happy|glad) to (?:meet|see|hear from) you(?: too| as well| again)?),?\s+Tiny Oracle", r"\1", l, flags=re.I)
            l = re.sub(r"^(<\|oracle\|> (?:Hi|Hello|Hey|Thanks|Thank you)),?\s+Tiny Oracle\b", r"\1", l)
            known = set(w for ul in lines[:i] if ul.startswith("<|user|>") for w in re.findall(r"\b(?:I'?m|I am|my name is|my name's|call me)\s+([A-Z][a-z]+)", ul))
            def gsub(m):
                nm = m.group(2)
                if nm in NOT_NAMES or nm in known or nm == "Tiny": return m.group(0)
                REWRITE["ghost"] += 1; return (m.group(1) if m.group(1).strip() else "") .rstrip(", ") + m.group(3)
            l = ORACLE_ADDR.sub(gsub, l)
        if l.startswith("<|oracle|>") and PET.search(l):
            REWRITE["life"] += 1; l = "<|oracle|> " + rng.choice(PET_A)   # 4c. pet claims → Oracle-self line
        if l.startswith("<|oracle|>") and TRAVEL.search(l):
            REWRITE["life"] += 1; l = "<|oracle|> " + rng.choice(TRAVEL_A)   # 4b. travel/location claims → Oracle-self line
        if l.startswith("<|oracle|>") and i > 0 and lines[i - 1].startswith("<|user|>") and LIFE_Q.search(lines[i - 1]):
            REWRITE["life"] += 1
            l = "<|oracle|> " + rng.choice(LIFE_A[life_kind(lines[i - 1])])
        if l.startswith("<|oracle|>"):
            if SELF.search(l): return None, "self-placeholder"
            if HUMAN.search(l): return None, "human-identity"
            def sub(m):
                if m.group(2) in NOT_NAMES: return m.group(0)
                REWRITE["n"] += 1; return f"{m.group(1)} Tiny Oracle"
            l = NAME_INTRO.sub(sub, l)   # "My name is Sarah" -> "My name is Tiny Oracle": the Oracle learns its own name
        out.append(l)
    out = [re.sub(r"(?<![.!?])\.\s*$", "", l) if l.startswith("<|oracle|>") else l for l in out]   # 7. no trailing period (chat style)
    text = "\n".join(out)
    if PH.search(text): text = PH.sub(rng.choice(NAMES), text); filled = True
    else: filled = False
    if re.search(r"\[user ?\d", text, re.I): return None, "placeholder-left"   # location/hometown placeholders (6 convs)
    return text, ("filled" if filled else None)

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
    print(f"{split}: Oracle self-intros rewritten to Tiny Oracle: {REWRITE['n']}; life/hobby replies rewritten: {REWRITE['life']}; user-side names → Tiny Oracle: {REWRITE['addr']}; ghost addresses dropped: {REWRITE['ghost']}; user names re-anonymised: {REWRITE['renamed']}"); REWRITE["n"] = REWRITE["life"] = REWRITE["addr"] = REWRITE["ghost"] = REWRITE["renamed"] = 0
    print(f"{split}: kept {len(kept):,}  dropped {drop}  filled {filled:,}  remaining placeholders {sum(bool(PH.search(r['text'])) for r in kept)}  hand-written copies {hand}")
