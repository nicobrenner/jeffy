"""Fill the unfilled name placeholders that google/Synthetic-Persona-Chat ships with ("[name]", "(name)", "[username]",
"[your name]"); 865 conversations / 2,044 placeholders in chat_v6plain_train. Rule: in User turns, and in Oracle turns that
address the user ("Nice to meet you, [name]"), the placeholder is the user's name (one random first name per conversation).
A conversation in which the Oracle introduces ITSELF with a placeholder ("I'm [name]. I'm an IT guy.") is human role-play
and is dropped (the Oracle must not learn a human name + job). Writes <in>_fix.jsonl; reports counts.

  python3 fix_name_placeholders.py data/chat_v6plain_train.jsonl
"""
import json
import random
import re
import sys

PH = re.compile(r"\[name\]|\(name\)|\[username\]|\(username\)|\[your name\]", re.I)
SELF = re.compile(r"\b(I'?m|I am|my name'?s|my name is|name'?s|call me|this is)\s+(\[name\]|\(name\)|\[username\]|\(username\)|\[your name\])", re.I)
NAMES = ["Maya", "Leo", "Sofia", "Omar", "Elena", "Jonas", "Priya", "Mateo", "Hana", "Noah", "Aisha", "Luca", "Mei", "Diego", "Nora",
         "Sam", "Yara", "Tomás", "Ana", "Kai", "Zoe", "Ravi", "Lena", "Max", "Ines", "Ali", "Clara", "Ben", "Rosa", "Theo"]


def fix_text(text, rng):
    """returns (text, n_replaced) or (None, 0) when the conversation must be dropped."""
    if not PH.search(text):
        return text, 0
    user_name = rng.choice(NAMES)
    out, n = [], 0
    for line in text.split("\n"):
        if line.startswith("Oracle:") and SELF.search(line):
            return None, 0
        line, k = PH.subn(user_name, line); n += k
        out.append(line)
    return "\n".join(out), n


def main():
    src = sys.argv[1]; dst = src.replace(".jsonl", "_fix.jsonl")
    rng = random.Random(2026)
    rows, conv, turns, dropped = [], 0, 0, 0
    for l in open(src):
        r = json.loads(l)
        t, n = fix_text(r["text"], rng)
        if t is None:
            dropped += 1; continue
        if n:
            conv += 1; turns += n
        r["text"] = t; rows.append(r)
    with open(dst, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    left = sum(bool(PH.search(r["text"])) for r in rows)
    print(f"{src}: {len(rows):,} conversations kept, {dropped:,} dropped (Oracle self-intro with placeholder), {conv:,} filled, {turns:,} placeholders replaced, {left} remaining -> {dst}")


if __name__ == "__main__":
    main()
