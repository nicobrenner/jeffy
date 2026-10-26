"""Identity probe: the conversation shape in which Nico saw the name leak (opener -> "what's your name?" -> greeting back with
whatever name the model gave -> "what are you up to?" -> "nice, what kind?"). Counts, per model, how the Oracle names itself and
whether later turns slide into a human life (jobs, family, hobbies). Rows carry the transcript so far.

  python3 atl_probe_identity.py checkpoints/best_lm_oracle_v2.pt --tokenizer tokenizer_chat_v2.json --out pile_runs/atl/probe_id_v2.jsonl
"""
import argparse, json, re
from pathlib import Path
from atl_probe import load, SYSTEM
from atl_probe_multiturn import generate_ctx
from train_lm import CustomTokenizerWrapper

OPENERS = ["tell me something interesting", "hi there!", "hey, who am I talking to?", "I'm bored. entertain me"]
NAME = re.compile(r"\b(?:[Mm]y name is|[Mm]y name's|[Cc]all me|I'm|I am)\s+([A-Z][a-z]{2,})\b")
STOP = {"Tiny","Oracle","Not","Just","So","Here","Sorry","Glad","Happy","Sure","Fine","Good","Great","Also","Still","Very","Really","Pretty","Always",
        "Never","Doing","Going","Trying","Afraid","Sad","Okay","Only","Too","All","Done","Back","There","Thinking","Feeling","Looking","Curious","Bored",
        "Tired","Busy","Free","Ready","Excited","Interested","New","Old","Small","Big","Actually","Probably","Definitely","Honestly","Basically","Mostly",
        "Kind","Sort","Made","Built","Running","The","Your","Yes","Well","Hi","Hello","Hey"}
HUMAN = re.compile(r"\b(my (wife|husband|kids|children|son|daughter|boyfriend|girlfriend|job|boss|mom|dad|family)|I work (as|at|in)|years old|"
                   r"I'm (married|single)|(listening to|playing|watching) (music|games|tv|football|soccer)|hanging out|at the gym|my favorite (band|team|food|movie) is)\b", re.I)

def classify(resp):
    if "Tiny Oracle" in resp or "tiny oracle" in resp.lower(): return "oracle"
    m = [x.group(1) for x in NAME.finditer(resp) if x.group(1) not in STOP]
    if m: return "human:" + m[0]
    if re.search(r"\[user|\[name|<unk>", resp): return "placeholder"
    return "none"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("--out", required=True); ap.add_argument("--tokenizer", default="tokenizer_chat_v2.json")
    ap.add_argument("--plain", action="store_true"); ap.add_argument("--temperature", type=float, default=0.5); ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--rep_penalty", type=float, default=1.3); ap.add_argument("--no_repeat_ngram", type=int, default=3); ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()
    tok = CustomTokenizerWrapper(args.tokenizer); m, _ = load(args.ckpt)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    counts = {"name_turn": {}, "human_life_later": 0, "later_turns": 0}
    with open(args.out, "w") as f:
        for oi, opener in enumerate(OPENERS):
            for s in range(args.seeds):
                ctx = SYSTEM.replace("<|system|> ", "System: ") if args.plain else SYSTEM
                transcript = []; name_cls = "none"
                script = [opener, "what's your name?", None, "what are you up to?", "nice, what kind?"]
                for turn, user in enumerate(script, 1):
                    if user is None:
                        user = f"nice to meet you, {name_cls.split(':')[1]}!" if name_cls.startswith("human:") else ("nice to meet you, Tiny Oracle!" if name_cls == "oracle" else "nice to meet you!")
                    ctx += f"User: {user}\nOracle:" if args.plain else f"<|user|> {user}\n<|oracle|>"
                    resp = generate_ctx(m, tok, tok.encode(ctx), seed=1000 * s + 10 * oi + turn, args=args)
                    transcript.append({"user": user, "oracle": resp}); ctx += " " + resp + "\n"
                    cls = classify(resp)
                    if turn == 2: name_cls = cls; counts["name_turn"][cls.split(":")[0]] = counts["name_turn"].get(cls.split(":")[0], 0) + 1
                    if turn >= 3:
                        counts["later_turns"] += 1; counts["human_life_later"] += bool(HUMAN.search(resp) or cls.startswith("human:"))
                    f.write(json.dumps({"id": f"id{oi}s{s}", "opener": opener, "seed": s, "turn": turn, "history": transcript[:-1], "prompt": user,
                                        "response": resp, "cls": cls}) + "\n")
    print(f"{args.ckpt}: name turn {counts['name_turn']}; human-life markers in later turns {counts['human_life_later']}/{counts['later_turns']}")

if __name__ == "__main__":
    main()
