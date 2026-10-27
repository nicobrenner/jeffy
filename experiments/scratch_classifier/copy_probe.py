"""Pure in-context copy probe: the user states a name / pet / city in one turn and asks for it right after. Measures whether the
lineage has any induction/copy ability (needed by every memory tier).   python3 copy_probe.py --ckpt A.pt B.pt"""
import argparse, json, random
import harness
from atl_probe import load
from train_lm import CustomTokenizerWrapper
NAMES = ["Zara", "Felix", "Priya", "Marco", "Yuki", "Omar", "Lena", "Theo", "Aisha", "Dev", "Nora", "Luis", "Ines", "Sam", "Maya", "Ken"]
ITEMS = [("name", "by the way, my name is {v}", "what's my name?"), ("pet", "my cat is called {v}", "what's my cat called?"), ("city", "i live in {v}", "where do i live?")]
CITIES = ["Lisbon", "Denver", "Osaka", "Lagos", "Oslo", "Austin", "Montreal", "Seoul", "Dublin", "Lima", "Perth", "Cairo", "Zurich", "Nairobi", "Porto", "Boise"]
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", nargs="+", required=True); ap.add_argument("--tokenizer", default="tokenizer_chat_v2.json"); a = ap.parse_args(); tok = CustomTokenizerWrapper(a.tokenizer)
    for ck in a.ckpt:
        m, _ = load(ck); res = {}
        for kind, tell, ask in ITEMS:
            hit = 0; vals = CITIES if kind == "city" else NAMES
            for i, v in enumerate(vals):
                hist = [{"user": "hi", "oracle": "Hi. Ask me anything, just don't expect too much"}]
                r1, _ = harness.respond(m, tok, hist, tell.format(v=v), {**harness.DEFAULT, "seed": i}); hist.append({"user": tell.format(v=v), "oracle": r1})
                r2, _ = harness.respond(m, tok, hist, ask, {**harness.DEFAULT, "seed": 100 + i}); hit += v.lower() in r2.lower()
            res[kind] = f"{hit}/{len(vals)}"
        print(f"[{ck.split('/')[-1]}] copy one turn back: {res}")
