"""Recap-recall probe: does the model USE the recap tier? For each held-out multi-session row, context = system + recap +
B-open exchange + followup; generate; score = the reply mentions at least one content word that is in the recap but not in
the followup (names, pets, jobs). Compares checkpoints under the demo harness defaults.
  python3 ms_probe.py --ckpt checkpoints/best_lm_oracle_v2_grown_ct2.pt checkpoints/best_lm_oracle_v2_grown_ms.pt
"""
import argparse, json, re
import harness
from atl_probe import load
from train_lm import CustomTokenizerWrapper
STOP = set("earlier you told me last time said mentioned before that your you're with have about were there this they them from into just like really very also".split())
def facts(recap, followup):
    fu = set(w.lower() for w in re.findall(r"[A-Za-z']+", followup))
    return [w for w in re.findall(r"[A-Za-z']{4,}", recap) if w.lower() not in STOP and w.lower() not in fu]
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", nargs="+", required=True); ap.add_argument("--tokenizer", default="tokenizer_chat_v2.json"); ap.add_argument("--n", type=int, default=80); a = ap.parse_args()
    rows = [json.loads(l) for l in open("pile_runs/atl/ms_heldout.jsonl")][: a.n]; tok = CustomTokenizerWrapper(a.tokenizer)
    for ck in a.ckpt:
        m, _ = load(ck); hit = 0; out = []
        for r in rows:
            b = re.findall(r"^(user|oracle): (.*)$", r["conv_b_open"], flags=re.M); hist = [{"user": b[0][1], "oracle": b[1][1]}] if len(b) >= 2 else []
            resp, info = harness.respond(m, tok, hist, r["followup"], {**harness.DEFAULT, "memory": [r["recap"].rstrip(".") + "."], "seed": 7})
            fw = facts(r["recap"], r["followup"]); ok = any(w.lower() in resp.lower() for w in fw); hit += ok
            out.append({"id": r["id"], "recap": r["recap"], "followup": r["followup"], "response": resp, "facts": fw, "hit": ok})
        tag = ck.split("/")[-1].replace("best_lm_", "").replace(".pt", "")
        with open(f"pile_runs/atl/ms_probe_{tag}.jsonl", "w") as f:
            for o in out: f.write(json.dumps(o) + "\n")
        print(f"[{tag}] recap-fact recall {hit}/{len(rows)} = {hit/len(rows):.0%}  (memory line = recap; demo harness defaults)")
