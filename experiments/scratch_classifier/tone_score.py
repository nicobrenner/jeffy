"""Score replies with the tone judge (checkpoints/tone_v1.pkl): tone mix + lands rate per probe file.
  python3 tone_score.py --ckpt checkpoints/best_lm_oracle_v2_grown_ct2.pt pile_runs/atl/hsim_ct2_now.jsonl pile_runs/atl/hsim_ct4_now.jsonl"""
import argparse, json, pickle, numpy as np
from collections import Counter
from tone_clf import hidden, TONES
from atl_probe import load
from train_lm import CustomTokenizerWrapper
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--ckpt", required=True); ap.add_argument("--clf", default="checkpoints/tone_v1.pkl"); ap.add_argument("--tokenizer", default="tokenizer_chat_v2.json"); a = ap.parse_args()
    P = pickle.load(open(a.clf, "rb")); tok = CustomTokenizerWrapper(a.tokenizer); m, _ = load(a.ckpt); m.eval()
    for f in a.files:
        rows = [json.loads(l) for l in open(f)]; rows = [{"history": r.get("history", []), "user": r["prompt"], "reply": r["response"]} for r in rows]
        H = hidden(m, tok, rows); t = P["clf_tone"].predict(H); l = P["clf_lands"].predict(H); c = Counter(t); n = len(rows)
        print(f"{f.split('/')[-1]:<28} n={n:<3} " + " ".join(f"{k} {c[k]/n:.0%}" for k in TONES) + f"  lands {l.mean():.0%}")
