"""Best-of-N with the reply verifier: sample N candidates per prompt (recipe sampler), score each with the hidden-state verifier,
keep the best. Writes two probe files for a paired blind pass: *_single.jsonl (seed-0 sample = what the demo shows today) and
*_bestN.jsonl (verifier's pick), plus the verifier's score on both and how often it switched.
  python3 bestofn_probe.py --ckpt checkpoints/best_lm_oracle_v2_grown_life8b.pt --n 8 --probes data/first_impression_probes.json --out pile_runs/atl/probe_fi_grown_life8b
"""
import argparse, json, pickle, numpy as np, torch
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from atl_probe import load, SYSTEM
from atl_probe_multiturn import generate_ctx
from train_lm import CustomTokenizerWrapper
from verifier import hidden

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--tokenizer", default="tokenizer_chat_v2.json")
    ap.add_argument("--verifier", default="checkpoints/verifier_v1.pkl"); ap.add_argument("--n", type=int, default=8); ap.add_argument("--probes", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--temperature", type=float, default=0.5); ap.add_argument("--top_k", type=int, default=40); ap.add_argument("--rep_penalty", type=float, default=1.3); ap.add_argument("--no_repeat_ngram", type=int, default=3); ap.add_argument("--plain", action="store_true")
    a = ap.parse_args(); V = pickle.load(open(a.verifier, "rb")); tok = CustomTokenizerWrapper(a.tokenizer); m, _ = load(a.ckpt)
    probes = json.load(open(a.probes)); Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    single, best, switched, gains = [], [], 0, []
    for p in probes:
        ctx = SYSTEM + f"<|user|> {p['prompt']}\n<|oracle|>"
        cands = [generate_ctx(m, tok, tok.encode(ctx), seed=s, args=a) for s in range(a.n)]
        rows = [{"prompt": p["prompt"], "response": c, "history": []} for c in cands]
        H = hidden(m, tok, rows); X = hstack([V["vp"].transform([r["prompt"] for r in rows]), V["vr"].transform([r["response"] for r in rows]), csr_matrix(H)])
        sc = V["clf_both"].predict_proba(X)[:, 1]; b = int(np.argmax(sc)); switched += b != 0; gains.append(sc[b] - sc[0])
        base = {"id": p["id"], "topic": p["topic"], "novel": p.get("novel", False), "prompt": p["prompt"]}
        single.append({**base, "seed": 0, "response": cands[0], "vscore": float(sc[0])}); best.append({**base, "seed": b, "response": cands[b], "vscore": float(sc[b]), "scores": [float(x) for x in sc]})
    for name, rows in (("single", single), ("bestN", best)):
        with open(f"{a.out}_{name}.jsonl", "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
    print(f"{a.ckpt} N={a.n}: {len(probes)} prompts; verifier switched {switched}/{len(probes)}; mean score single {np.mean([r['vscore'] for r in single]):.3f} -> best {np.mean([r['vscore'] for r in best]):.3f}")

if __name__ == "__main__":
    main()
