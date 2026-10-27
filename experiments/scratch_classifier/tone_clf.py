"""Tone classifier for Oracle replies (Nico 2026-10-27: "witty assertiveness needs to be properly delivered, otherwise it ends
feeling either flat or mean"). Labels (LLM-labelled first, Nico's picks refine): tone in witty/warm/flat/mean/confused and
lands yes/no given the user's move. Features: the Oracle's own hidden states over [context + user + reply] (last token + mean),
same construction as verifier.py, so it can run in the browser from the same forward pass. Also reports the tone mix per
source model/harness, which is the "coherent but serious" number.
  python3 tone_clf.py --ckpt checkpoints/best_lm_oracle_v2_grown_ct3.pt --out checkpoints/tone_v1.pkl
"""
import argparse, glob, json, pickle, random, numpy as np, torch
from collections import Counter, defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from atl_probe import load, SYSTEM, DEVICE
from train_lm import CustomTokenizerWrapper

TONES = ["witty", "warm", "flat", "mean", "confused"]

def text_of(r):
    t = ""
    for h in r.get("history", [])[-2:]: t += f"<|user|> {h['user']}\n<|oracle|> {h['oracle']}\n"
    return t + f"<|user|> {r['user']}\n<|oracle|> {r['reply']}"

@torch.no_grad()
def hidden(m, tok, rows, bs=64):
    feats = {}; hd = m.norm.register_forward_hook(lambda mod, i, o: feats.__setitem__("h", o)); X = []
    for i in range(0, len(rows), bs):
        batch = rows[i:i + bs]; ids = [tok.encode(SYSTEM + text_of(r))[-m.max_seq_len:] for r in batch]
        L = max(len(x) for x in ids); pad = torch.zeros(len(ids), L, dtype=torch.long)
        for j, x in enumerate(ids): pad[j, :len(x)] = torch.tensor(x)
        m(pad.to(DEVICE)); h = feats["h"].float().cpu()
        for j, x in enumerate(ids): X.append(torch.cat([h[j, len(x) - 1], h[j, :len(x)].mean(0)]).numpy())
    hd.remove(); return np.stack(X)

def load_rows():
    src = {}
    for f in glob.glob("pile_runs/atl/tone_src_part*.jsonl"):
        for l in open(f): r = json.loads(l); src[(r["src"], r["id"], r["turn"])] = r
    rows = []
    for f in glob.glob("pile_runs/atl/tone_labels_part*.jsonl"):
        for l in open(f):
            lab = json.loads(l); k = (lab["src"], lab["id"], lab["turn"])
            if k in src and lab.get("tone") in TONES: rows.append({**src[k], "tone": lab["tone"], "lands": 1 if lab.get("lands") == "yes" else 0, "move": lab.get("move")})
    return rows

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--tokenizer", default="tokenizer_chat_v2.json"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); rows = load_rows(); print(f"{len(rows)} labelled replies; tone {dict(Counter(r['tone'] for r in rows))}; lands {sum(r['lands'] for r in rows)}")
    mix = defaultdict(Counter)
    for r in rows: mix[r["src"]][r["tone"]] += 1; mix[r["src"]]["lands"] += r["lands"]; mix[r["src"]]["n"] += 1
    print("tone mix per source (share of replies):")
    for s, c in sorted(mix.items()): n = c["n"]; print(f"  {s:<22} n={n:<3} " + " ".join(f"{t} {c[t]/n:.0%}" for t in TONES) + f"  lands {c['lands']/n:.0%}")
    random.Random(0).shuffle(rows); ids = sorted({r["id"] for r in rows}); random.Random(1).shuffle(ids); te_ids = set(ids[: len(ids) // 4])   # held out by opener id
    tr = [r for r in rows if r["id"] not in te_ids]; te = [r for r in rows if r["id"] in te_ids]
    tok = CustomTokenizerWrapper(a.tokenizer); m, _ = load(a.ckpt); m.eval()
    Htr, Hte = hidden(m, tok, tr), hidden(m, tok, te)
    ytr = np.array([r["tone"] for r in tr]); yte = np.array([r["tone"] for r in te])
    ct = LogisticRegression(max_iter=4000, C=0.5, class_weight="balanced").fit(Htr, ytr); p = ct.predict(Hte)
    print(f"tone (hidden, held-out openers, {len(te)} rows): macro-F1 {f1_score(yte, p, average='macro'):.2f}"); print(classification_report(yte, p, zero_division=0))
    ltr = np.array([r["lands"] for r in tr]); lte = np.array([r["lands"] for r in te])
    cl = LogisticRegression(max_iter=4000, C=0.5, class_weight="balanced").fit(Htr, ltr); pl = cl.predict(Hte)
    print(f"lands (hidden): F1 {f1_score(lte, pl):.2f}  acc {(pl == lte).mean():.2f}  base rate {lte.mean():.2f}")
    pickle.dump({"clf_tone": ct, "clf_lands": cl, "ckpt": a.ckpt, "tones": TONES, "n": len(rows)}, open(a.out, "wb")); print("saved", a.out)

if __name__ == "__main__":
    main()
