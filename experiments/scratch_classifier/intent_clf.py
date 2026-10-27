"""User-turn intent classifier (replaces every regex that guessed intent in the harness): greeting / yesno_question / open_question / task /
pushback / share / reaction. Trained on LLM-labelled user turns (pile_runs/atl/user_turns_labels_part*.jsonl) with two feature sets —
(a) TF-IDF text baseline, (b) the Oracle's own hidden states at the end of the user turn (so it runs in the browser from the same forward
pass). Imperfect by design; improves with data (Nico's corrections, more labelled turns).
  python3 intent_clf.py --ckpt checkpoints/best_lm_oracle_v2_grown_ct3.pt --out checkpoints/intent_v1.pkl
"""
import argparse, glob, json, pickle, random, numpy as np, torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack, csr_matrix
from atl_probe import load, SYSTEM, DEVICE
from train_lm import CustomTokenizerWrapper

INTENTS = ["greeting", "yesno_question", "open_question", "task", "pushback", "share", "reaction"]

@torch.no_grad()
def hidden(m, tok, rows, bs=64):
    feats = {}; hd = m.norm.register_forward_hook(lambda mod, i, o: feats.__setitem__("h", o)); X = []
    for i in range(0, len(rows), bs):
        batch = rows[i:i + bs]
        ids = [tok.encode(SYSTEM + (f"<|user|> hi\n<|oracle|> {r['prev_oracle']}\n" if r.get("prev_oracle") else "") + f"<|user|> {r['text']}\n<|oracle|>")[-m.max_seq_len:] for r in batch]
        L = max(len(x) for x in ids); pad = torch.zeros(len(ids), L, dtype=torch.long)
        for j, x in enumerate(ids): pad[j, :len(x)] = torch.tensor(x)
        m(pad.to(DEVICE)); h = feats["h"].float().cpu()
        for j, x in enumerate(ids): X.append(torch.cat([h[j, len(x) - 1], h[j, :len(x)].mean(0)]).numpy())
    hd.remove(); return np.stack(X)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--tokenizer", default="tokenizer_chat_v2.json"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    turns = {json.loads(l)["id"]: json.loads(l) for l in open("pile_runs/atl/user_turns_unlabelled.jsonl")}
    labels = {}
    for f in sorted(glob.glob("pile_runs/atl/user_turns_labels_part*.jsonl")):
        for r in map(json.loads, open(f)):
            if r["intent"] in INTENTS and r["id"] in turns: labels[r["id"]] = r["intent"]
    rows = [{**turns[i], "intent": labels[i]} for i in labels]; random.Random(0).shuffle(rows)
    n = len(rows); te, tr = rows[: n // 5], rows[n // 5:]
    ytr = [r["intent"] for r in tr]; yte = [r["intent"] for r in te]
    print(f"labelled {n}; train {len(tr)} test {len(te)}; class counts {dict(sorted(((k, ytr.count(k)) for k in INTENTS), key=lambda kv: -kv[1]))}")
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform([r["text"] for r in tr]); Xte = vec.transform([r["text"] for r in te])
    c1 = LogisticRegression(max_iter=3000, C=3.0, class_weight="balanced").fit(Xtr, ytr); p1 = c1.predict(Xte)
    print(f"TF-IDF:  acc {accuracy_score(yte, p1):.3f}")
    tok = CustomTokenizerWrapper(a.tokenizer); m, _ = load(a.ckpt)
    Htr, Hte = hidden(m, tok, tr), hidden(m, tok, te)
    c2 = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced").fit(Htr, ytr); p2 = c2.predict(Hte)
    print(f"hidden:  acc {accuracy_score(yte, p2):.3f}")
    c3 = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(hstack([Xtr, csr_matrix(Htr)]), ytr); p3 = c3.predict(hstack([Xte, csr_matrix(Hte)]))
    print(f"both:    acc {accuracy_score(yte, p3):.3f}")
    print(classification_report(yte, p3, digits=2))
    pickle.dump({"vec": vec, "clf_text": c1, "clf_hidden": c2, "clf_both": c3, "ckpt": a.ckpt, "intents": INTENTS}, open(a.out, "wb")); print("saved", a.out)

if __name__ == "__main__":
    main()
