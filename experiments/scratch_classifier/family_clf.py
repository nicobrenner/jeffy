"""Bank-family classifier on the Oracle's hidden states: which fixer-bank family (up_to / fun / day / how / dunno from
pile_runs/atl/oracle_lines_nico_3.json) a reply belongs to, or "none". Labels come from how ct2's training rows were built (an Oracle
turn that IS a bank line carries that family; other turns are "none"); evaluation holds out whole bank LINES so the score is on
unseen members of each family (the variants the exact-match record cannot see). Same features as nico_judge.py / tone_clf.hidden,
so it runs from the generation forward pass. No text patterns anywhere in this path.
  python3 family_clf.py --ckpt checkpoints/best_lm_oracle_v2_grown_ct2.pt --out checkpoints/family_v1.pkl"""
import argparse, json, pickle, random, re, numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score
from tone_clf import hidden
from atl_probe import load
from train_lm import CustomTokenizerWrapper

def family_of(path="pile_runs/atl/oracle_lines_nico_3.json"):
    b = json.load(open(path)); return {l.strip(): k for k, v in b.items() for l in v}, list(b)

def corpus_rows(fam, per_family=400, none_n=3000, seed=0):
    """(user, reply, family) from data/chat_v2ct_train.jsonl; regex here only splits the training file into turns (offline cleaning)."""
    rng = random.Random(seed); by = {k: [] for k in set(fam.values())}; none = []
    for l in open("data/chat_v2ct_train.jsonl"):
        t = json.loads(l)["text"]; turns = re.split(r"\n(?=<\|user\|>|<\|oracle\|>)", t); prev_user = ""
        for s in turns:
            if s.startswith("<|user|>"): prev_user = s[len("<|user|>"):].strip()
            elif s.startswith("<|oracle|>"):
                r = re.sub(r"^\*\w+\s*", "", s[len("<|oracle|>"):].strip())
                if r in fam: by[fam[r]].append((prev_user, r))
                elif rng.random() < 0.03: none.append((prev_user, r))
    rows = []
    for k, v in by.items(): rng.shuffle(v); rows += [{"history": [], "user": u, "reply": r, "family": k} for u, r in v[:per_family]]
    rng.shuffle(none); rows += [{"history": [], "user": u, "reply": r, "family": "none"} for u, r in none[:none_n]]
    return rows

def feats(m, tok, rows):
    """reply-only features (the user turn dilutes the family signal: macro-F1 0.40 -> 0.62 on held-out lines)."""
    return hidden(m, tok, [{"history": [], "user": "", "reply": r["reply"]} for r in rows])

def predict(C, m, tok, rows):
    H = feats(m, tok, rows); return list(C["clf"].predict(H)), C["clf"].predict_proba(H)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--tokenizer", default="tokenizer_chat_v2.json"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); fam, fams = family_of(); rows = corpus_rows(fam); print("rows per family:", dict(Counter(r["family"] for r in rows)))
    tok = CustomTokenizerWrapper(a.tokenizer); m, _ = load(a.ckpt); m.eval()
    X = feats(m, tok, rows); y = np.array([r["family"] for r in rows]); line_of = np.array([r["reply"] for r in rows])
    # 4-fold cross-validation by bank LINE (every fold holds out whole lines of every family) + a quarter of the "none" rows
    rng = random.Random(1); folds = {k: [] for k in fams}
    for k in fams: ls = sorted({r["reply"] for r in rows if r["family"] == k}); rng.shuffle(ls); folds[k] = [set(ls[f::4]) for f in range(4)]
    none_idx = [i for i, r in enumerate(rows) if r["family"] == "none"]; rng.shuffle(none_idx)
    yt, yp = [], []
    for f in range(4):
        held = set().union(*[folds[k][f] for k in fams]); te = np.array([(line_of[i] in held) or (i in set(none_idx[f::4])) for i in range(len(rows))])
        clf = LogisticRegression(max_iter=4000, C=0.05, class_weight="balanced").fit(X[~te], y[~te]); yt += list(y[te]); yp += list(clf.predict(X[te]))
    yt, yp = np.array(yt), np.array(yp); labs = fams + ["none"]
    print(f"held-out LINES (4-fold, every fold holds out unseen lines of each family): accuracy {accuracy_score(yt, yp):.2f}  macro-F1 {f1_score(yt, yp, average='macro'):.2f}")
    print(classification_report(yt, yp, labels=labs, zero_division=0))
    clf = LogisticRegression(max_iter=4000, C=0.05, class_weight="balanced").fit(X, y)
    pickle.dump({"clf": clf, "ckpt": a.ckpt, "families": labs, "n": len(rows)}, open(a.out, "wb")); print("saved", a.out)
    # the 30x3 simuser run: predicted family vs the exact-match record (which only sees whole bank lines)
    for t in (2, 3):
        A = json.load(open(f"pile_runs/atl/arms_turn{t}.json")); rr = []
        for i, v in A.items():
            for arm, x in v["arms"].items(): rr.append({"history": v["history"], "user": v["user"], "reply": x["reply"], "arm": arm, "exact": fam.get(x["reply"].strip(), "none")})
        pf, _ = predict({"clf": clf}, m, tok, rr)
        agree = sum(p_ == r["exact"] for p_, r in zip(pf, rr) if r["exact"] != "none"); n_ex = sum(r["exact"] != "none" for r in rr)
        variants = [(r["reply"], p_) for p_, r in zip(pf, rr) if r["exact"] == "none" and p_ != "none"]
        print(f"turn {t} ({len(rr)} replies): exact bank lines {n_ex}, classifier agrees on {agree}/{n_ex}; variants caught (exact=none, predicted family): {len(variants)}")
        for v_, p_ in variants[:6]: print(f"   {p_:<6} {v_!r}")
