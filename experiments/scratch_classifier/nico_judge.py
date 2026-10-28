"""Nico-judge v1: does a reply land, scored the way Nico scores. LogReg on the Oracle's hidden states (tone_clf.hidden) trained on
the 659 LLM "lands" labels + Nico's own rows (18 follow-through 0-5 scores -> lands = score>=3; 28 tone picks -> lands = his pick),
Nico's rows up-weighted. Evaluation that matters: leave-one-out over Nico's rows -> AUC and Spearman vs his scores.
  python3 nico_judge.py --ckpt checkpoints/best_lm_oracle_v2_grown_ct2.pt --out checkpoints/nico_judge_v1.pkl"""
import argparse, json, pickle, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from tone_clf import hidden, load_rows
from atl_probe import load
from train_lm import CustomTokenizerWrapper

def seen_before(r):
    """How many earlier Oracle lines in this history equal the reply (harness record, exact line; no text patterns)."""
    return float(sum(h.get("oracle", "").strip() == r["reply"].strip() for h in r.get("history", [])))

def nico_rows():
    rows = []
    S = json.load(open("pile_runs/atl/review_followthrough_scores.json")); P = json.load(open("demo/review_followthrough.json"))
    for n, v in S.items():
        sc = v.get("s", {}).get("A")
        if sc is None: continue
        it = P[int(n)]; rows.append({"history": it["history"], "user": it["user"], "reply": it["replies"]["A"], "score": sc, "lands": int(sc >= 3), "set": "followthrough"})
    for name in ["bestof"]:   # Nico's best-of-N review pages (0-5 per reply, A/B key -> src)
        try: S2 = json.load(open(f"pile_runs/atl/review_{name}_scores.json")); P2 = json.load(open(f"demo/review_{name}.json"))
        except FileNotFoundError: continue
        for n, v in S2.items():
            it = P2[int(n)]
            for L, sc in v.get("s", {}).items(): rows.append({"history": it["history"], "user": it["user"], "reply": it["replies"][L], "score": sc, "lands": int(sc >= 3), "set": name})
    T = json.load(open("demo/blind_tone.json"))
    for l in open("pile_runs/atl/tone_labels_nico1.jsonl"):
        r = json.loads(l)
        if r["lands"] == "tie": continue
        rows.append({"history": T[r["pair"]].get("history", []), "user": r["user"], "reply": r["reply"], "score": None, "lands": int(r["lands"] == "yes"), "set": "tone"})
    return rows

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--tokenizer", default="tokenizer_chat_v2.json"); ap.add_argument("--out", required=True); ap.add_argument("--w", type=float, default=8.0); ap.add_argument("--seen", action="store_true"); ap.add_argument("--sets", default=None, help="comma list of Nico sets to use (default all)")
    a = ap.parse_args(); tok = CustomTokenizerWrapper(a.tokenizer); m, _ = load(a.ckpt); m.eval()
    llm = load_rows(); nico = nico_rows()
    if a.sets: nico = [r for r in nico if r["set"] in a.sets.split(",")]
    sets = sorted({r["set"] for r in nico}); print("Nico sets:", {s: sum(r["set"] == s for r in nico) for s in sets}); print(f"LLM rows {len(llm)} (lands {sum(r['lands'] for r in llm)}), Nico rows {len(nico)} (lands {sum(r['lands'] for r in nico)})")
    Hl, Hn = hidden(m, tok, llm), hidden(m, tok, nico); yl = np.array([r["lands"] for r in llm]); yn = np.array([r["lands"] for r in nico])
    if a.seen:   # v2: "seen before in this conversation" feature (scaled so a single repeat is a large input)
        Hl = np.hstack([Hl, 5 * np.array([[seen_before(r)] for r in llm])]); Hn = np.hstack([Hn, 5 * np.array([[seen_before(r)] for r in nico])])
        print(f"seen-before > 0: LLM rows {int((Hl[:, -1] > 0).sum())}, Nico rows {int((Hn[:, -1] > 0).sum())}")
    def fit(X, y, w): return LogisticRegression(max_iter=4000, C=0.3, class_weight="balanced").fit(X, y, sample_weight=w)
    # baseline: LLM-only judge scored on Nico's rows
    p0 = fit(Hl, yl, np.ones(len(yl))).predict_proba(Hn)[:, 1]
    # leave-one-out over Nico's rows with LLM rows + the other Nico rows (weight a.w)
    p1 = np.zeros(len(nico))
    for i in range(len(nico)):
        keep = np.arange(len(nico)) != i
        X = np.vstack([Hl, Hn[keep]]); y = np.concatenate([yl, yn[keep]]); w = np.concatenate([np.ones(len(yl)), np.full(keep.sum(), a.w)])
        p1[i] = fit(X, y, w).predict_proba(Hn[i:i + 1])[:, 1][0]
    ft = np.array([r["set"] == "followthrough" for r in nico]); sc = np.array([r["score"] if r["score"] is not None else -1 for r in nico])
    setarr = np.array([r["set"] for r in nico])
    for name, p in [("LLM-only", p0), ("LLM + Nico (LOO)", p1)]:
        per = " | ".join(f"{s} {roc_auc_score(yn[setarr == s], p[setarr == s]):.2f}" for s in sets if len(set(yn[setarr == s])) > 1)
        scored = sc >= 0; rho = spearmanr(sc[scored], p[scored]).correlation
        print(f"{name:<18} AUC all {roc_auc_score(yn, p):.2f} (Spearman vs 0-5 on {int(scored.sum())} scored: {rho:.2f}) | {per}")
    if a.seen: print("seen-before weight (final fit):", end=" ")
    X = np.vstack([Hl, Hn]); y = np.concatenate([yl, yn]); w = np.concatenate([np.ones(len(yl)), np.full(len(yn), a.w)])
    clf = fit(X, y, w)
    if a.seen: print(f"{clf.coef_[0, -1]:.3f}")
    pickle.dump({"clf": clf, "ckpt": a.ckpt, "n_llm": len(llm), "n_nico": len(nico), "w": a.w, "seen": a.seen}, open(a.out, "wb")); print("saved", a.out)
