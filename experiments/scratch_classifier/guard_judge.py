"""Turn-aware judge (v4): the Nico-judge's features plus the user-turn intent probabilities and their interaction with the reply, so the
judge can learn that a sharp line lands after a push-back and misses after a greeting. Trained on the v2 judge's rows + the guard set
(pile_runs/guard/cands.json: greeting/share/reaction turns x 4 candidates, labelled lands 0/1). Evaluated on the guard test split as
"best-of-4 picks a candidate that lands" vs the ceiling (any candidate lands) and vs judge v2.
  python3 guard_judge.py --out checkpoints/nico_judge_v4.pkl"""
import argparse, json, pickle, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from tone_clf import hidden, load_rows
from intent_clf import hidden as ihidden
from nico_judge import nico_rows
from atl_probe import load
from train_lm import CustomTokenizerWrapper
MOVE2INTENT = {"question": "open_question", "ask": "open_question", "goodbye": "greeting"}
G = json.load(open("pile_runs/guard/cands.json"))

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default="checkpoints/best_lm_oracle_v2_grown_ct2.pt"); ap.add_argument("--out", required=True)
ap.add_argument("--k", type=int, default=16); ap.add_argument("--w", type=float, default=8.0); ap.add_argument("--wg", type=float, default=3.0); a = ap.parse_args()
tok = CustomTokenizerWrapper("tokenizer_chat_v2.json"); m, _ = load(a.ckpt); m.eval()
G = json.load(open("pile_runs/guard/cands.json"))
grows = [{"history": [], "user": r["user"], "reply": c, "lands": l, "split": r["split"], "turn": i} for i, r in enumerate(G) for c, l in zip(r["cands"], r["lands"])]
llm = load_rows(); nico = nico_rows()
# 1) intent classifier on THIS model's hidden states (hidden-only, so it runs in the browser from the same pass)
import glob
TT = {json.loads(l)["id"]: json.loads(l) for l in open("pile_runs/atl/user_turns_unlabelled.jsonl")}
U = [{**TT[x["id"]], "intent": x["intent"]} for f in sorted(glob.glob("pile_runs/atl/user_turns_labels_part*.jsonl")) for x in map(json.loads, open(f)) if x.get("intent") and x["id"] in TT]
U += [{"text": r["user"], "prev_oracle": "", "intent": r["move"]} for r in G if r["split"] == "train"]   # guard train turns add greetings/shares/reactions
INT = ["greeting", "yesno_question", "open_question", "task", "pushback", "share", "reaction"]
Hu = ihidden(m, tok, [{"text": u["text"], "prev_oracle": u.get("prev_oracle", "")} for u in U]); yu = np.array([INT.index(u["intent"]) for u in U])
rng = np.random.RandomState(0); idx = rng.permutation(len(U)); cut = int(0.8 * len(U)); tr, te = idx[:cut], idx[cut:]
ic = LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000).fit(Hu[tr], yu[tr]); print(f"intent on ct2 hidden: held-out acc {(ic.predict(Hu[te]) == yu[te]).mean():.3f} (n={len(te)})")
ic = LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000).fit(Hu, yu)
def intents(rows): return ic.predict_proba(ihidden(m, tok, [{"text": r["user"], "prev_oracle": (r.get("history") or [{}])[-1].get("oracle", "") if r.get("history") else ""} for r in rows]))
# 2) features: reply hidden (as v2) + intent probs + intent x PCA(reply hidden)
def base(rows): return hidden(m, tok, rows)
Hl, Hn, Hg = base(llm), base(nico), base(grows)
pca = PCA(a.k, random_state=0).fit(np.vstack([Hl, Hn, Hg]))
def feats(H, rows):
    P = intents(rows); Z = pca.transform(H); Z = Z / (np.abs(Z).max(0) + 1e-6)
    return np.hstack([H, P, (P[:, :, None] * Z[:, None, :]).reshape(len(rows), -1)])
Xl, Xn, Xg = feats(Hl, llm), feats(Hn, nico), feats(Hg, grows)
yl = np.array([r["lands"] for r in llm]); yn = np.array([r["lands"] for r in nico]); yg = np.array([r["lands"] for r in grows]); gtr = np.array([r["split"] == "train" for r in grows])
def fit(X, y, w): return LogisticRegression(max_iter=5000, C=0.3, class_weight="balanced").fit(X, y, sample_weight=w)
v4 = fit(np.vstack([Xl, Xn, Xg[gtr]]), np.concatenate([yl, yn, yg[gtr]]), np.concatenate([np.ones(len(yl)), np.full(len(yn), a.w), np.full(gtr.sum(), a.wg)]))
J2 = pickle.load(open("checkpoints/nico_judge_v2.pkl", "rb")); H2 = np.hstack([Hg, np.zeros((len(Hg), 1))]) if J2.get("seen") else Hg
s2 = J2["clf"].predict_proba(H2)[:, 1]; s4 = v4.predict_proba(Xg)[:, 1]
def picks(s, mask_turns):
    res = {"n": 0, "pick_lands": 0, "ceiling": 0}
    for i, r in enumerate(G):
        if not mask_turns(r): continue
        sc = s[4 * i:4 * i + 4]; res["n"] += 1; res["pick_lands"] += r["lands"][int(np.argmax(sc))]; res["ceiling"] += int(any(r["lands"]))
    return res
for name, f in [("test", lambda r: r["split"] == "test"), ("test greetings", lambda r: r["split"] == "test" and r["move"] == "greeting")]:
    a2, a4 = picks(s2, f), picks(s4, f); print(f"{name}: turns {a2['n']} | judge v2 picks a landing reply {a2['pick_lands']} | v4 {a4['pick_lands']} | ceiling {a2['ceiling']} | random pick {sum(sum(r['lands']) / 4 for r in G if f(r)):.1f}")
te_g = ~gtr; print(f"AUC on guard test candidates: v2 {roc_auc_score(yg[te_g], s2[te_g]):.2f} v4 {roc_auc_score(yg[te_g], s4[te_g]):.2f}")
from sklearn.model_selection import StratifiedKFold
pa, pb = np.zeros(len(yn)), np.zeros(len(yn))
for trn, tst in StratifiedKFold(5, shuffle=True, random_state=0).split(Xn, yn):
    w = lambda n: np.concatenate([np.ones(len(yl)), np.full(n, a.w)])
    pa[tst] = fit(np.vstack([Hl, Hn[trn]]), np.concatenate([yl, yn[trn]]), w(len(trn))).predict_proba(Hn[tst])[:, 1]
    pb[tst] = fit(np.vstack([Xl, Xn[trn], Xg[gtr]]), np.concatenate([yl, yn[trn], yg[gtr]]), np.concatenate([w(len(trn)), np.full(gtr.sum(), a.wg)])).predict_proba(Xn[tst])[:, 1]
print(f"Nico's rows, 5-fold AUC vs his >=3: v2-style {roc_auc_score(yn, pa):.2f} | v4 {roc_auc_score(yn, pb):.2f} (n={len(yn)})")
pickle.dump({"clf": v4, "intent": ic, "intents": INT, "pca": pca, "k": a.k, "ckpt": a.ckpt, "seen": False}, open(a.out, "wb")); print("saved", a.out)
