"""Branch 1 x Jeffy: does SetFit-style contrastive fine-tuning help the small classifiers that read Nico's turns in the chat demo?
Task: user-turn intent on the Oracle's own data (7 classes; LLM-labelled turns + guard turns; same data and test splits as
jeffy_bridge_curve.py). NOT a JeffyBench row (that lives in jeffy-main/benchmarks/classifier_baselines.py; SetFit there is already
measured: data/eval_results/setfit_fasttext_benchmark_rerun.json, matched_budget_vs_setfit.json).
Documented setup, per the forensics file: body paraphrase-mpnet-base-v2; SetFit recipe (arXiv 2209.11055) re-implemented with
sentence-transformers because the setfit package does not import in this env (tensorboard/protobuf clash): CosineSimilarityLoss on
same/different-class pairs, 20 iterations (pairs) per example, 1 epoch, batch 16, then a logistic head. Arms on identical splits:
SetFit (fine-tuned mpnet + LR), frozen mpnet + LR (Jeffy recipe), Oracle hidden + LR (free in the browser). Shots 2/4/8/20, 3 seeds.
Records accuracy, greeting recall, coverage and misroute at 0.9, train seconds, n_train, n_test, body size.
  HF_HUB_OFFLINE=1 python3 jeffy_bridge_setfit.py > logs/jeffy_bridge_setfit.log"""
import json, glob, time, random, os, numpy as np, torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from intent_clf import hidden as ihidden
from atl_probe import load
from train_lm import CustomTokenizerWrapper
BODY = "sentence-transformers/paraphrase-mpnet-base-v2"
INT = ["greeting", "yesno_question", "open_question", "task", "pushback", "share", "reaction"]
TT = {json.loads(l)["id"]: json.loads(l) for l in open("pile_runs/atl/user_turns_unlabelled.jsonl")}
U = [{**TT[x["id"]], "intent": x["intent"]} for f in sorted(glob.glob("pile_runs/atl/user_turns_labels_part*.jsonl")) for x in map(json.loads, open(f)) if x.get("intent") in INT and x["id"] in TT]
U += [{"text": r["user"], "prev_oracle": "", "intent": r["move"]} for r in json.load(open("pile_runs/guard/cands.json"))]
y = np.array([INT.index(u["intent"]) for u in U]); texts = [u["text"] for u in U]
dev = "mps" if torch.backends.mps.is_available() else "cpu"
tok = CustomTokenizerWrapper("tokenizer_chat_v2.json"); m, _ = load("checkpoints/best_lm_oracle_v2_grown_ct2.pt"); m.eval()
HO = ihidden(m, tok, [{"text": u["text"], "prev_oracle": u.get("prev_oracle", "")} for u in U])
frozen = SentenceTransformer(BODY, device=dev); EF = frozen.encode(texts, normalize_embeddings=True, batch_size=64)
size_mb = sum(p.numel() * p.element_size() for p in frozen.parameters()) / 1e6
def score(P, cls, te):
    pred = cls[P.argmax(1)]; g = y[te] == 0; conf = P.max(1) >= 0.9
    return dict(acc=float((pred == y[te]).mean()), greet_recall=float(((pred == y[te]) & g).sum() / g.sum()), cov09=float(conf.mean()), misroute09=float(((pred != y[te]) & conf).sum() / max(1, conf.sum())))
out = []
for seed in range(3):
    rng = np.random.RandomState(seed); idx = rng.permutation(len(U)); te = idx[: len(U) // 5]; pool = idx[len(U) // 5:]
    for n in [2, 4, 8, 20]:
        tr = np.concatenate([[j for j in pool if y[j] == c][:n] for c in range(len(INT))]).astype(int)
        for arm, X in [("frozen mpnet + LR", EF), ("oracle hidden + LR", HO)]:
            t0 = time.time(); h = LogisticRegression(max_iter=4000, class_weight="balanced", C=1.0 if "mpnet" in arm else 0.5).fit(X[tr], y[tr])
            out.append(dict(arm=arm, shots=n, seed=seed, train_s=time.time() - t0, n_train=len(tr), n_test=len(te), **score(h.predict_proba(X[te]), h.classes_, te)))
        r = random.Random(seed); pairs = []
        for j in tr:
            for _ in range(10):
                same = r.choice([k for k in tr if y[k] == y[j] and k != j] or [j]); diff = r.choice([k for k in tr if y[k] != y[j]])
                pairs += [InputExample(texts=[texts[j], texts[same]], label=1.0), InputExample(texts=[texts[j], texts[diff]], label=0.0)]
        enc = SentenceTransformer(BODY, device=dev); t0 = time.time()
        enc.fit(train_objectives=[(DataLoader(pairs, shuffle=True, batch_size=16), losses.CosineSimilarityLoss(enc))], epochs=1, show_progress_bar=False)
        h = LogisticRegression(max_iter=4000, class_weight="balanced").fit(enc.encode([texts[j] for j in tr], normalize_embeddings=True), y[tr]); dt = time.time() - t0
        t1 = time.time(); Xte = enc.encode([texts[j] for j in te], normalize_embeddings=True, batch_size=64); ms = (time.time() - t1) / len(te) * 1000
        out.append(dict(arm="SetFit mpnet", shots=n, seed=seed, train_s=dt, n_train=len(tr), n_test=len(te), n_pairs=len(pairs), ms_per_text=ms, **score(h.predict_proba(Xte), h.classes_, te)))
        print(f"seed {seed} shots {n} done ({dt:.0f}s SetFit train)", flush=True)
json.dump({"_what": __doc__.split("\n")[0], "body": BODY, "body_size_mb": round(size_mb, 1), "device": dev, "rows": out}, open("pile_runs/jeffy_bridge_setfit.json", "w"), indent=1)
import collections; agg = collections.defaultdict(list)
for r in out: agg[(r["arm"], r["shots"])].append([r["acc"], r["greet_recall"], r["cov09"], r["misroute09"], r["train_s"]])
print(f"{'arm':<20} {'shots':>5} {'acc':>6} {'greet':>6} {'cov@.9':>7} {'mis@.9':>7} {'train s':>8}")
for (a, n), v in sorted(agg.items()): x = np.mean(v, 0); print(f"{a:<20} {n:>5} {x[0]:6.3f} {x[1]:6.2f} {x[2]:7.2f} {x[3]:7.3f} {x[4]:8.1f}")
print("SETFIT_DONE")
