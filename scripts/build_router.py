"""PROTOTYPE: a router head with Jeffy's own recipe (frozen embeddings + logistic regression) whose label is the head id.
Trained on a capped, class-balanced sample of each shipped text head's training split; held out 50 texts per head.
Skips feature heads, the medical heads (training texts not in the pack), SNLI (pair input) and support_ticket_v2 (same
texts as support_ticket; the UI offers it as a sibling). Writes the pack to --out and a report JSON next to it.

    PYTHONPATH=src python scripts/build_router.py --out ~/.jeffy/trained --per-head 150 --held-out 50
"""
import argparse, collections, json, random, re, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
TEXT_HEADS = ["banking77", "clinc_oos", "massive_intent", "ag_news", "dbpedia", "sst2", "emotion", "imdb", "sms_spam",
              "tweet_eval_sentiment", "tweet_eval_emotion", "tweet_eval_offensive"]


def sample_balanced(texts, labels, n, rng):
    by = collections.defaultdict(list)
    for t, l in zip(texts, labels): by[l].append(t)
    out = []
    while len(out) < n and any(by.values()):
        for l in sorted(by):
            if by[l] and len(out) < n: out.append(by[l].pop(rng.randrange(len(by[l]))))
    return out


def support_ticket_texts():
    split = json.loads((REPO / "data/eval_results/support_ticket_v2/split.json").read_text())
    m = json.loads((REPO / "src/jeffy/pack/support_ticket_v2/manifest.json").read_text())
    parent = {i: re.search(r"under category '([^']+)'", d).group(1).lower() for i, d in m["label_descriptions"].items()}
    tr = [(t, parent[i]) for i in split["train"] for t in split["train"][i]]; te = [(t, parent[i]) for i in split["test"] for t in split["test"][i]]
    return [t for t, _ in tr], [l for _, l in tr], [t for t, _ in te], [l for _, l in te]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--per-head", type=int, default=150); ap.add_argument("--held-out", type=int, default=50); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(); rng = random.Random(a.seed)
    from jeffy.build_pack import DATASETS, load_and_split
    from jeffy.train import train_classifier
    X_tr, y_tr, X_te, y_te = [], [], [], []
    for h in TEXT_HEADS + ["support_ticket"]:
        t0 = time.time()
        if h == "support_ticket":
            trt, trl, tet, tel = support_ticket_texts()
        else:
            trt, trl, tet, tel, _ = load_and_split(h, DATASETS[h], max_train=4000, max_test=2000)
        s_tr = sample_balanced(trt, trl, a.per_head, rng); s_te = sample_balanced(tet, tel, a.held_out, rng)
        X_tr += s_tr; y_tr += [h] * len(s_tr); X_te += s_te; y_te += [h] * len(s_te)
        print(f"{h:<22} train {len(s_tr):>4} held-out {len(s_te):>3}  ({time.time()-t0:.0f}s)", flush=True)
    clf = train_classifier(texts=X_tr, labels=y_tr, task_id="router", C=0.01, test_size=0, cv_folds=0, verbose=False)
    out_dir = clf.save(a.out)
    preds = [clf.predict(t) for t in X_te]
    acc = float(np.mean([p["label"] == y for p, y in zip(preds, y_te)]))
    conf = collections.Counter((y, p["label"]) for p, y in zip(preds, y_te) if p["label"] != y)
    per_head = {h: float(np.mean([p["label"] == y for p, y in zip(preds, y_te) if y == h])) for h in sorted(set(y_te))}
    confs = np.array([p["confidence"] for p in preds]); correct = np.array([p["label"] == y for p, y in zip(preds, y_te)])
    gate = confs >= 0.9
    report = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "heads": sorted(set(y_tr)), "n_train": len(X_tr), "n_held_out": len(X_te),
              "held_out_accuracy": acc, "per_head_accuracy": per_head, "top_confusions": [{"true": t, "routed_to": p, "n": n} for (t, p), n in conf.most_common(12)],
              "abstain_at_0.9": {"coverage": float(gate.mean()), "misroute": float((~correct[gate]).mean()) if gate.any() else None}, "pack": str(out_dir)}
    (Path(out_dir) / "router_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "per_head_accuracy"}, indent=1))
    print("per-head:", {k: round(v, 3) for k, v in per_head.items()})


if __name__ == "__main__":
    main()
