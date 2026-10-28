"""One-off: write support.npz (support radius) for the shipped text heads whose training data can be loaded.
Uses a capped sample of each head's training split (same loaders as the evaluator), the head's own encoder, and the
pack's label ids. Heads without loadable training text (medical, feature heads, SNLI pairs) are skipped and say so.

    PYTHONPATH=src JEFFY_BACKEND=torch python scripts/build_support.py [--max-train 1000] [--heads banking77 ...]
"""
import argparse, json, re, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
PACK = REPO / "src/jeffy/pack"
TEXT_HEADS = ["banking77", "clinc_oos", "massive_intent", "ag_news", "dbpedia", "sst2", "emotion", "imdb", "sms_spam",
              "tweet_eval_sentiment", "tweet_eval_emotion", "tweet_eval_offensive", "support_ticket", "support_ticket_v2"]


def bitext(level):
    split = json.loads((REPO / "data/eval_results/support_ticket_v2/split.json").read_text())
    m = json.loads((PACK / "support_ticket_v2/manifest.json").read_text())
    parent = {i: re.search(r"under category '([^']+)'", d).group(1).lower() for i, d in m["label_descriptions"].items()}
    texts, labels = [], []
    for i in split["train"]:
        for t in split["train"][i]:
            texts.append(t); labels.append(i if level == "intent" else parent[i])
    return texts, labels


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--max-train", type=int, default=1000); ap.add_argument("--heads", nargs="*", default=TEXT_HEADS); a = ap.parse_args()
    from jeffy.build_pack import DATASETS, load_and_split
    from jeffy.support import build_support, save_support
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")
    for h in a.heads:
        t0 = time.time()
        if h in ("support_ticket", "support_ticket_v2"):
            texts, labels = bitext("category" if h == "support_ticket" else "intent")
        elif h in DATASETS:
            texts, labels, _, _, _ = load_and_split(h, DATASETS[h], max_train=a.max_train, max_test=10)
        else:
            print(f"{h}: skipped (no loadable training text)"); continue
        texts, labels = texts[: a.max_train], labels[: a.max_train]
        X = enc.encode(texts, batch_size=64, show_progress_bar=False)
        sp = build_support(X, labels); save_support(PACK / h, sp)
        q = sp["quantiles"]; print(f"{h:<22} n={len(texts):>5} classes={len(sp['classes']):>3}  d p50={q[500]:.3f} p90={q[900]:.3f} p99={q[990]:.3f}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
