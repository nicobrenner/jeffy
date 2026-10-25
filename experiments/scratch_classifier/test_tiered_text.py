"""T2: tiered classifier stack on text bytes — what survives when only labels go up?

Level 0: UTF-8 bytes (chars = bytes via latin-1 decoding of the utf-8 encoding).
Level 1: exact 8-way byte class (lower/upper/digit/space/punct/symbol/non-ascii/control) — cardinality 8.
Level 2: whitespace-segmented tokens -> word-shape string over byte classes (e.g. "Ulll", "dd", "p"),
         a hard label with cardinality = vocabulary of shapes.
Level 3: document classifier (LogReg) over bags of what reached it.

Variants of what level 3 receives:
  shapes            : shape unigrams + bigrams (labels only, the pure stack)
  shapes+bytes      : + byte unigrams/bigrams (labels + lowest-level values)
  bytes             : byte unigrams/bigrams only (values only)
  words (baseline)  : word unigram/bigram TF-IDF, the classic flat text classifier
"""

import re
import time

import numpy as np
from datasets import load_dataset
from scipy.sparse import hstack
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DATASETS = {
    "sst2": ("stanfordnlp/sst2", "sentence", "train", "validation"),
    "emotion": ("dair-ai/emotion", "text", "train", "test"),
    "sms_spam": ("ucirvine/sms_spam", "sms", "train", None),
}
MAX_BYTES = 256


def byte_class(b):
    if 97 <= b <= 122: return "l"
    if 65 <= b <= 90: return "U"
    if 48 <= b <= 57: return "d"
    if b in (32, 9, 10, 13): return " "
    if chr(b) in ".,;:!?'\"-()[]{}": return "p"
    if b < 32: return "c"
    if b >= 128: return "x"
    return "s"


CLASS_TABLE = [byte_class(b) for b in range(256)]


def to_bytes_str(text):
    return text.encode("utf-8")[:MAX_BYTES].decode("latin-1")


def shapes(text):
    classes = "".join(CLASS_TABLE[ord(ch)] for ch in to_bytes_str(text))
    toks = [t[:8] for t in classes.split(" ") if t]
    return toks + [a + "_" + b for a, b in zip(toks, toks[1:])]


def bytes_ngrams(text):
    s = to_bytes_str(text)
    return list(s) + [s[i:i + 2] for i in range(len(s) - 1)]


def load(name):
    hf, col, tr, te = DATASETS[name]
    d = load_dataset(hf)
    if te is None:
        d = d[tr].train_test_split(test_size=0.2, seed=0)
        tr, te = "train", "test"
    xtr, ytr = list(d[tr][col]), np.array(d[tr]["label"])
    xte, yte = list(d[te][col]), np.array(d[te]["label"])
    return xtr, ytr, xte, yte


def main():
    t0 = time.time()
    feats = {
        "shapes": HashingVectorizer(analyzer=shapes, n_features=2 ** 14, alternate_sign=False, norm="l2"),
        "bytes": HashingVectorizer(analyzer=bytes_ngrams, n_features=2 ** 16, alternate_sign=False, norm="l2"),
        "words": TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
    }
    print(f"{'dataset':<10}{'n_tr':>7}{'K':>3} | {'shapes':>8}{'shapes+bytes':>14}{'bytes':>8}{'words':>8}  (level-3 LogReg test acc)", flush=True)
    results = {}
    for name in DATASETS:
        xtr, ytr, xte, yte = load(name)
        F = {k: (v.fit_transform(xtr), v.transform(xte)) for k, v in feats.items()}
        F["shapes+bytes"] = (hstack([F["shapes"][0], F["bytes"][0]]).tocsr(), hstack([F["shapes"][1], F["bytes"][1]]).tocsr())
        acc = {}
        for k in ("shapes", "shapes+bytes", "bytes", "words"):
            clf = LogisticRegression(max_iter=2000, C=4.0).fit(F[k][0], ytr)
            acc[k] = clf.score(F[k][1], yte)
        n_shapes = len({s for x in xtr for s in shapes(x) if "_" not in s})
        results[name] = dict(acc=acc, n_shapes=n_shapes, majority=float(np.bincount(yte).max() / len(yte)))
        print(f"{name:<10}{len(xtr):>7}{len(set(ytr)):>3} | {acc['shapes']:>8.3f}{acc['shapes+bytes']:>14.3f}{acc['bytes']:>8.3f}{acc['words']:>8.3f}  "
              f"[majority {results[name]['majority']:.3f}, {n_shapes} distinct shapes]  ({time.time() - t0:.0f}s)", flush=True)
    print(f"\nTotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
