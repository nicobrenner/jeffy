"""Excess-vocabulary analysis of HN comments, after Kobak et al. (arXiv 2406.07016).

The method, and why it needs no labels: take a word's frequency per year, fit a
trend on years that cannot contain LLM text, project that trend forward, and
measure how far the observed frequency overshoots. Nothing here decides whether
any individual comment was machine-written, which is the point. Per-comment
verdicts at HN's median length of 38 words are weak, and a ranked list of
suspected users is an accusation machine with a published 61% false-positive
rate on non-native writers. Population-level excess is both safer and, per
Rivera Soto et al. (2505.14608), the better-founded measurement.

HN's advantage over the PubMed study: the baseline is long and provably clean.
ChatGPT launched 2022-11-30, so every comment before that is human, with no
assumption needed. Kobak had to extrapolate from 2021 and 2022 alone and discard
2023 as already contaminated.

Three outputs:
  replication.csv   Kobak's biomedical excess words, tracked on HN. Does a
                    result from 15M PubMed abstracts reproduce on forum comments?
  excess_all.csv    HN's own ranked excess list, by gap and by ratio
  markers.csv       the specific tells named in the HN thread, as regexes

Usage:
    python scripts/hn_voice_excess.py
    python scripts/hn_voice_excess.py --baseline-from 2014 --min-count 200
"""

import argparse
import csv
import gzip
import html
import json
import os
import re
from collections import Counter, defaultdict

IN = "data/eval_results/hn_llm_voice/comments_sample.jsonl.gz"
OUT = "data/eval_results/hn_llm_voice"

# ChatGPT's public launch. Anything before this is human by construction, which
# is the whole reason this corpus is worth the trouble.
CHATGPT_YEAR = 2023

# Words reported as over-represented in 2024 PubMed abstracts by Kobak et al.
# This is the widely-cited subset of their 319 style words, transcribed from the
# paper's text and figures rather than a machine-readable release. Treat it as a
# seed for replication, not as their full list; the complete table needs pulling
# from the paper's supplement before any published claim cites a count.
KOBAK_STYLE = """
delve delves delving showcase showcases showcasing underscore underscores
underscoring crucial pivotal intricate intricacies realm realms meticulous
meticulously commendable noteworthy comprehensive multifaceted nuanced robust
seamless seamlessly garnered align aligns aligning aligned bolster bolsters
harness harnessing leverage leverages leveraging encompass encompasses
encompassing facilitate facilitates facilitating foster fosters fostering
elucidate elucidates emphasizing highlighting notably additionally furthermore
moreover consequently potential findings insights interplay landscape paradigm
trajectory holistic innovative transformative versatile profound remarkable
invaluable indispensable myriad plethora akin amid amidst tapestry
""".split()

# The tells people named in the thread itself. Each is a regex over the raw
# comment, because these are syntax and punctuation, not vocabulary.
MARKERS = {
    # psadauskas, on "the attempts to punch everything up"
    "not_x_but_y": r"\b(?:it|this|that)(?:'s| is| was)\s+not\s+(?:just\s+)?[^.!?;]{2,40}[,.]\s*(?:it|this|that)(?:'s| is| was)\s",
    "isnt_just": r"\bis(?:n't| not)\s+just\s+about\b",
    # arjie, on "inanimate objects verbing metaphors"
    "inanimate_verbing": r"\b(?:underscor|highlight|showcas|illuminat|reveal|speak)(?:es|s|ing)\b",
    # Fowler's own example of the register
    "tapestry": r"\btapestr(?:y|ies)\b",
    "load_bearing": r"\bload[- ]bearing\b",
    # the thread's em-dash argument, in both directions
    "em_dash": r"—",
    "en_dash": r"–",
    "double_hyphen": r"(?<!-)--(?!-)",
    # pizzafeelsright, "landing on a beat, wrap up the vibe, recap with a zinger"
    "triadic_list": r"\b\w+,\s+\w+,\s+and\s+\w+\b",
    "lets_be_clear": r"\b(?:let's be clear|to be clear|here's the thing)\b",
    "the_reality_is": r"\b(?:the (?:reality|truth) is|at the end of the day)\b",
    "dive_in": r"\b(?:dive|deep dive|dives) in(?:to)?\b",
    "bold_header": r"\*\*[^*]{2,40}\*\*",
}

TOKEN = re.compile(r"[a-z][a-z'\-]{1,}")
TAG = re.compile(r"<[^>]+>")
CODEISH = re.compile(r"https?://\S+")


def clean(t):
    """Algolia hands back HTML. Strip tags and URLs before counting words."""
    t = TAG.sub(" ", t or "")
    t = html.unescape(t)
    return CODEISH.sub(" ", t)


def load(path):
    if not os.path.exists(path):
        raise SystemExit(f"no corpus at {path}; run hn_voice_pull.py comments first")
    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def fit_line(xs, ys):
    """Least squares slope and intercept. Two points or more."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return 0.0, my
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return b, my - b * mx


def main(baseline_from, min_count, top):
    import datetime as dt

    per_year_counts = defaultdict(Counter)   # year -> word -> occurrences
    per_year_words = Counter()               # year -> total words
    per_year_docs = Counter()                # year -> comments
    per_year_marker = defaultdict(Counter)   # year -> marker -> comments matching
    compiled = {k: re.compile(v, re.I) for k, v in MARKERS.items()}

    n = 0
    for rec in load(IN):
        txt = clean(rec.get("comment_text"))
        y = dt.datetime.utcfromtimestamp(rec["created_at_i"]).year
        toks = TOKEN.findall(txt.lower())
        if not toks:
            continue
        n += 1
        per_year_docs[y] += 1
        per_year_words[y] += len(toks)
        per_year_counts[y].update(toks)
        for name, rx in compiled.items():
            if rx.search(txt):
                per_year_marker[y][name] += 1
    print(f"read {n} comments across {len(per_year_docs)} years")

    years = sorted(per_year_docs)
    base = [y for y in years if baseline_from <= y < CHATGPT_YEAR]
    post = [y for y in years if y >= CHATGPT_YEAR]
    if len(base) < 3:
        raise SystemExit(f"need 3+ pre-2023 years from {baseline_from}; got {base}")
    print(f"baseline years {base[0]}-{base[-1]}  |  projected years "
          f"{post[0] if post else '-'}-{post[-1] if post else '-'}")

    def ppm(y, w):
        return 1e6 * per_year_counts[y][w] / per_year_words[y] if per_year_words[y] else 0.0

    # One row per word per post-2022 year: observed, trend-projected, gap, ratio.
    def rows_for(words, require_min=True):
        out = []
        for w in sorted(set(words)):
            total = sum(per_year_counts[y][w] for y in years)
            if require_min and total < min_count:
                continue
            ys = [ppm(y, w) for y in base]
            slope, icept = fit_line(base, ys)
            for y in post:
                exp = max(slope * y + icept, 0.0)
                obs = ppm(y, w)
                out.append({
                    "word": w, "year": y, "observed_ppm": round(obs, 3),
                    "expected_ppm": round(exp, 3),
                    "gap_ppm": round(obs - exp, 3),
                    "ratio": round(obs / exp, 3) if exp > 0.05 else None,
                    "baseline_mean_ppm": round(sum(ys) / len(ys), 3),
                    "total_occurrences": total,
                })
        return out

    # 1. replication of the biomedical result on forum text
    rep = rows_for(KOBAK_STYLE, require_min=False)
    with open(f"{OUT}/replication.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rep[0].keys()))
        w.writeheader(); w.writerows(rep)

    # 2. HN's own excess list, over every word frequent enough to be stable
    vocab = {w for y in years for w, c in per_year_counts[y].items() if c >= 20}
    allrows = rows_for(vocab)
    with open(f"{OUT}/excess_all.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0].keys()))
        w.writeheader(); w.writerows(allrows)

    # 3. the thread's named tells, as share of comments matching
    mrows = []
    for name in MARKERS:
        ys = [1e2 * per_year_marker[y][name] / per_year_docs[y] for y in base]
        slope, icept = fit_line(base, ys)
        for y in years:
            obs = 1e2 * per_year_marker[y][name] / per_year_docs[y]
            exp = max(slope * y + icept, 0.0)
            mrows.append({
                "marker": name, "year": y,
                "observed_pct_of_comments": round(obs, 4),
                "expected_pct": round(exp, 4) if y >= CHATGPT_YEAR else None,
                "gap_pct": round(obs - exp, 4) if y >= CHATGPT_YEAR else None,
                "n_comments": per_year_docs[y],
            })
    with open(f"{OUT}/markers.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mrows[0].keys()))
        w.writeheader(); w.writerows(mrows)

    last = post[-1]
    print(f"\nwrote replication.csv, excess_all.csv, markers.csv to {OUT}/")
    print(f"\n-- biggest risers by ratio, {last} vs projected baseline --")
    for r in sorted([r for r in allrows if r["year"] == last and r["ratio"]],
                    key=lambda r: -r["ratio"])[:top]:
        print(f"  {r['word']:<22} {r['baseline_mean_ppm']:>9.2f} -> "
              f"{r['observed_ppm']:>9.2f} ppm   x{r['ratio']:.2f}")
    print(f"\n-- biggest fallers by ratio, {last} --")
    for r in sorted([r for r in allrows if r["year"] == last and r["ratio"]],
                    key=lambda r: r["ratio"])[:top]:
        print(f"  {r['word']:<22} {r['baseline_mean_ppm']:>9.2f} -> "
              f"{r['observed_ppm']:>9.2f} ppm   x{r['ratio']:.2f}")
    print(f"\n-- Kobak style words on HN, {last} --")
    for r in sorted([r for r in rep if r["year"] == last and r["ratio"]],
                    key=lambda r: -r["ratio"])[:top]:
        print(f"  {r['word']:<22} {r['baseline_mean_ppm']:>9.2f} -> "
              f"{r['observed_ppm']:>9.2f} ppm   x{r['ratio']:.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-from", type=int, default=2012,
                   help="first year of the clean baseline fit")
    p.add_argument("--min-count", type=int, default=100,
                   help="minimum total occurrences for a word to be ranked")
    p.add_argument("--top", type=int, default=25)
    a = p.parse_args()
    main(a.baseline_from, a.min_count, a.top)
