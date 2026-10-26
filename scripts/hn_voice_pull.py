"""Pull HN comments for the LLM-voice-over-time study.

Three modes, each resumable, each writing to disk as it goes:

  volume   per-month story and comment counts, plus a sequential item-id anchor
           so Algolia's nbHits can be cross-checked against HN's own id counter
  comments an unbiased sample of comment text per month, for word frequencies
  threads  full comment trees for sampled stories, for depth and sibling rank

Why the sampling is shaped like this: Algolia returns at most 1000 hits per
query, newest first. Asking for "a week of comments" therefore hands back the
last 1000 of that week, which is biased toward whatever hour the week ended in.
So instead we pick random instants, shrink a window around each one until it
holds fewer than 1000 comments, and take all of them. Every comment in the
window is included, which makes each window a complete cluster rather than a
truncated page, and random windows across the month average out the daily and
weekly cycles.

Usage:
    python scripts/hn_voice_pull.py volume
    python scripts/hn_voice_pull.py comments --per-month 2500
    python scripts/hn_voice_pull.py threads --per-year 60
"""

import argparse
import csv
import datetime as dt
import gzip
import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter

OUT = "data/eval_results/hn_llm_voice"
ALGOLIA = "https://hn.algolia.com/api/v1"
FIREBASE = "https://hacker-news.firebaseio.com/v0"

# HN's first comment is from 2007. 2008 is the first full year with real volume.
FIRST_YEAR = 2008
LAST = dt.datetime(2026, 10, 1)


# Be a good guest. Algolia documents 10,000 requests/hour per IP for the HN
# search API, which is 2.78/s; we cap well under it, per host, and identify
# ourselves so they can get in touch instead of having to block us. HN is a
# free community service and this study is not worth degrading it.
MAX_RPS = 1.2
UA = ("jeffy-hn-voice-study/0.1 (research on comment-style drift over time; "
      "contact: github.com/jeffyclassify)")

_last_call = {}


def _throttle(host):
    wait = _last_call.get(host, 0.0) + 1.0 / MAX_RPS - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_call[host] = time.monotonic()


def api(url, tries=6):
    """GET, rate-limited per host, with backoff that honours Retry-After."""
    host = urllib.parse.urlsplit(url).netloc
    for i in range(tries):
        _throttle(host)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                # back off hard and long; being throttled means we were rude
                delay = float(e.headers.get("Retry-After") or 0) or 30 * (i + 1)
                print(f"  [{e.code} from {host}, sleeping {delay:.0f}s]", flush=True)
                time.sleep(delay)
                continue
            if i == tries - 1:
                raise
            time.sleep(2.0 * (i + 1))
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2.0 * (i + 1))


def months():
    y, m = FIRST_YEAR, 1
    while dt.datetime(y, m, 1) < LAST:
        a = dt.datetime(y, m, 1)
        b = dt.datetime(y + (m == 12), m % 12 + 1, 1)
        yield y, m, int(a.timestamp()), int(b.timestamp())
        y, m = y + (m == 12), m % 12 + 1


def count(tag, a, b):
    """Exact count for a window, or None if Algolia's estimate isn't exhaustive.

    This matters more than it looks. Algolia reports nbHits for any query, but
    only guarantees it below an internal cap: measured on HN, a 1-day window in
    2021 came back exhaustive at 10,578 while a 1-week window reported 63,259
    and a 1-YEAR window reported 76,734, both with exhaustiveNbHits false. The
    counter saturates somewhere around 64k and then lies quietly. Any volume
    chart built on monthly nbHits is fiction, so refuse a non-exhaustive answer
    rather than return it.
    """
    d = api(f"{ALGOLIA}/search_by_date?tags={tag}"
            f"&numericFilters=created_at_i>{a},created_at_i<{b}&hitsPerPage=1")
    return d["nbHits"] if d.get("exhaustiveNbHits") else None


def count_exact(tag, a, b, depth=0):
    """Exact count by bisecting the window until every piece is exhaustive."""
    n = count(tag, a, b)
    if n is not None:
        return n
    if b - a <= 60 or depth > 12:
        return count(tag, a, b) or 0      # give up; a minute of HN is small
    mid = (a + b) // 2
    return count_exact(tag, a, mid, depth + 1) + count_exact(tag, mid, b, depth + 1)


# --------------------------------------------------------------------------- #
# volume
# --------------------------------------------------------------------------- #

def run_volume():
    """Exact item volume per month, from HN's sequential item ids.

    Every HN item, of every type, takes the next integer id. So the oldest id in
    a month, differenced against the oldest id of the next month, is the exact
    number of items created in that month, including the dead and deleted ones
    Algolia will not show. No estimation anywhere.

    Composition (how many of those items are comments) is sampled separately by
    `volume_mix`, because splitting it from Algolia would need exhaustive
    sub-window counts for every month.
    """
    path = f"{OUT}/volume_by_month.csv"
    fields = ["year", "month", "start", "first_item_id", "stories_exact"]
    have = {}
    if os.path.exists(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                have[(int(r["year"]), int(r["month"]))] = r
        print(f"resuming: {len(have)} months already on disk")

    f = open(path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=fields)
    if not have:
        w.writeheader()

    for y, m, a, b in months():
        if (y, m) in have:
            continue
        d = api(f"{ALGOLIA}/search_by_date?tags=(story,comment)"
                f"&numericFilters=created_at_i>{a},created_at_i<{a + 129600}"
                f"&hitsPerPage=1000")
        ids = [int(h["objectID"]) for h in d["hits"]]
        # Stories are few enough per month (tens of thousands) that bisecting to
        # exhaustive windows is cheap, so this column is exact too.
        rec = {"year": y, "month": m, "start": a,
               "first_item_id": min(ids) if ids else "",
               "stories_exact": count_exact("story", a, b)}
        w.writerow(rec)
        f.flush()
        print(f"{y}-{m:02d}  anchor={rec['first_item_id']}  "
              f"stories={rec['stories_exact']:>6}", flush=True)
        time.sleep(0.1)
    f.close()
    print(f"\nwrote {path}")


def run_comment_counts():
    """Exact comments per month, by bisecting to exhaustive windows.

    Slower than asking for a monthly nbHits, and the only version that is true:
    the saturated estimate said 64,007 comments in March 2021 where the real
    figure is 307,073, a factor of 4.8. See API_CAVEATS.md.
    """
    path = f"{OUT}/comments_by_month.csv"
    fields = ["year", "month", "start", "comments_exact"]
    have = set()
    if os.path.exists(path):
        with open(path) as f:
            have = {(int(r["year"]), int(r["month"])) for r in csv.DictReader(f)}
        print(f"resuming: {len(have)} months already on disk")
    f = open(path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=fields)
    if not have:
        w.writeheader()
    for y, m, a, b in months():
        if (y, m) in have:
            continue
        n = count_exact("comment", a, b)
        w.writerow({"year": y, "month": m, "start": a, "comments_exact": n})
        f.flush()
        print(f"{y}-{m:02d}  comments={n:>8,}", flush=True)
    f.close()
    print(f"\nwrote {path}")


def run_volume_mix(per_year=400, seed=42):
    """Sample item ids uniformly per year and ask Firebase what each one is.

    Gives the comment share of all items, with dead and deleted included, which
    is the multiplier that turns exact id deltas into exact comment counts.
    """
    rng = random.Random(seed)
    vol = f"{OUT}/volume_by_month.csv"
    if not os.path.exists(vol):
        raise SystemExit("run `volume` first")
    anchors = {}
    with open(vol) as f:
        for r in csv.DictReader(f):
            if r["first_item_id"]:
                anchors[(int(r["year"]), int(r["month"]))] = int(r["first_item_id"])

    path = f"{OUT}/volume_mix.csv"
    fields = ["year", "sampled", "comment", "story", "pollopt", "job",
              "poll", "missing", "dead", "deleted"]
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {int(r["year"]) for r in csv.DictReader(f)}
    f = open(path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=fields)
    if not done:
        w.writeheader()

    years = sorted({y for y, _ in anchors})
    for y in years:
        if y in done:
            continue
        lo = min(v for (yy, _), v in anchors.items() if yy == y)
        nxt = [v for (yy, _), v in anchors.items() if yy == y + 1]
        hi = min(nxt) if nxt else lo + 1
        if hi <= lo:
            continue
        c = Counter()
        for _ in range(per_year):
            it = api(f"{FIREBASE}/item/{rng.randrange(lo, hi)}.json")
            if not it:
                c["missing"] += 1
                continue
            c[it.get("type") or "missing"] += 1
            if it.get("dead"):
                c["dead"] += 1
            if it.get("deleted"):
                c["deleted"] += 1
        rec = {"year": y, "sampled": per_year}
        rec.update({k: c.get(k, 0) for k in fields[2:]})
        w.writerow(rec)
        f.flush()
        share = c["comment"] / per_year
        print(f"{y}  ids {lo}-{hi}  comment share {share:.3f}  "
              f"dead {c['dead']/per_year:.3f}  missing {c['missing']/per_year:.3f}",
              flush=True)
    f.close()
    print(f"\nwrote {path}")


# --------------------------------------------------------------------------- #
# comments
# --------------------------------------------------------------------------- #

KEEP = ("objectID", "created_at_i", "author", "story_id", "parent_id",
        "story_title", "comment_text")


def shrink_window(center, want=900):
    """Find a window around `center` holding <= `want` comments, and return it.

    Starts from a wide guess and halves. Returns (a, b, nbHits) or None if even
    a two-minute window is too busy, which only happens if HN gets very loud.
    """
    half = 6 * 3600
    for _ in range(14):
        a, b = center - half, center + half
        n = count("comment", a, b)
        if n <= want:
            if n >= want // 6 or half <= 120:
                return a, b, n
            half *= 2          # too quiet, widen to avoid wasting a request
        else:
            half //= 2
        if half < 60:
            return None
    return None


def run_comments(per_month, seed=42):
    import datetime as dt
    rng = random.Random(seed)
    path = f"{OUT}/comments_sample.jsonl.gz"
    done_path = f"{OUT}/comments_done.json"

    done = json.load(open(done_path)) if os.path.exists(done_path) else {}
    seen = set()
    if os.path.exists(path):
        # A killed process leaves the gzip truncated mid-block, and reading it
        # then raises EOFError partway through. Keep every record that survived,
        # then rewrite the file from those so the tail is a clean stream again;
        # appending to a truncated member would corrupt everything after it.
        recs, truncated = [], False
        try:
            with gzip.open(path, "rt") as f:
                for line in f:
                    try:
                        recs.append(json.loads(line))
                    except Exception:
                        pass
        except (EOFError, OSError) as e:
            truncated = True
            print(f"  [truncated archive: {e.__class__.__name__}, repairing]")
        seen = {r["objectID"] for r in recs}
        if truncated:
            tmp = path + ".repair"
            with gzip.open(tmp, "wt") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            os.replace(tmp, path)
            # the done map counted comments that may not have reached disk, so
            # trust the archive over it and let short months top themselves up
            per = Counter()
            for r in recs:
                d = dt.datetime.utcfromtimestamp(r["created_at_i"])
                per[f"{d.year}-{d.month:02d}"] += 1
            done = {k: min(v, done.get(k, v)) for k, v in per.items()}
            json.dump(done, open(done_path, "w"), indent=1)
        print(f"resuming: {len(seen)} comments already on disk")

    out = gzip.open(path, "at")
    for y, m, a, b in months():
        key = f"{y}-{m:02d}"
        if done.get(key, 0) >= per_month:
            continue
        got = done.get(key, 0)
        tries = 0
        while got < per_month and tries < 40:
            tries += 1
            w = shrink_window(rng.randrange(a, b))
            if not w:
                continue
            wa, wb, n = w
            d = api(f"{ALGOLIA}/search_by_date?tags=comment"
                    f"&numericFilters=created_at_i>{wa},created_at_i<{wb}"
                    f"&hitsPerPage=1000")
            new = 0
            for h in d["hits"]:
                if h["objectID"] in seen or not h.get("comment_text"):
                    continue
                seen.add(h["objectID"])
                rec = {k: h.get(k) for k in KEEP}
                # parent_id == story_id means the comment replies to the
                # submission itself, so it sits at depth 1. Cheap, and it comes
                # free with the search hit; deeper positions need `threads`.
                rec["is_top_level"] = (h.get("parent_id") == h.get("story_id"))
                rec["window"] = [wa, wb]
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                new += 1
            got += new
            time.sleep(0.1)
        out.flush()
        done[key] = got
        json.dump(done, open(done_path, "w"), indent=1)
        print(f"{key}  +{got:>5} comments  (total on disk {len(seen)})",
              flush=True)
    out.close()
    print(f"\nwrote {path}  ({len(seen)} comments)")


# --------------------------------------------------------------------------- #
# threads
# --------------------------------------------------------------------------- #

def walk(story_id):
    """Return one row per comment in a story, with depth and sibling rank.

    Algolia's /items/<id> returns the whole nested tree in a single request,
    children in HN's own display order. That order is the ranked order, so a
    comment's index among its siblings is its rank, and depth 1 means a direct
    reply to the submission. Doing this through Firebase instead would cost one
    request per comment, roughly 170k requests for the sample we want.
    """
    root = api(f"{ALGOLIA}/items/{story_id}")
    if not root:
        return [], None
    rows = []

    def rec(node, depth):
        for i, c in enumerate(node.get("children") or []):
            if c.get("type") == "comment":
                rows.append({
                    "objectID": str(c.get("id")), "story_id": str(story_id),
                    "created_at_i": c.get("created_at_i"),
                    "author": c.get("author"),
                    "depth": depth, "sibling_rank": i,
                    "n_kids": len(c.get("children") or []),
                    # author None with text None is HN's shape for a dead or
                    # deleted comment; it still occupies a tree position
                    "dead_or_deleted": c.get("author") is None,
                    "comment_text": c.get("text") or "",
                })
            rec(c, depth + 1)

    rec(root, 1)
    meta = {"story_id": str(story_id), "title": root.get("title"),
            "score": root.get("points"), "created_at_i": root.get("created_at_i"),
            "author": root.get("author"), "url": root.get("url"),
            "n_comments": len(rows),
            "max_depth": max((r["depth"] for r in rows), default=0)}
    return rows, meta


def run_threads(per_year, seed=42):
    rng = random.Random(seed)
    cpath, spath = f"{OUT}/thread_comments.jsonl.gz", f"{OUT}/thread_stories.jsonl"
    seen = set()
    if os.path.exists(spath):
        with open(spath) as f:
            seen = {json.loads(l)["story_id"] for l in f if l.strip()}
        print(f"resuming: {len(seen)} stories already walked")

    cf, sf = gzip.open(cpath, "at"), open(spath, "a")
    per_month = max(1, per_year // 12)
    for y, m, a, b in months():
        # Front-page-sized threads: where the conversation, and any bot
        # attention, actually lands. 40 points is roughly the front page floor.
        # Sampled per month rather than per year, because /search ranks by
        # points and a yearly pool would be all mega-threads.
        d = api(f"{ALGOLIA}/search_by_date?tags=story&numericFilters="
                f"created_at_i>{a},created_at_i<{b},points>40,num_comments>20"
                f"&hitsPerPage=1000")
        pool = [h["objectID"] for h in d["hits"] if h["objectID"] not in seen]
        rng.shuffle(pool)
        got = 0
        for sid in pool[:per_month]:
            rows, meta = walk(sid)
            if not meta:
                continue
            for r in rows:
                cf.write(json.dumps(r, ensure_ascii=False) + "\n")
            sf.write(json.dumps(meta, ensure_ascii=False) + "\n")
            seen.add(sid)
            got += 1
            cf.flush(); sf.flush()
            time.sleep(0.1)
        print(f"{y}-{m:02d}: pool={len(pool)} walked={got} "
              f"(total stories {len(seen)})", flush=True)
    cf.close(); sf.close()
    print(f"\nwrote {cpath} and {spath}  ({len(seen)} stories)")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["volume", "volume_mix", "comment_counts",
                                    "comments", "threads"])
    # 5000/month is set by the marker base rates, not the vocabulary. Unigram
    # frequencies are stable at a fraction of this, but the tells people
    # actually named in the thread run at 0.2% to 2.4% of comments, so measuring
    # a change in them needs tens of thousands of comments per year.
    p.add_argument("--per-month", type=int, default=5000)
    p.add_argument("--per-year", type=int, default=60)
    args = p.parse_args()
    if args.mode == "volume":
        run_volume()
    elif args.mode == "volume_mix":
        run_volume_mix(args.per_year)
    elif args.mode == "comment_counts":
        run_comment_counts()
    elif args.mode == "comments":
        run_comments(args.per_month)
    else:
        run_threads(args.per_year)
