# HN API measurement traps, found while building this study

All verified 2026-10-29 against the live APIs. Each of these silently returns a
plausible wrong number rather than an error, so anything published off this
corpus should be checked against this list first.

## 1. `nbHits` saturates and stops being exact, without saying so in the number

Algolia reports `nbHits` for any query but only guarantees it below an internal
cap. The honest field is `exhaustiveNbHits`, which nothing forces you to read.

Comments in March 2021, same window start, growing width:

| window | nbHits | exhaustiveNbHits |
|---|---|---|
| 1 hour | 260 | true |
| 6 hours | 1,871 | true |
| 1 day | 10,578 | true |
| 1 week | 63,259 | **false** |
| 1 month | 64,007 | **false** |
| 1 year | 76,734 | **false** |

A one-year window reporting 76,734 while one week reports 63,259 is the tell.
The counter saturates somewhere between 10.5k and 63k.

Below the cap it is exact to the unit: 600s window returned nbHits 33 and
enumerated 33; 1800s returned 104 and 104; 3600s returned 260 and 260.

**What we do instead:** count by bisecting windows until every piece reports
`exhaustiveNbHits: true` (`count_exact` in `hn_voice_pull.py`), and take total
volume from HN's sequential item ids rather than from any count API.

**Numbers this invalidated:** an earlier pass of this study reported 523,905
comments in one week of 2026 and monthly comment counts for the whole archive.
All of it was the saturated estimate. Discarded.

## 2. Item ids are sequential across all item types, which gives exact volume free

Every HN item, of every type, takes the next integer. So the oldest id in a
month differenced against the oldest id of the next month is the exact number of
items created in that month, **including dead and deleted items that Algolia
will not return**. No estimation. Composition (what share are comments) is a
separate uniform sample of ids resolved through Firebase.

## 3. Default text search does prefix and typo matching, so term counts are fiction

Comments in 2021:

| query | default | with `typoTolerance=false` + `restrictSearchableAttributes=comment_text` |
|---|---|---|
| `tapestry` | 68 (non-exhaustive) | 67 (exhaustive) |
| `"tapestry"` | 55 (exhaustive) | 55 (exhaustive) |
| `tapestr` | **6,543** | 68 |
| `tapestryy` | **68** | 0 |

A truncated word returning 6,543 and a misspelled word returning the same count
as the correct one are both prefix/typo behaviour. Strict parameters fix the
typo case but prefix matching on the final token survives, which is why
`tapestr` still returns 68.

## 4. Search normalises inflections, so you cannot count word forms with it

| query | nbHits | word actually highlighted in the hit |
|---|---|---|
| `delve` | 488 | delve |
| `delves` | 379 | **delve** |
| `delving` | 155 | delving |

Querying `delves` returns hits highlighting `delve`. Singular and plural are
conflated, so per-inflection frequency cannot come from the search endpoint.
This is why excess-vocabulary counting must run over retrieved comment text with
our own tokenizer, not off `nbHits`.

Multi-word behaviour: `load-bearing` and `"load bearing"` both return 109, while
unquoted `load bearing` returns 142, so unquoted multi-word is an AND of terms
rather than a phrase.

## 5. `/items/<story_id>` returns the whole comment tree in one request

Walking a thread through Firebase costs one request per comment, roughly 170k
requests for the sample this study wants. The Algolia items endpoint returns the
full nested tree in a single call, children in HN's display order, so depth and
sibling rank come straight out of the nesting. Verified on the Fowler thread
(50290177): 163 comments, max depth 6, depth histogram
`{1:48, 2:56, 3:36, 4:16, 5:5, 6:2}`.

Comment `points` is null throughout, as expected since HN hides comment scores.
Depth and sibling rank are the only available ranking proxy.

## 6. Sampling a time window newest-first is biased, so windows must be complete

`search_by_date` returns at most 1000 hits, newest first. Requesting a week
therefore returns the last 1000 comments of that week, biased toward whatever
hour it ended in. The sampler instead picks a random instant and shrinks a
window around it until it holds under 1000, then takes every comment in it, so
each window is a complete cluster rather than a truncated page.

Measured window widths needed to stay under 900 comments, which also tracks how
much busier HN has become: 2010 ~360 min, 2016 ~90 min, 2021 ~180 min, 2026 ~45
min.

## 7. Marker base rates are low, which sets the sample size

Share of comments matching, in one 452-comment window from June 2026:

```
triadic_list  2.4%    em_dash         1.1%    double_hyphen  0.4%
not_x_but_y   0.2%    the_reality_is  0.2%    en_dash        0.2%
tapestry      0.0%    load_bearing    0.0%    inanimate_verbing 0.0%
```

At a 1% document rate, measuring a change needs tens of thousands of comments
per year, not thousands. Markers rarer than roughly 0.1% are below what this
corpus can measure and should be reported as such rather than charted as zero.
