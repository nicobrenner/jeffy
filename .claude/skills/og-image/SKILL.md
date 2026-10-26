---
name: og-image
description: Create or regenerate an Open Graph / social preview card (og.png) for a jeffyclassify.com page, and wire up the og:/twitter: meta tags. Covers the house visual style, the spec-driven generator at site/tools/og/make_og.py, the vendored IBM Plex Mono fonts, inline SVG charts for page heroes, and how to verify a card before sharing a link. Use whenever a new page or blog post is added, or when published numbers change.
---

# Social cards for jeffyclassify.com

Every card is **1200x630 PNG**, rendered from a **JSON spec** by
`site/tools/og/make_og.py`. Never draw one by hand.

The first seven cards on the site were made ad hoc and the code was thrown
away. When the SetFit and FastText numbers turned out to be misconfigured, the
`when-llms-fail-at-classification` card still showed "SetFit 54%" and "qwen3 3%"
baked into a PNG that nobody could regenerate. **A card is a published claim.
It needs a source file like every other published claim.**

## Making one

```bash
cd ~/work/jeffy/site
python tools/og/make_og.py tools/og/specs/<page-slug>.json
python tools/og/make_og.py tools/og/specs/*.json      # rebuild all after a numbers change
```

Spec lives at `site/tools/og/specs/<page-slug>.json`, output path is relative
to the site root:

```json
{
  "out": "blog/polish-guy-was-right/og.png",
  "title": ["The Polish Guy", "Was Right"],
  "subtitle": "51 languages, one frozen encoder, no GPU, no fine-tuning",
  "accent": "#ff6d3a",
  "kpis": [{"value": "83.3%", "caption": "mean, 51 languages", "color": "orange"}],
  "bars": [{"label": "Polish", "value": 0.8544, "display": "85.4%", "color": "orange"}],
  "bar_range": [0.74, 0.875],
  "bar_note": "MASSIVE intent, 60 classes. Axis starts at 74%",
  "footer_right": "frozen LaBSE + logistic regression"
}
```

- `title` — 1 or 2 lines. Line 2 gets the accent colour. Keep each under ~18
  characters or the fitter will shrink it
- `kpis` — 1 to 3 boxes, the big numbers. More than 3 and they stop reading
- `bars` — up to 7 rows. `value` is the real number, `display` is the printed
  string, so you can show "118ms" or "<1ms" alongside percentages
- `bar_range` — zooms the bar axis so near-identical values are
  distinguishable. **If you use it, say where the axis starts in `bar_note`.**
  A cropped axis with no note is a lie by cropping
- `color` — a `PALETTE` name (`accent`, `orange`, `blue`, `pink`, `dim`...) or
  a hex string. Omit to get the card's accent
- `accent` — override per post, to match that page's own palette

Layout is automatic: kpis left and bars right when both are present, otherwise
whichever is there fills the width. Every text run is auto-shrunk to fit, which
the hand-made cards were not (the old benchmark card clipped its own footer).

## House style

Dark `#0a0a0c` with a faint 40px grid, IBM Plex Mono throughout, Jeffy green
`#22c99a` as the default accent, KPI boxes as rounded `#121216` panels with a
`#2a2a30` border, and a footer rule with `jeffyclassify.com` left and a
one-line technical signature right.

Fonts are **vendored** at `site/tools/og/fonts/` (IBM Plex Mono
Regular/Medium/SemiBold/Bold, OFL-1.1, licence alongside). They used to live
only in a session scratchpad, which is the other reason no card could be
regenerated. Do not reach for a system font; DejaVu is not the house face.

## Wiring the page

An og-less page falls back to the generic `/blog/og.png`, which is the site
logo card and says nothing. Give every real page its own:

```html
<meta property="og:image" content="https://jeffyclassify.com/<path>/og.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="<the card's claim, in words>">
<meta name="twitter:image" content="https://jeffyclassify.com/<path>/og.png">
<meta name="twitter:card" content="summary_large_image">
```

`og:image:alt` is not decoration: it is the only version of the card a screen
reader or a text-only client gets.

## Hero charts inside the page

Different job, same rule: generate from the data file, never hand-draw.
`site/tools/og/make_lang_chart.py` renders the 51-language chart on the Polish
post as **static inline SVG** read straight from
`data/eval_results/labse_51_results.csv`, so the chart cannot drift from the
numbers in the prose.

Static SVG rather than Chart.js when the chart is above the fold: it paints with
the page and needs no CDN. Use Chart.js only where interaction earns it (the
`/benchmark/` page).

Two things that bite:

- **SVG text scales with the viewBox.** A 10px label at a 720-wide viewBox is
  ~5px on a 390px phone. Keep provenance and source lines in the HTML
  `<figcaption>`, not inside the SVG
- The Polish post has a full-page `#flock` canvas at `z-index: -1`, and every
  direct child of `body` that holds content is promoted with
  `position: relative`. A new `<figure>` must be added to that selector list or
  it renders behind the flock

## Verify before sharing

```bash
# dimensions and that it is not a stale copy
python -c "from PIL import Image; im=Image.open('blog/<slug>/og.png'); print(im.size, im.mode)"

# the deployed card, not the local one
curl -sI https://jeffyclassify.com/blog/<slug>/og.png | head -3
```

Then preview the unfurl with <https://www.opengraph.xyz/> or Slack's own
debugger. **Cards are cached hard by every platform.** Facebook, LinkedIn and
Slack keep them for days, so a card shared with wrong numbers stays wrong long
after the fix is deployed. Get it right before the first share, and if you must
replace one, change the filename (`og-v2.png`) rather than the bytes.

## When published numbers change

Grep for the number across the site and fix every surface, cards included:

```bash
cd ~/work/jeffy/site && grep -rn "54%\|0.5425" --include=*.html --include=*.json .
```

Cards carrying numbers to check after the benchmark correction:
`blog/when-llms-fail-at-classification/og.png` (shows SetFit 54%, qwen3 3%),
`blog/jeffy-vs-llm-classification/og.png`, and the `/benchmark/` page charts.
None of these have specs yet; writing them is how they stop being unfixable.
