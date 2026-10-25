"""Post 2 figure: validation perplexity vs tokens-per-parameter for byte-level models (text) and MNIST-as-bytes.

Sources (all 2026-10-23, experiments/scratch_classifier/RESEARCH_LOG.md + pile_runs/rope_byte/*/results.json):
  per-checkpoint curves  : d64 RoPE @7,200 t/p (40 points, every 180 t/p); byte d16 no-RoPE 1,800 vs 3,600 (matched table)
  best/final endpoints   : d32 RoPE (6.96@1,529 / 7.06@1,800), d64 RoPE @1,800 (5.46@720 / 5.62@1,800),
                           d128 RoPE (3.70@1,350 / 3.85@1,800), MNIST d32 @1,800 (1.92@1,440 / 1.99@1,800), MNIST d32 @7,200 (1.88@4,860 / 1.90@7,200)
No per-epoch histories were saved for the endpoint runs (checkpoints carry only best_ppl), so those are drawn as best -> final segments.
CPU only (matplotlib Agg).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- validated palette (dataviz reference, light mode) ---
C = {"d16": "#2a78d6", "d32": "#eb6834", "d64": "#1baf7a", "d128": "#eda100"}
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CHINCHILLA = 20

# --- data ---
d64_7200 = [(180, 7.11), (360, 6.01), (540, 6.07), (720, 5.90), (900, 5.86), (1080, 6.15), (1260, 5.84), (1440, 5.81), (1620, 5.99), (1800, 5.91),
            (1980, 6.07), (2160, 5.80), (2340, 5.49), (2520, 5.84), (2700, 5.97), (2880, 5.94), (3060, 5.71), (3240, 5.89), (3420, 6.05), (3600, 5.85),
            (3780, 5.75), (3960, 5.78), (4140, 5.65), (4320, 5.70), (4500, 5.81), (4680, 5.86), (4860, 5.52), (5040, 5.74), (5220, 6.20), (5400, 5.62),
            (5580, 5.80), (5760, 6.01), (5940, 5.79), (6120, 5.35), (6300, 5.65), (6480, 5.85), (6660, 5.80), (6840, 5.86), (7020, 5.74), (7200, 5.87)]
d16_1800 = [(500, 25.81), (1000, 16.85), (1500, 15.46), (1729, 14.67), (1800, 14.72)]
d16_3600 = [(500, 27.99), (1000, 17.00), (1500, 15.40), (1729, 14.42), (1800, 15.04), (2400, 14.29), (3000, 14.27), (3409, 14.19), (3600, 14.23)]
endpoints_text = {  # name: (color, best(t/p, ppl), final(t/p, ppl))
    "d32 RoPE, 58K": (C["d32"], (1529, 6.96), (1800, 7.06)),
    "d64 RoPE, 215K": (C["d64"], (720, 5.46), (1800, 5.62)),
    "d128 RoPE, 824K": (C["d128"], (1350, 3.70), (1800, 3.85)),
}
endpoints_mnist = {
    "MNIST d32, 1,800 t/p": (C["d32"], (1440, 1.920), (1800, 1.989)),
    "MNIST d32, 7,200 t/p": (C["d64"], (4860, 1.881), (7200, 1.897)),
}


XT = [20, 100, 500, 1800, 7200]


def style(ax, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.set_xlim(10, 30000)
    ax.set_xticks(XT); ax.set_xticklabels([f"{v:,}" for v in XT]); ax.set_xticks([], minor=True)
    ax.set_xlabel("tokens per parameter (log scale)", color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2)
    ax.axvline(CHINCHILLA, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2, zorder=1)
    ax.text(0.11, 0.97, "Chinchilla-\noptimal\n≈ 20 t/p", transform=ax.transAxes, color=MUTED, fontsize=8.5, va="top", ha="left")


def mark(ax, color, best, final):
    ax.plot(best[0], best[1], "o", color=color, markersize=8, zorder=3)
    ax.plot(final[0], final[1], "s", markerfacecolor=SURFACE, markeredgecolor=color, markeredgewidth=2, markersize=8, zorder=3)


def segment(ax, color, best, final, label, dy=0):
    ax.plot([best[0], final[0]], [best[1], final[1]], color=color, linewidth=2, zorder=2)
    mark(ax, color, best, final)
    ax.annotate(label, (final[0], final[1]), xytext=(7, dy), textcoords="offset points", color=INK, fontsize=9, va="center")


fig = plt.figure(figsize=(14, 6.2), facecolor=SURFACE)
gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], left=0.055, right=0.985, top=0.83, bottom=0.17, wspace=0.24)
axT = fig.add_subplot(gs[0]); axM = fig.add_subplot(gs[1])

# --- text panel ---
axT.set_yscale("log")
axT.set_ylim(3, 32)
axT.set_yticks([3, 4, 5, 7, 10, 15, 20, 30]); axT.set_yticklabels(["3", "4", "5", "7", "10", "15", "20", "30"]); axT.set_yticks([], minor=True)
style(axT, "validation perplexity (log scale)")
axT.set_title("Text: byte-level models keep improving far past 20 t/p", color=INK, fontsize=11, loc="left")
x, y = zip(*d16_1800); axT.plot(x, y, color=C["d16"], linewidth=2, zorder=2)
x, y = zip(*d16_3600); axT.plot(x, y, color=C["d16"], linewidth=2, linestyle=(0, (5, 2)), zorder=2)
mark(axT, C["d16"], (3409, 14.19), (3600, 14.23))
axT.plot(1729, 14.67, "o", color=C["d16"], markersize=8, zorder=3)
axT.annotate("d16 no-RoPE, 34K params\n1,800 (solid) vs 3,600 (dashed) runs\nfinal ≈ best through 3,600 t/p", (3600, 14.23), xytext=(8, 0), textcoords="offset points", color=INK, fontsize=9, va="center")
x, y = zip(*d64_7200); axT.plot(x, y, color=C["d64"], linewidth=2, linestyle=(0, (5, 2)), zorder=2)
mark(axT, C["d64"], (6120, 5.35), (7200, 5.87))
axT.annotate("d64, 7,200 t/p run (370 epochs): oscillates 5.3–6.2,\nno upward drift; new best 5.35 at 6,120 t/p", (6120, 5.35), xytext=(6, -11), textcoords="offset points", color=INK, fontsize=9, ha="center", va="top")
segment(axT, C["d32"], (1529, 6.96), (1800, 7.06), "d32 RoPE, 58K, 1,800 t/p run: final ≈ best", dy=4)
axT.plot([720, 1800], [5.46, 5.62], color=C["d64"], linewidth=2, zorder=2); mark(axT, C["d64"], (720, 5.46), (1800, 5.62))
axT.annotate("d64 RoPE, 215K, 1,800 t/p run:\nfinal ≈ best", (720, 5.46), xytext=(-9, -3), textcoords="offset points", color=INK, fontsize=9, ha="right", va="center")
segment(axT, C["d128"], (1350, 3.70), (1800, 3.85), "d128 RoPE, 824K, 1,800 t/p run: final ≈ best", dy=-5)

# --- MNIST panel ---
axM.set_ylim(1.84, 2.03)
style(axM, "validation perplexity")
axM.set_title("MNIST as raw bytes: same model class, same behaviour", color=INK, fontsize=11, loc="left")
segment(axM, C["d32"], (1440, 1.920), (1800, 1.989), "MNIST d32, 58K params\n1,800 t/p run: final ≈ best")
segment(axM, C["d64"], (4860, 1.881), (7200, 1.897), "7,200 t/p run\nfinal ≈ best")
axM.annotate("MNIST d32, best 1.88 at 4,860 t/p (243× Chinchilla)\nno overfitting through 7,200 t/p", (4860, 1.881), xytext=(-8, -16), textcoords="offset points", color=INK2, fontsize=8.5, ha="right", va="top")

fig.suptitle("Byte-level models do not overfit under extreme overtraining", color=INK, fontsize=13, x=0.055, y=0.97, ha="left")
fig.text(0.055, 0.885, "●  best checkpoint     □  final checkpoint     ——  per-checkpoint curve     - - -  extended run     ⋯  Chinchilla-optimal tokens/param",
         color=INK2, fontsize=9, va="bottom")
fig.text(0.055, 0.025, "Per-checkpoint curves: d64 @7,200 t/p (every 180 t/p) and d16 1,800 vs 3,600 (matched points). The d32/d64/d128 @1,800 and both MNIST runs saved only best + final; shown as best → final segments.\n"
         "RoPE byte models, L=4, vocab 256; text = 7.7 MB byte corpus (d128: 570 MB), MNIST = 60K images as 784-byte sequences. Chinchilla-optimal ≈ 20 tokens/param.",
         color=INK2, fontsize=8, va="bottom")
fig.savefig("/Users/nico/Documents/Projects/jeffy/experiments/scratch_classifier/pile_runs/rope_byte/overtraining_curve.png", dpi=130, facecolor=SURFACE)
print("saved pile_runs/rope_byte/overtraining_curve.png")
