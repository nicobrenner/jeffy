"""Training curve of a shared prior (Kodak bpp per epoch) with codec reference lines and a fitted projection.

Fits bpp(e) = a + b * e^(-c) (power-law approach to an asymptote) on the observed epochs and extrapolates,
reporting the asymptote and the epoch at which each codec line would be crossed (or 'never' if below the asymptote).

Usage: python3 plot_prior_curve.py full_d64_20ep [more tags...]
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

RES, RES3 = {}, {}
for f in sorted(Path("pile_runs/rope_byte").glob("shared_prior_results*.json")):
    RES.update(json.load(open(f)))
CODECS = json.load(open("pile_runs/rope_byte/kodak_codecs.json"))["means"]
TAGS = sys.argv[1:] or ["full_d64_20ep"]
OUT = Path("pile_runs/rope_byte/prior_training_curve.png")
REF = {"PNG": CODECS["PNG (PIL optimize)"], "WebP lossless": CODECS["WebP lossless (m6)"], "JPEG XL (e9)": CODECS["JPEG XL lossless e9"]}


def power(e, a, b, c):
    return a + b * np.power(e, -c)


def main():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    horizon = 100
    for tag in TAGS:
        r = {**RES3, **RES}[tag]
        ep = np.array([x["epoch"] for x in r["epochs"]], float)
        bpp = np.array([x["kodak_bpp"] for x in r["epochs"]])
        ax.plot(ep, bpp, "o-", label=f"{tag} ({r['params']:,} params), measured")
        (a, b, c), _ = curve_fit(power, ep, bpp, p0=(bpp.min() - 0.5, bpp[0] - bpp.min(), 1.0), maxfev=20000)
        ex = np.arange(ep.max(), horizon + 1)
        ax.plot(ex, power(ex, a, b, c), "--", alpha=0.7, label=f"fit a+b·e^-c, asymptote {a:.2f} bpp")
        print(f"{tag}: fit asymptote {a:.3f} bpp (b={b:.2f}, c={c:.2f}); last measured {bpp[-1]:.3f} at epoch {int(ep[-1])}")
        for name, v in REF.items():
            if v <= a:
                print(f"  {name} ({v:.2f}): never at this configuration (asymptote {a:.2f})")
            else:
                e_cross = (b / (v - a)) ** (1 / c)
                print(f"  {name} ({v:.2f}): crossed at epoch ≈ {e_cross:.0f}" + ("  (already)" if e_cross <= ep.max() else ""))
    for name, v in REF.items():
        ax.axhline(v, ls=":", lw=1.2, color="gray")
        ax.text(horizon, v + 0.05, f"{name} {v:.2f}", ha="right", fontsize=9, color="gray")
    ax.set_xlabel("epoch (60K windows ≈ 145M tokens each)")
    ax.set_ylabel("Kodak-24 bpp (lower is better)")
    ax.set_xscale("log")
    ax.set_title("Shared byte-LM prior: Kodak bpp vs training, with projection")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    print(f"Saved {OUT.resolve()}")


if __name__ == "__main__":
    main()
