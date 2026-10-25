"""Where do the bits go? Per-pixel coding cost (bits, summed over R,G,B) of a shared prior on Kodak images,
rendered next to the image. Lossless codecs have no 'output image' to show; this is the informative picture.

Usage: python3 viz_prior_bits.py checkpoints/prior_full_d64_best.pt kodim13 kodim07 kodim20
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from shared_prior_model import CONFIGS, PriorLM
from train_shared_prior import as_ids

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/prior_full_d64_best.pt"
NAMES = sys.argv[2:] or ["kodim13", "kodim07", "kodim20"]
P = 28
OUT = Path("pile_runs/rope_byte") / f"kodak_bits_{Path(CKPT).stem}.png"


@torch.no_grad()
def pixel_bits(m, tiles, bs=16):
    """(n_tiles, 28, 28) bits per pixel summed over channels."""
    out = np.zeros((len(tiles), P, P), np.float32)
    grid, nch = m.grid, m.n_ch
    for i in range(0, len(tiles), bs):
        ids = as_ids(m, tiles[i:i + bs], DEVICE)
        logp = F.log_softmax(m(ids)[:, :-1].float(), dim=-1)
        nll = (-logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1) / 0.6931471805599453).cpu().numpy()
        nll = np.concatenate([np.full((len(ids), 1), 8.0 if m.is_patch[0] else 0.0, np.float32), nll], 1)
        if not m.interleave:
            nll = nll.reshape(-1, 3, grid * grid).sum(1)  # channels were separate sequences
        else:
            nll = nll.reshape(len(ids), grid * grid, 3).sum(2)
        nll = nll.reshape(-1, grid, grid)
        out[i:i + bs] = nll[:, 1:, 1:] if m.l_context else nll
    return out


def main():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    m = PriorLM(**ck["arch"], **CONFIGS[ck["config"]])
    m.load_state_dict(ck["model_state"])
    m = m.to(DEVICE).eval()
    kodak = np.load("data/prior_kodak.npz")
    tiles, names, (H, W) = kodak["tiles"], list(kodak["names"]), kodak["hw"]
    gh, gw = H // P, W // P

    fig, axes = plt.subplots(len(NAMES), 2, figsize=(11, 3.9 * len(NAMES)))
    axes = np.atleast_2d(axes)
    for row, name in enumerate(NAMES):
        k = names.index(f"{name}.png")
        img = np.array(Image.open(f"data/kodak/{name}.png").convert("RGB"))[:H, :W]
        b = pixel_bits(m, tiles[k])
        bits_img = b.reshape(gh, gw, P, P).transpose(0, 2, 1, 3).reshape(H, W)
        bpp = bits_img.mean()
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"{name}  ({bpp:.2f} bpp, PNG {kodak['png_bpp'][k]:.2f})")
        im = axes[row, 1].imshow(bits_img, cmap="magma", vmin=0, vmax=np.percentile(bits_img, 99))
        axes[row, 1].set_title("bits per pixel (R+G+B), model surprise")
        plt.colorbar(im, ax=axes[row, 1], fraction=0.03)
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle(f"Where the bits go — {Path(CKPT).stem} ({m.n_params():,} params)")
    fig.tight_layout()
    fig.savefig(OUT, dpi=110)
    print(f"Saved {OUT.resolve()}")


if __name__ == "__main__":
    main()
