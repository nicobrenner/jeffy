"""Teacher-forced argmax reconstructions of unseen images using the frozen shared prior (model D)."""

import math

import numpy as np
import torch
from PIL import Image

from train_rope_byte_2d import RoPE2DByteLM

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
PHOTO_DIR = "pile_runs/rope_byte/photo_compressor"
P = 28
IMAGES = ["shared_prior_pluto_nasa_700", "shared_prior_sample_rgb_448x364"]


@torch.no_grad()
def reconstruct(model, arr):
    H, W, _ = arr.shape
    out = np.zeros_like(arr)
    for ch in range(3):
        for r in range(H // P):
            for c in range(W // P):
                px = arr[r * P:(r + 1) * P, c * P:(c + 1) * P, ch].flatten()
                ids = torch.tensor(px, dtype=torch.long).unsqueeze(0).to(DEVICE)
                pred = model(ids)[0, :-1].argmax(-1).cpu().numpy()
                out[r * P:(r + 1) * P, c * P:(c + 1) * P, ch] = np.concatenate([[px[0]], pred]).reshape(P, P)
    return out


def main():
    ckpt = torch.load(f"{PHOTO_DIR}/color_700_shared_D.pt", map_location=DEVICE, weights_only=False)
    model = RoPE2DByteLM(vocab_size=256, d_model=32, n_heads=2, n_layers=2, height=P, width=P, bidirectional=False).to(DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    for name in IMAGES:
        arr = np.array(Image.open(f"{PHOTO_DIR}/{name}.png").convert("RGB"))
        rec = reconstruct(model, arr)
        mse = np.mean((rec.astype(float) - arr.astype(float)) ** 2)
        psnr = 10 * math.log10(255 ** 2 / mse) if mse > 0 else float("inf")
        out = f"{PHOTO_DIR}/{name}_recon_D.png"
        Image.fromarray(rec).save(out)
        print(f"{name}: PSNR {psnr:.1f} dB -> {out}", flush=True)


if __name__ == "__main__":
    main()
