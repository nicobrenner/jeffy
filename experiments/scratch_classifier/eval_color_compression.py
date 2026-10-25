"""Evaluate color photo compression with proper metric.

For single-image compression (COIN-style), the model weights ARE the
compressed representation. The correct metric is:
  total_bits = model_size_bits + arithmetic_coded_bits
  bpp = total_bits / total_pixels
Not val_loss, which is meaningless when overfitting is the goal.
"""

import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from train_rope_byte_2d import RoPE2DByteLM

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
PHOTO_DIR = "pile_runs/rope_byte/photo_compressor"


def main():
    img = Image.open(f"{PHOTO_DIR}/color_700_original.png")
    arr = np.array(img)
    H, W, C = arr.shape
    print(f"Image: {H}x{W}x{C}")

    png_bytes = os.path.getsize(f"{PHOTO_DIR}/color_700_original.png")
    png_bpp = (png_bytes * 8) / (H * W)
    print(f"PNG: {png_bytes:,} bytes = {png_bpp:.2f} bpp")

    pH, pW = H // 28, W // 28
    covered_pixels = pH * 28 * pW * 28

    total_model_bits = 0
    total_coded_bits = 0

    for ch, ch_name in enumerate(["R", "G", "B"]):
        print(f"\n--- Channel {ch_name} ---")
        ch_arr = arr[:, :, ch]

        patches = []
        for r in range(pH):
            for c in range(pW):
                patch = ch_arr[r * 28 : (r + 1) * 28, c * 28 : (c + 1) * 28].flatten()
                patches.append(torch.tensor(patch, dtype=torch.long))

        n_patches = len(patches)

        ckpt = torch.load(
            f"{PHOTO_DIR}/color_700_{ch_name}.pt",
            map_location=DEVICE,
            weights_only=False,
        )
        model = RoPE2DByteLM(
            vocab_size=256, d_model=64, n_heads=4, n_layers=4,
            height=28, width=28, bidirectional=False,
        ).to(DEVICE)
        model.load_state_dict(ckpt["model_state"], strict=False)

        n_params = sum(p.numel() for p in model.parameters())
        model_bits_f16 = n_params * 16
        print(f"  Model: {n_params:,} params = {model_bits_f16 // 8:,} bytes (f16)")

        model.eval()
        total_ce = 0.0
        total_tokens = 0
        with torch.no_grad():
            for patch in patches:
                ids = patch.unsqueeze(0).to(DEVICE)
                logits = model(ids)
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, 256),
                    ids[:, 1:].reshape(-1),
                    reduction="sum",
                )
                total_ce += loss.item()
                total_tokens += ids.shape[1] - 1

        avg_ce = total_ce / total_tokens
        coded_bits = total_ce / math.log(2)

        print(f"  Avg CE: {avg_ce:.4f} nats = {avg_ce / math.log(2):.4f} bits/byte")
        print(f"  Coded: {coded_bits:,.0f} bits = {coded_bits / 8:,.0f} bytes")
        ch_total = model_bits_f16 + coded_bits
        print(f"  Channel total (f16 + coded): {ch_total / 8:,.0f} bytes")

        total_model_bits += model_bits_f16
        total_coded_bits += coded_bits

    total_bits = total_model_bits + total_coded_bits
    our_bpp = total_bits / covered_pixels
    our_bytes = total_bits / 8

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Model weights (3 channels, f16): {total_model_bits / 8:,.0f} bytes ({total_model_bits / 8 / 1024:.1f} KB)")
    print(f"Arithmetic coded residuals:      {total_coded_bits / 8:,.0f} bytes ({total_coded_bits / 8 / 1024:.1f} KB)")
    print(f"Our total:                       {our_bytes:,.0f} bytes ({our_bytes / 1024:.1f} KB) = {our_bpp:.2f} bpp")
    print(f"PNG:                             {png_bytes:,} bytes ({png_bytes / 1024:.1f} KB) = {png_bpp:.2f} bpp")
    winner = "our method" if our_bpp < png_bpp else "PNG"
    ratio = max(our_bpp, png_bpp) / min(our_bpp, png_bpp)
    print(f"Winner: {winner} ({ratio:.2f}x)")


if __name__ == "__main__":
    main()
