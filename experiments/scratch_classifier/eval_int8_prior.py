"""C3: how small can the shipped prior get? Quantize model D's weights and re-measure coding cost.

Reports, for f16 / int8 (per-tensor symmetric) / int8 + gzip of the weight bytes:
model bytes, CE-estimated coded bytes on the training photo and on Kodak-style unseen crops, and total.
"""

import gzip
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from train_rope_byte_2d import RoPE2DByteLM

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
PHOTO_DIR = "pile_runs/rope_byte/photo_compressor"
P = 28
IMAGES = {
    "train_photo_700": f"{PHOTO_DIR}/color_700_original.png",
    "pluto_nasa_700": f"{PHOTO_DIR}/shared_prior_pluto_nasa_700.png",
}


def load_model(state):
    m = RoPE2DByteLM(vocab_size=256, d_model=32, n_heads=2, n_layers=2, height=P, width=P, bidirectional=False).to(DEVICE)
    m.load_state_dict(state, strict=False)
    m.eval()
    return m


PARAM_KEYS = {k for k, _ in RoPE2DByteLM(vocab_size=256, d_model=32, n_heads=2, n_layers=2, height=P, width=P, bidirectional=False).named_parameters()}


def quantize_int8(state):
    q_state, payload = {}, bytearray()
    for k, v in state.items():
        if k not in PARAM_KEYS:
            q_state[k] = v
            continue
        scale = v.abs().max().item() / 127.0 or 1.0
        q = torch.round(v / scale).clamp(-127, 127).to(torch.int8)
        q_state[k] = (q.float() * scale).to(v.dtype)
        payload += q.cpu().numpy().tobytes() + np.float16(scale).tobytes()
    return q_state, bytes(payload)


@torch.no_grad()
def ce_bytes(model, arr):
    total = 0.0
    H, W, _ = arr.shape
    for ch in range(3):
        for r in range(H // P):
            for c in range(W // P):
                px = arr[r * P:(r + 1) * P, c * P:(c + 1) * P, ch].flatten()
                ids = torch.tensor(px, dtype=torch.long).unsqueeze(0).to(DEVICE)
                logp = F.log_softmax(model(ids)[0].float(), dim=-1)
                tgt = torch.tensor(px[1:], dtype=torch.long, device=DEVICE)
                total += 8.0 + (-logp[:-1][torch.arange(P * P - 1), tgt] / math.log(2)).sum().item()
    return total / 8


def main():
    ckpt = torch.load(f"{PHOTO_DIR}/color_700_shared_D.pt", map_location=DEVICE, weights_only=False)
    f16_state = {k: v.cpu() for k, v in ckpt["model_state"].items()}
    n_params = sum(v.numel() for k, v in f16_state.items() if k in PARAM_KEYS)
    f16_bytes = n_params * 2
    f16_gz = len(gzip.compress(b"".join(v.to(torch.float16).numpy().tobytes() for k, v in f16_state.items() if k in PARAM_KEYS), 9))

    q_state, q_payload = quantize_int8(f16_state)
    int8_bytes = len(q_payload)
    int8_gz = len(gzip.compress(q_payload, 9))

    imgs = {k: np.array(Image.open(p).convert("RGB")) for k, p in IMAGES.items()}
    print(f"params {n_params:,} | f16 {f16_bytes / 1024:.1f} KB (gz {f16_gz / 1024:.1f}) | int8 {int8_bytes / 1024:.1f} KB (gz {int8_gz / 1024:.1f})\n", flush=True)

    rows = []
    for label, state, mbytes in [("f16", f16_state, f16_bytes), ("int8", q_state, int8_bytes), ("int8+gz", q_state, int8_gz)]:
        m = load_model(state)
        for name, arr in imgs.items():
            coded = ce_bytes(m, arr)
            rows.append((label, name, mbytes, coded))
            print(f"{label:<8}{name:<20} model {mbytes / 1024:6.1f} KB  coded {coded / 1024:7.1f} KB  total {(mbytes + coded) / 1024:7.1f} KB", flush=True)

    base = {n: c for l, n, _, c in rows if l == "f16"}
    print("\nCE penalty from int8 vs f16:")
    for l, n, _, c in rows:
        if l == "int8":
            print(f"  {n}: {(c - base[n]) / 1024:+.2f} KB ({(c / base[n] - 1) * 100:+.2f}%)")


if __name__ == "__main__":
    main()
