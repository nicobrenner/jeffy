"""Classical lossless codecs on the same 756x504 Kodak crops used for the shared prior.

PNG (PIL optimize), WebP lossless (method 6, exact), JPEG 2000 lossless (OpenJPEG), JPEG XL lossless (cjxl -d 0, effort 7 and 9).
Every output is decoded and compared bit-exact before its size is counted.
Prints mean bpp (all 3 channels) and bits/sub-pixel, and the per-image table; saves pile_runs/rope_byte/kodak_codecs.json.
"""

import io
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

P = 28
KODAK = sorted(Path("data/kodak").glob("kodim*.png"))
OUT = Path("pile_runs/rope_byte/kodak_codecs.json")


def crop(arr):
    H, W = arr.shape[:2]
    return arr[: (H // P) * P, : (W // P) * P]


def pil_codec(arr, fmt, **kw):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt, **kw)
    back = np.array(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
    assert np.array_equal(back, arr), f"{fmt} not lossless"
    return len(buf.getvalue())


def jxl(arr, effort):
    with tempfile.TemporaryDirectory() as d:
        src, dst, dec = Path(d) / "in.png", Path(d) / "out.jxl", Path(d) / "dec.png"
        Image.fromarray(arr).save(src, format="PNG")
        subprocess.run(["cjxl", str(src), str(dst), "-d", "0", "-e", str(effort), "--quiet"], check=True, capture_output=True)
        subprocess.run(["djxl", str(dst), str(dec), "--quiet"], check=True, capture_output=True)
        assert np.array_equal(np.array(Image.open(dec).convert("RGB")), arr), "jxl not lossless"
        return dst.stat().st_size


CODECS = {
    "PNG (PIL optimize)": lambda a: pil_codec(a, "PNG", optimize=True),
    "WebP lossless (m6)": lambda a: pil_codec(a, "WEBP", lossless=True, quality=100, method=6, exact=True),
    "JPEG 2000 lossless": lambda a: pil_codec(a, "JPEG2000", irreversible=False),
    "JPEG XL lossless e7": lambda a: jxl(a, 7),
    "JPEG XL lossless e9": lambda a: jxl(a, 9),
}


def main():
    t0 = time.time()
    rows = []
    for p in KODAK:
        arr = crop(np.array(Image.open(p).convert("RGB")))
        npx = arr.shape[0] * arr.shape[1]
        row = {"image": p.name}
        for name, fn in CODECS.items():
            row[name] = fn(arr) * 8 / npx
        rows.append(row)
        print(f"{p.name}: " + "  ".join(f"{k.split(' ')[0]}{'-' + k.split(' ')[-1] if 'XL' in k else ''} {row[k]:5.2f}" for k in CODECS), flush=True)
    print(f"\n{'codec':<22}{'mean bpp':>10}{'bits/sub-pixel':>16}{'vs PNG':>9}")
    means = {k: float(np.mean([r[k] for r in rows])) for k in CODECS}
    png = means["PNG (PIL optimize)"]
    for k, v in means.items():
        print(f"{k:<22}{v:>10.3f}{v / 3:>16.3f}{v / png:>9.3f}x")
    OUT.write_text(json.dumps(dict(means=means, per_image=rows), indent=1))
    print(f"\nSaved {OUT}. Total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
