"""Push heavy experiment artifacts to Hugging Face Hub, where every upload is an immutable commit.

  python3 push_artifacts.py models            # checkpoints/prior_*_best.pt, best_lm_oracle_v*.pt + result JSONs + README table
  python3 push_artifacts.py data              # pre-cut windows (prior_train*.npy), Kodak tiles, chat jsonl, oracle_gen
  python3 push_artifacts.py models --dry-run  # list only

Repos (created on first push; private unless --public):
  models -> nicobrenner/jeffy-byte-priors     (model repo)
  data   -> nicobrenner/jeffy-prior-data      (dataset repo)
Nothing is ever overwritten destructively: HF keeps every revision; re-pushing a changed file creates a new commit.
"""

import argparse
import json
import time
from pathlib import Path

from huggingface_hub import HfApi

USER = "nicobrenner"
REPOS = {"models": (f"{USER}/jeffy-byte-priors", "model"), "data": (f"{USER}/jeffy-prior-data", "dataset")}
ROOT = Path(__file__).resolve().parent

MODEL_FILES = [
    *sorted((ROOT / "checkpoints").glob("prior_*_best.pt")),
    *sorted((ROOT / "checkpoints").glob("best_lm_oracle_v*.pt")),
    *sorted((ROOT / "checkpoints").glob("ep_oracle_*.pt")),          # boundary checkpoints (rewind points), 2026-10-26+
    *sorted((ROOT / "checkpoints").glob("grown_*.pt")),              # function-preserving growth inits
    *sorted((ROOT / "checkpoints").glob("verifier_*.pkl")),          # reply verifiers (hidden-state LogReg)
    *sorted((ROOT / "demo").glob("oracle_*_int8.onnx")),             # what the demo serves
    *sorted((ROOT / "pile_runs/rope_byte").glob("shared_prior_results*.json")),
    *sorted((ROOT / "pile_runs/rope_byte").glob("kodak_*.json")),
    *sorted((ROOT / "pile_runs/rope_byte").glob("oracle_depth_*.json")),
    ROOT / "pile_runs/rope_byte/photo_compressor/color_700_shared_D.pt",
]
DATA_FILES = [
    *sorted((ROOT / "data").glob("prior_train*.npy")),
    *sorted((ROOT / "data").glob("prior_kodak*.npz")),
    *sorted((ROOT / "data").glob("chat_v*.jsonl")),
    *sorted((ROOT / "data").glob("oracle_v*.jsonl")),
    *sorted((ROOT / "data/oracle_gen").glob("*.jsonl")),
]


def readme_models():
    rows = []
    for f in sorted((ROOT / "pile_runs/rope_byte").glob("shared_prior_results*.json")):
        for tag, v in json.load(open(f)).items():
            rows.append((tag, v.get("params", "?"), v.get("best_kodak_bpp", float("nan")), len(v.get("epochs", []))))
    lines = ["# Jeffy byte-LM image priors", "", "Shared lossless image priors: a plain causal byte LM over 28x28 tiles (2D RoPE), trained on CLIC-2020, evaluated on Kodak-24.",
             "PNG (optimized) = 13.448 bpp, WebP-lossless = 9.540, JPEG XL (e9) = 8.688 on the same 756x504 crops.", "",
             "| tag | params | best Kodak bpp | epochs |", "|---|---:|---:|---:|"]
    lines += [f"| {t} | {p:,} | {b:.3f} | {e} |" for t, p, b, e in sorted(rows, key=lambda r: r[2])]
    lines += ["", "Code: `experiments/scratch_classifier/` in the jeffy repo (`shared_prior_model.py`, `train_shared_prior.py`, `bytelm_codec.py`).",
              f"Pushed {time.strftime('%Y-%m-%d %H:%M')}."]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=REPOS)
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    repo, rtype = REPOS[args.what]
    files = [f for f in (MODEL_FILES if args.what == "models" else DATA_FILES) if f.exists()]
    total = sum(f.stat().st_size for f in files)
    print(f"{repo} ({rtype}, {'public' if args.public else 'private'}): {len(files)} files, {total / 1048576:.0f} MB")
    for f in files:
        print(f"  {f.stat().st_size / 1048576:8.1f} MB  {f.relative_to(ROOT)}")
    if args.dry_run:
        return
    api = HfApi()
    api.create_repo(repo, repo_type=rtype, private=not args.public, exist_ok=True)
    ops = [(f, str(f.relative_to(ROOT))) for f in files]
    if args.what == "models":
        readme = ROOT / "pile_runs/rope_byte/README_hf.md"
        readme.write_text(readme_models())
        ops.append((readme, "README.md"))
    for f, dest in ops:
        api.upload_file(path_or_fileobj=str(f), path_in_repo=dest, repo_id=repo, repo_type=rtype,
                        commit_message=f"add {dest} ({f.stat().st_size / 1048576:.1f} MB)")
        print(f"  pushed {dest}")
    print(f"done: https://huggingface.co/{'datasets/' if rtype == 'dataset' else ''}{repo}")


if __name__ == "__main__":
    main()
