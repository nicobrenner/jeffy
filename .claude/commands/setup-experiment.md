# Experiment Setup Checklist

Before launching ANY training run, A/B test, finetuning job, or evaluation script — whether on MPS, rented GPU, or CPU — walk through this checklist. No exceptions.

## 1. Checkpointing & Resume (MANDATORY)

Every training script MUST have before launch:

- [ ] **Per-epoch checkpoint saving** — model state, optimizer state, scheduler state, epoch number, best metric. Save EVERY epoch, not just at the end.
- [ ] **Resume support** — `--resume` flag or automatic detection of existing checkpoint. Must reload all state and continue from where it left off.
- [ ] **Best model saved separately** — `best_model.pt` updated whenever validation improves.
- [ ] **Incremental results saving** — for multi-config experiments, save results after each config completes so partial runs are usable.
- [ ] **Unbuffered output** — always run with `python3 -u` so stdout isn't buffered.
- [ ] **Log file** — redirect output to a log file: `python3 -u script.py > logfile.log 2>&1`

**The fix goes in the code BEFORE launch. Never launch first and patch after.**

## 2. GPU Time Optimization

GPU time (MPS or rented) is the bottleneck. Never waste it.

- [ ] **Pre-tokenize all data** before renting a GPU. Data must be downloadable and ready.
- [ ] **Pre-download model checkpoints** if resuming or finetuning.
- [ ] **Test the script on CPU first** with 1-2 epochs to catch bugs before using GPU time.
- [ ] **Never run CPU-only work on a rented GPU** — data prep, tokenization, analysis, plotting all happen locally.
- [ ] **Never leave a GPU idle** waiting for downloads, data processing, or human decisions.
- [ ] **Download checkpoints DURING training** — don't wait until after. Rented GPUs can be reclaimed.

## 3. Before Renting a GPU

- [ ] Data is tokenized and uploaded (HuggingFace: `nicobrenner/jeffy-pile-tokenized` or equivalent).
- [ ] Training script is tested locally (1-2 epochs on CPU/MPS).
- [ ] Checkpointing verified — kill and resume works.
- [ ] All dependencies documented (use pre-built PyTorch images).
- [ ] Exact launch commands written out and ready to paste.
- [ ] Download/sync commands ready for pulling checkpoints mid-training.
- [ ] **Script selects CUDA when available.** `grep -n "device" script.py` — if it only knows `mps`/`cpu`, fix it before renting. (Oracle v4 and v5-take-1 trained on CPU on rented GPUs because of this.)
- [ ] **Pipeline asserts the GPU is used:** `python3 -c "import torch; assert torch.cuda.is_available()"` on the box, and abort unless the first training log line says `Device: cuda`. Check `nvidia-smi` utilization within 2 minutes of launch.
- [ ] Liveness checks must not self-match: use `pgrep -f '[p]ython3 -u script.py'`, never `pgrep -f script.py`.
- [ ] Remote `nohup` launches over ssh: redirect all three fds AND run the launch ssh in the background with a 30 s kill (`$SSH "... &" & LAUNCH_PID=$!; wait up to 30 s; kill $LAUNCH_PID`). Even with `setsid nohup ... < /dev/null`, the ssh client sometimes never returns (2026-10-24, three times); killing the local client is safe because the remote job is detached. Never edit a bash script while it is running — bash reads it incrementally.
- [ ] Teardown is in a `trap cleanup EXIT`, and the final checkpoint/log download runs **before** it — verify the files landed locally (`ls -la`, load the checkpoint) before the instance is destroyed.

## 4. Before Killing/Pausing/Destroying

- [ ] **Download all checkpoints** — best model, latest checkpoint, and any intermediate saves.
- [ ] **Download all logs** — training logs, evaluation results, any generated output.
- [ ] **Verify downloads** — check file sizes, load a checkpoint to confirm it's valid.
- [ ] **Save the exact command** used to launch — makes resume trivial.
- [ ] **Document where you stopped** — epoch number, metrics at that point.

## 5. Monitoring

- [ ] Log file is being written to (verify with `tail -f`).
- [ ] Process is actually running (`ps aux | grep python`).
- [ ] For multi-day runs: set up periodic checkpoint downloads, not just at the end.

## 6. MPS-Specific (M3 Max)

- [ ] Check what's already running on MPS (`ps aux | grep python`).
- [ ] Don't compete for MPS unless explicitly intended — use `--device cpu` for lightweight jobs.
- [ ] MPS `.detach()` bug: use `torch.no_grad()` context manager instead.

## 7. Experiment Tracking

- [ ] Clear output directory defined before launch.
- [ ] Results saved as JSON for easy comparison.
- [ ] Config saved alongside results (hyperparameters, data paths, git hash if applicable).
- [ ] **Every output path carries the run tag** (`results_<tag>.json`, `<model>_<tag>_best.pt`, `train_<tag>.log`). Two runs must never share a filename.
- [ ] **Result files are load-merge-save, never truncate.** A pipeline pulling from a remote box writes to a tag-specific local path.
- [ ] **Back up anything a run could overwrite** (`cp x.json x_bak_$(date +%Y%m%d).json`) before launch. (2026-10-24: the d128 pull overwrote the d64 20-epoch results; reconstructed from monitor messages.)
- [ ] **After the run:** research-log entry, then `git add` scripts + result JSON + figures + md and commit. Weights, data and run dirs are gitignored; results are not.
