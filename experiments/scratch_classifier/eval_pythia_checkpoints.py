"""
Evaluate Pythia checkpoints on our validation data for direct comparison.
Downloads Pythia models at step 1000 (2.1B tokens) and computes val_ppl
on the same validation split we use for training.
"""
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPTNeoXForCausalLM, AutoTokenizer


def eval_model(model, val_path, seq_len, batch_size, device, max_batches=200):
    data = np.memmap(val_path, dtype=np.uint16, mode='r')
    n_chunks = len(data) // seq_len
    total_loss = 0.0
    n_batches = 0

    model.eval()
    with torch.no_grad():
        batch = []
        for i in range(min(n_chunks, max_batches * batch_size)):
            start = i * seq_len
            chunk = torch.from_numpy(data[start:start + seq_len].astype(np.int64))
            batch.append(chunk)

            if len(batch) == batch_size:
                input_ids = torch.stack(batch).to(device)
                outputs = model(input_ids)
                logits = outputs.logits
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    input_ids[:, 1:].reshape(-1),
                )
                total_loss += loss.item()
                n_batches += 1
                batch = []

                if n_batches >= max_batches:
                    break

    avg_loss = total_loss / max(n_batches, 1)
    ppl = math.exp(min(avg_loss, 20))
    return avg_loss, ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-path", required=True)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-batches", type=int, default=200)
    args = parser.parse_args()

    models_to_eval = [
        ("pythia-14m", "EleutherAI/pythia-14m", "step1000"),
        ("pythia-70m", "EleutherAI/pythia-70m", "step1000"),
        ("pythia-160m", "EleutherAI/pythia-160m", "step1000"),
        ("pythia-410m", "EleutherAI/pythia-410m", "step1000"),
    ]

    print("Evaluating Pythia checkpoints at step 1000 (2.1B tokens)")
    print("Val data: %s" % args.val_path)
    print("Seq len: %d, Batch size: %d, Device: %s" % (args.seq_len, args.batch_size, args.device))
    print("=" * 70)

    results = []
    for name, model_id, revision in models_to_eval:
        print("\nLoading %s (revision=%s)..." % (name, revision))
        t0 = time.time()
        model = GPTNeoXForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype=torch.float32
        ).to(args.device)
        n_params = sum(p.numel() for p in model.parameters())
        load_time = time.time() - t0
        print("  %s params (%.1fM), loaded in %.0fs" % ("{:,}".format(n_params), n_params/1e6, load_time))

        t0 = time.time()
        val_loss, val_ppl = eval_model(
            model, args.val_path, args.seq_len, args.batch_size, args.device, args.max_batches
        )
        eval_time = time.time() - t0
        tok_param = 2.1e9 / n_params

        print("  val_loss=%.4f  val_ppl=%.2f  tok/param=%.1f  (%.0fs)" % (
            val_loss, val_ppl, tok_param, eval_time))

        results.append((name, n_params, val_loss, val_ppl, tok_param))
        del model
        if args.device == "mps":
            torch.mps.empty_cache()

    print("\n" + "=" * 70)
    print("SUMMARY: Pythia models at step 1000 (2.1B tokens seen)")
    print("%-15s %10s %10s %10s %10s" % ("Model", "Params", "tok/param", "val_loss", "val_ppl"))
    print("-" * 55)
    for name, n_params, val_loss, val_ppl, tok_param in results:
        print("%-15s %9.1fM %10.1f %10.4f %10.2f" % (
            name, n_params/1e6, tok_param, val_loss, val_ppl))


if __name__ == "__main__":
    main()
