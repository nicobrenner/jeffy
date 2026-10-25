"""
Evaluate our custom CausalLM on LAMBADA (OpenAI version).
LAMBADA measures last-word prediction: given a passage, predict the final word.
"""
import argparse
import math
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from train_pile_scaling import CausalLM, MODEL_CONFIGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-size", required=True, choices=MODEL_CONFIGS.keys())
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-seq-len", type=int, default=1024)
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model_size]
    vocab_size = 50304

    print("Loading model: %s (%s)" % (args.model_size, cfg["label"]))
    model = CausalLM(
        vocab_size=vocab_size,
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        max_seq_len=args.max_seq_len,
        tie_weights=False,
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    n_params = sum(p.numel() for p in model.parameters())
    print("  %s params (%.1fM)" % ("{:,}".format(n_params), n_params / 1e6))

    model = model.to(args.device).eval()

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    print("Loading LAMBADA (OpenAI)...")
    ds = load_dataset("EleutherAI/lambada_openai", "default", split="test")
    print("  %d examples" % len(ds))

    correct = 0
    total = 0
    total_loss = 0.0
    t0 = time.time()

    with torch.no_grad():
        for i, example in enumerate(ds):
            text = example["text"]
            tokens = tokenizer.encode(text)

            if len(tokens) > args.max_seq_len:
                tokens = tokens[-args.max_seq_len:]

            input_ids = torch.tensor([tokens], device=args.device)

            logits = model(input_ids)

            last_word = text.split()[-1]
            last_word_tokens = tokenizer.encode(" " + last_word)

            if len(last_word_tokens) == 1:
                target_pos = len(tokens) - 1
                target_token = tokens[-1]
                pred_logits = logits[0, target_pos - 1]
                pred_token = pred_logits.argmax().item()

                loss = F.cross_entropy(
                    pred_logits.unsqueeze(0),
                    torch.tensor([target_token], device=args.device),
                )
                total_loss += loss.item()

                if pred_token == target_token:
                    correct += 1
            else:
                n_word_tokens = len(last_word_tokens)
                start_pos = len(tokens) - n_word_tokens
                word_correct = True
                word_loss = 0.0

                for j in range(n_word_tokens):
                    pos = start_pos + j
                    target = tokens[pos]
                    pred_logits = logits[0, pos - 1]
                    pred_token = pred_logits.argmax().item()

                    loss = F.cross_entropy(
                        pred_logits.unsqueeze(0),
                        torch.tensor([target], device=args.device),
                    )
                    word_loss += loss.item()

                    if pred_token != target:
                        word_correct = False

                total_loss += word_loss / n_word_tokens
                if word_correct:
                    correct += 1

            total += 1

            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                acc = correct / total
                ppl = math.exp(min(total_loss / total, 20))
                print("  [%d/%d] acc=%.4f ppl=%.2f (%.1fs)" % (
                    i + 1, len(ds), acc, ppl, elapsed))

    elapsed = time.time() - t0
    acc = correct / total
    avg_loss = total_loss / total
    ppl = math.exp(min(avg_loss, 20))

    print("\n" + "=" * 60)
    print("LAMBADA Results: %s" % args.model_size)
    print("  Accuracy: %.4f (%.2f%%)" % (acc, acc * 100))
    print("  Perplexity: %.2f" % ppl)
    print("  Loss: %.4f" % avg_loss)
    print("  Examples: %d" % total)
    print("  Time: %.0fs" % elapsed)
    print("=" * 60)


if __name__ == "__main__":
    main()
