"""Approach A: Text generation via iterative MLM decoding.

Uses our trained hybrid model's MLM head to generate text by:
1. Start with a partially-masked sequence (prompt + [MASK] tokens)
2. Run through the MLM head to get predictions for all masked positions
3. Unmask the highest-confidence token
4. Repeat until all positions are filled

This is similar to how mask-predict models (Ghazvininejad et al., 2019)
and diffusion language models work — parallel decoding, iteratively refined.
"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

HERE = Path(__file__).resolve().parent


def load_model(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    vocab_map = ckpt["vocab_map"]

    from experiments.scratch_classifier.train_hybrid import HybridTransformer
    model = HybridTransformer(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config["d_ff"],
        max_seq_len=128,
        task_classes=config.get("task_classes", {}),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, config, vocab_map


def generate_mlm(model, tokenizer, vocab_map, prompt, gen_length=20,
                 temperature=1.0, device="cpu"):
    compact_to_full = {v: k for k, v in vocab_map.items()}

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_compact = [vocab_map.get(t, 0) for t in prompt_ids]

    mask_id = 0
    seq = prompt_compact + [mask_id] * gen_length
    seq = seq[:128]
    n_masks = seq.count(mask_id)

    input_ids = torch.tensor([seq], device=device)
    attention_mask = torch.ones_like(input_ids)

    steps = []
    for step in range(n_masks):
        with torch.no_grad():
            hidden = model._transformer_forward(input_ids, attention_mask)
            logits = model.mlm_head(hidden)

        mask_positions = (input_ids[0] == mask_id).nonzero(as_tuple=True)[0]
        if len(mask_positions) == 0:
            break

        best_conf = -float("inf")
        best_pos = None
        best_token = None
        for pos in mask_positions:
            probs = F.softmax(logits[0, pos] / temperature, dim=-1)
            top_prob, top_idx = probs.max(dim=-1)
            if top_prob.item() > best_conf:
                best_conf = top_prob.item()
                best_pos = pos.item()
                best_token = top_idx.item()

        input_ids[0, best_pos] = best_token
        full_token_id = compact_to_full.get(best_token, 0)
        token_str = tokenizer.decode([full_token_id])
        steps.append((step + 1, best_pos, token_str, best_conf))

    full_ids = [compact_to_full.get(t.item(), 0) for t in input_ids[0]]
    generated = tokenizer.decode(full_ids, skip_special_tokens=True)
    return generated, steps


def generate_parallel(model, tokenizer, vocab_map, prompt, gen_length=20,
                      n_iterations=10, temperature=1.0, device="cpu"):
    """Alternative: unmask multiple tokens per step (faster, noisier)."""
    compact_to_full = {v: k for k, v in vocab_map.items()}

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_compact = [vocab_map.get(t, 0) for t in prompt_ids]

    mask_id = 0
    seq = prompt_compact + [mask_id] * gen_length
    seq = seq[:128]

    input_ids = torch.tensor([seq], device=device)
    attention_mask = torch.ones_like(input_ids)

    tokens_per_step = max(1, gen_length // n_iterations)

    for iteration in range(n_iterations):
        mask_positions = (input_ids[0] == mask_id).nonzero(as_tuple=True)[0]
        if len(mask_positions) == 0:
            break

        with torch.no_grad():
            hidden = model._transformer_forward(input_ids, attention_mask)
            logits = model.mlm_head(hidden)

        confidences = []
        for pos in mask_positions:
            probs = F.softmax(logits[0, pos] / temperature, dim=-1)
            top_prob, top_idx = probs.max(dim=-1)
            confidences.append((pos.item(), top_idx.item(), top_prob.item()))

        confidences.sort(key=lambda x: x[2], reverse=True)
        n_unmask = min(tokens_per_step, len(confidences))
        for pos, tok, conf in confidences[:n_unmask]:
            input_ids[0, pos] = tok

    full_ids = [compact_to_full.get(t.item(), 0) for t in input_ids[0]]
    generated = tokenizer.decode(full_ids, skip_special_tokens=True)
    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="The weather today is")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--parallel", action="store_true",
                        help="Use parallel decoding (faster, noisier)")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--checkpoint", type=str,
                        default="experiments/scratch_classifier/checkpoints/best_hybrid.pt")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {args.checkpoint}...")
    model, config, vocab_map = load_model(args.checkpoint, device)
    print(f"  {config['vocab_size']} vocab, d_model={config['d_model']}")

    print(f"\nPrompt: \"{args.prompt}\"")
    print(f"Generating {args.length} tokens (temperature={args.temperature})")
    print("=" * 60)

    if args.parallel:
        result = generate_parallel(
            model, tokenizer, vocab_map, args.prompt,
            gen_length=args.length, n_iterations=args.iterations,
            temperature=args.temperature, device=device,
        )
        print(f"\nResult: {result}")
    else:
        result, steps = generate_mlm(
            model, tokenizer, vocab_map, args.prompt,
            gen_length=args.length, temperature=args.temperature,
            device=device,
        )
        print("\nStep-by-step decoding:")
        for step, pos, token, conf in steps:
            print(f"  Step {step:2d}: pos={pos:3d}  token='{token}'  conf={conf:.3f}")
        print(f"\nResult: {result}")

    print("\n" + "=" * 60)
    prompts = [
        "I love this movie because",
        "The stock market today",
        "Dear customer, your account has been",
        "Breaking news: scientists discover",
    ]
    print("\nAdditional examples:")
    for p in prompts:
        if args.parallel:
            out = generate_parallel(
                model, tokenizer, vocab_map, p,
                gen_length=args.length, n_iterations=args.iterations,
                temperature=args.temperature, device=device,
            )
        else:
            out, _ = generate_mlm(
                model, tokenizer, vocab_map, p,
                gen_length=args.length, temperature=args.temperature,
                device=device,
            )
        print(f"  \"{p}\" -> {out}")


if __name__ == "__main__":
    main()
