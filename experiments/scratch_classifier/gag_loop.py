"""GAG (Generate-Augmented Generation) Loop: Classifier-Guided Text Generation.

Combines our LM (4.6M, TinyStories) with our classifier (6M, multi-task)
to test the generate→verify→refine pattern at micro scale.

The LM generates candidate continuations. The classifier scores them on
sentiment, topic coherence, or other learned dimensions. The loop picks
the best candidate or re-samples when quality is low.

This is speculative decoding at ~11M params total — no one does this at
this scale. If it works, it shows the MVA pattern (small orchestrator +
verifier) is viable.

Usage:
    python gag_loop.py
    python gag_loop.py --prompt "The little girl was very"
    python gag_loop.py --strategy best-of-n --n-candidates 10
    python gag_loop.py --strategy adaptive --sentiment positive
    python gag_loop.py --strategy reject --task sst2 --target-label 1
"""

import argparse
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from train_lm import CausalTransformer
from train_hybrid import HybridTransformer


def load_lm(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = CausalTransformer(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        max_seq_len=cfg["max_seq_len"],
        tie_weights=cfg.get("tie_weights", False),
        n_iterations=cfg.get("n_iterations", 1),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    full_to_compact = ckpt["full_to_compact"]
    compact_to_full = ckpt["compact_to_full"]
    return model, full_to_compact, compact_to_full, cfg


def load_classifier(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = HybridTransformer(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        max_seq_len=cfg["max_seq_len"],
        task_classes=cfg.get("task_classes"),
        tie_weights=cfg.get("tie_weights", False),
        n_iterations=cfg.get("n_iterations", 1),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    vocab_map = ckpt.get("vocab_map", {})
    return model, vocab_map, cfg


# Cache the Qwen tokenizer to avoid reloading per call
_qwen_tok = None

def get_qwen_tokenizer():
    global _qwen_tok
    if _qwen_tok is None:
        from transformers import AutoTokenizer as AT
        _qwen_tok = AT.from_pretrained("Qwen/Qwen2.5-0.5B")
    return _qwen_tok


def classify_text_fast(text, clf_model, clf_vocab_map, task, device):
    """Run classifier on text. Returns (label, confidence, probs)."""
    tok = get_qwen_tokenizer()
    qwen_ids = tok.encode(text, add_special_tokens=False)
    clf_ids = [clf_vocab_map.get(tid, 0) for tid in qwen_ids]
    if len(clf_ids) > 128:
        clf_ids = clf_ids[:128]
    input_ids = torch.tensor([clf_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(input_ids, dtype=torch.bool)

    with torch.no_grad():
        logits = clf_model.forward_classify(input_ids, mask, task)
    probs = F.softmax(logits, dim=-1)
    label = probs.argmax(dim=-1).item()
    confidence = probs.max(dim=-1).values.item()
    return label, confidence, probs[0]


def generate_candidate(lm_model, input_ids, max_new_tokens, temperature, top_k):
    """Generate one candidate continuation from the LM."""
    with torch.no_grad():
        output_ids = lm_model.generate(
            input_ids, max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=top_k,
        )
    return output_ids[0]


def decode_compact(ids, compact_to_full, gpt2_tok, eos_id):
    """Convert compact LM ids back to text."""
    full_ids = [compact_to_full[c] for c in ids]
    try:
        eos_pos = full_ids.index(eos_id)
        full_ids = full_ids[:eos_pos]
    except ValueError:
        pass
    return gpt2_tok.decode(full_ids)


# --- Strategies ---

def strategy_best_of_n(lm_model, clf_model, clf_vocab_map,
                       input_ids, compact_to_full, gpt2_tok,
                       task, target_label, n_candidates,
                       max_new_tokens, temperature, top_k, device):
    """Generate N candidates, score each with classifier, pick best."""
    candidates = []
    for i in range(n_candidates):
        out = generate_candidate(lm_model, input_ids, max_new_tokens, temperature, top_k)
        text = decode_compact(out.tolist(), compact_to_full, gpt2_tok, gpt2_tok.eos_token_id)
        label, conf, probs = classify_text_fast(text, clf_model, clf_vocab_map, task, device)
        if target_label is not None:
            score = probs[target_label].item()
        else:
            score = conf
        candidates.append((text, label, conf, score, probs))

    candidates.sort(key=lambda x: x[3], reverse=True)
    return candidates


def strategy_reject(lm_model, clf_model, clf_vocab_map,
                    input_ids, compact_to_full, gpt2_tok,
                    task, target_label, threshold,
                    max_new_tokens, temperature, top_k, device,
                    max_attempts=20):
    """Generate until classifier approves (score > threshold)."""
    attempts = []
    for i in range(max_attempts):
        out = generate_candidate(lm_model, input_ids, max_new_tokens, temperature, top_k)
        text = decode_compact(out.tolist(), compact_to_full, gpt2_tok, gpt2_tok.eos_token_id)
        label, conf, probs = classify_text_fast(text, clf_model, clf_vocab_map, task, device)
        if target_label is not None:
            score = probs[target_label].item()
        else:
            score = conf
        attempts.append((text, label, conf, score))
        if score >= threshold:
            return attempts, True

    return attempts, False


def strategy_adaptive(lm_model, clf_model, clf_vocab_map,
                      input_ids, compact_to_full, gpt2_tok,
                      task, target_label,
                      max_new_tokens, temperature, top_k, device):
    """Adaptive: start with 1 candidate, double if quality is low."""
    thresholds = [(0.8, 1), (0.6, 2), (0.4, 4), (0.2, 8), (0.0, 16)]
    all_candidates = []

    for threshold, batch_size in thresholds:
        for _ in range(batch_size):
            out = generate_candidate(lm_model, input_ids, max_new_tokens, temperature, top_k)
            text = decode_compact(out.tolist(), compact_to_full, gpt2_tok, gpt2_tok.eos_token_id)
            label, conf, probs = classify_text_fast(text, clf_model, clf_vocab_map, task, device)
            if target_label is not None:
                score = probs[target_label].item()
            else:
                score = conf
            all_candidates.append((text, label, conf, score, probs))

            if score >= threshold:
                return all_candidates, threshold

    all_candidates.sort(key=lambda x: x[3], reverse=True)
    return all_candidates, 0.0


SENTIMENT_LABELS = {0: "negative", 1: "positive"}
AG_NEWS_LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
EMOTION_LABELS = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}


def label_name(task, label_id):
    if task == "sst2":
        return SENTIMENT_LABELS.get(label_id, str(label_id))
    elif task == "ag_news":
        return AG_NEWS_LABELS.get(label_id, str(label_id))
    elif task == "emotion":
        return EMOTION_LABELS.get(label_id, str(label_id))
    return str(label_id)


def main():
    parser = argparse.ArgumentParser(description="GAG: Generate-Augmented Generation")
    parser.add_argument("--lm-checkpoint", type=str,
                        default="checkpoints/best_lm_base.pt")
    parser.add_argument("--clf-checkpoint", type=str,
                        default="checkpoints/best_hybrid_tied_iter2.pt")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Generation prompt (default: sample prompts)")
    parser.add_argument("--strategy", type=str, default="best-of-n",
                        choices=["best-of-n", "reject", "adaptive", "compare"])
    parser.add_argument("--task", type=str, default="sst2",
                        help="Classifier task head to use for scoring")
    parser.add_argument("--target-label", type=int, default=None,
                        help="Target class label (e.g., 1 for positive sentiment)")
    parser.add_argument("--n-candidates", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Acceptance threshold for reject strategy")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = "cpu" if args.cpu else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load models ---
    print("Loading LM...", end=" ", flush=True)
    t0 = time.time()
    lm_model, full_to_compact, compact_to_full, lm_cfg = load_lm(
        args.lm_checkpoint, device
    )
    lm_params = sum(p.numel() for p in lm_model.parameters())
    print(f"{lm_params/1e6:.1f}M params ({time.time()-t0:.1f}s)")

    print("Loading classifier...", end=" ", flush=True)
    t0 = time.time()
    clf_model, clf_vocab_map, clf_cfg = load_classifier(
        args.clf_checkpoint, device
    )
    clf_params = sum(p.numel() for p in clf_model.parameters())
    print(f"{clf_params/1e6:.1f}M params ({time.time()-t0:.1f}s)")

    print(f"Combined: {(lm_params+clf_params)/1e6:.1f}M params")
    print(f"Classifier tasks: {list(clf_cfg.get('task_classes', {}).keys())}")
    print(f"Scoring task: {args.task}", flush=True)

    gpt2_tok = AutoTokenizer.from_pretrained("gpt2")

    prompts = [args.prompt] if args.prompt else [
        "Once upon a time",
        "The little girl was very",
        "Lily and Tom went to the",
    ]

    for prompt_text in prompts:
        print(f"\n{'='*70}")
        print(f"Prompt: \"{prompt_text}\"")
        print(f"Strategy: {args.strategy} | Task: {args.task} | Target: {args.target_label}")
        print(f"{'='*70}")

        # Encode prompt for LM
        orig_ids = gpt2_tok.encode(prompt_text)
        compact_ids = [full_to_compact.get(t, 0) for t in orig_ids]
        input_ids = torch.tensor([compact_ids], dtype=torch.long, device=device)

        t0 = time.time()

        if args.strategy == "best-of-n":
            candidates = strategy_best_of_n(
                lm_model, clf_model, clf_vocab_map,
                input_ids, compact_to_full, gpt2_tok,
                args.task, args.target_label, args.n_candidates,
                args.max_tokens, args.temperature, args.top_k, device,
            )
            elapsed = time.time() - t0
            print(f"\n{args.n_candidates} candidates in {elapsed*1000:.1f}ms "
                  f"({elapsed/args.n_candidates*1000:.1f}ms/candidate)")

            print(f"\n{'Rank':>4} {'Score':>6} {'Label':>12} {'Conf':>6}  Text")
            print(f"{'-'*4} {'-'*6} {'-'*12} {'-'*6}  {'-'*40}")
            for i, (text, label, conf, score, probs) in enumerate(candidates):
                lbl = label_name(args.task, label)
                preview = text[:100].replace("\n", " ")
                marker = " <<<" if i == 0 else ""
                print(f"{i+1:>4} {score:>6.3f} {lbl:>12} {conf:>6.3f}  {preview}{marker}")

            print(f"\nBest ({label_name(args.task, candidates[0][1])}, "
                  f"score={candidates[0][3]:.3f}):")
            print(f"  {candidates[0][0][:300]}")

        elif args.strategy == "reject":
            attempts, accepted = strategy_reject(
                lm_model, clf_model, clf_vocab_map,
                input_ids, compact_to_full, gpt2_tok,
                args.task, args.target_label, args.threshold,
                args.max_tokens, args.temperature, args.top_k, device,
            )
            elapsed = time.time() - t0
            n = len(attempts)
            print(f"\n{n} attempts in {elapsed*1000:.1f}ms "
                  f"({'accepted' if accepted else 'exhausted'} "
                  f"at threshold={args.threshold})")
            for i, (text, label, conf, score) in enumerate(attempts):
                lbl = label_name(args.task, label)
                preview = text[:80].replace("\n", " ")
                marker = " <<<" if (accepted and i == n-1) else ""
                print(f"  [{i+1}] score={score:.3f} ({lbl}) {preview}{marker}")

            final = attempts[-1] if accepted else max(attempts, key=lambda x: x[3])
            print(f"\nSelected (score={final[3]:.3f}):")
            print(f"  {final[0][:300]}")

        elif args.strategy == "adaptive":
            candidates, threshold = strategy_adaptive(
                lm_model, clf_model, clf_vocab_map,
                input_ids, compact_to_full, gpt2_tok,
                args.task, args.target_label,
                args.max_tokens, args.temperature, args.top_k, device,
            )
            elapsed = time.time() - t0
            n = len(candidates)
            print(f"\n{n} candidates in {elapsed*1000:.1f}ms "
                  f"(stopped at threshold={threshold})")
            for i, (text, label, conf, score, probs) in enumerate(candidates):
                lbl = label_name(args.task, label)
                preview = text[:80].replace("\n", " ")
                print(f"  [{i+1}] score={score:.3f} ({lbl}) {preview}")

            best = candidates[0] if candidates[0][3] >= candidates[-1][3] else candidates[-1]
            print(f"\nBest (score={best[3]:.3f}):")
            print(f"  {best[0][:300]}")

        elif args.strategy == "compare":
            # Compare unguided vs guided generation
            print("\n--- Unguided (single sample) ---")
            out = generate_candidate(lm_model, input_ids, args.max_tokens, args.temperature, args.top_k)
            text = decode_compact(out.tolist(), compact_to_full, gpt2_tok, gpt2_tok.eos_token_id)
            label, conf, probs = classify_text_fast(text, clf_model, clf_vocab_map, args.task, device)
            if args.target_label is not None:
                score = probs[args.target_label].item()
            else:
                score = conf
            print(f"  Score: {score:.3f} ({label_name(args.task, label)})")
            print(f"  {text[:200]}")

            print(f"\n--- Guided (best of {args.n_candidates}) ---")
            candidates = strategy_best_of_n(
                lm_model, clf_model, clf_vocab_map,
                input_ids, compact_to_full, gpt2_tok,
                args.task, args.target_label, args.n_candidates,
                args.max_tokens, args.temperature, args.top_k, device,
            )
            best = candidates[0]
            print(f"  Score: {best[3]:.3f} ({label_name(args.task, best[1])})")
            print(f"  {best[0][:200]}")

            improvement = best[3] - score
            print(f"\n  Improvement: {improvement:+.3f} "
                  f"({score:.3f} → {best[3]:.3f})")

    # --- Latency benchmark ---
    print(f"\n{'='*70}")
    print("Latency benchmark (100 iterations)")
    print(f"{'='*70}")

    orig_ids = gpt2_tok.encode("Once upon a time")
    compact_ids = [full_to_compact.get(t, 0) for t in orig_ids]
    input_ids = torch.tensor([compact_ids], dtype=torch.long, device=device)

    # LM generation only
    t0 = time.time()
    for _ in range(100):
        generate_candidate(lm_model, input_ids, 20, 0.8, 50)
    lm_ms = (time.time() - t0) / 100 * 1000

    # Classifier only
    t0 = time.time()
    for _ in range(100):
        classify_text_fast("Once upon a time there was a happy little girl",
                           clf_model, clf_vocab_map, args.task, device)
    clf_ms = (time.time() - t0) / 100 * 1000

    # Full loop (generate + classify)
    t0 = time.time()
    for _ in range(100):
        out = generate_candidate(lm_model, input_ids, 20, 0.8, 50)
        text = decode_compact(out.tolist(), compact_to_full, gpt2_tok, gpt2_tok.eos_token_id)
        classify_text_fast(text, clf_model, clf_vocab_map, args.task, device)
    loop_ms = (time.time() - t0) / 100 * 1000

    print(f"  LM generate (20 tok):   {lm_ms:6.1f}ms")
    print(f"  Classifier score:       {clf_ms:6.1f}ms")
    print(f"  Full loop (gen+score):  {loop_ms:6.1f}ms")
    print(f"  10x best-of-N:          {loop_ms*10:6.1f}ms")
    print(f"  50x loops/second:       would take {loop_ms*50:.0f}ms")


if __name__ == "__main__":
    main()
