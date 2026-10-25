"""Approach B & C: Text generation via classifier-as-scorer (JevChat-style).

Generates text by treating the classifier as a quality scorer:
1. Given prompt + text_so_far, create candidate completions (each appending one token)
2. Score each candidate using the classifier (which completion is "best"?)
3. Sample from the score distribution
4. Append chosen token, repeat

Two modes:
  --mode hybrid   Uses our 6M hybrid model's classification heads (Approach B)
  --mode jeffy    Uses current Jeffy (BGE + logreg) as the scorer (Approach C)

The key insight from JevChat: present complete hypothesis strings, not isolated tokens.
"""

import argparse
import time
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

HERE = Path(__file__).resolve().parent


def load_hybrid(ckpt_path, device="cpu"):
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


class HybridScorer:
    """Score candidate texts using our hybrid model's classification heads."""

    def __init__(self, model, tokenizer, vocab_map, task_name, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.vocab_map = vocab_map
        self.task_name = task_name
        self.device = device

    def score_candidates(self, candidates):
        encoded = self.tokenizer(
            candidates, truncation=True, padding=True,
            max_length=128, return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        remapped = torch.zeros_like(input_ids)
        for i in range(input_ids.shape[0]):
            for j in range(input_ids.shape[1]):
                remapped[i, j] = self.vocab_map.get(input_ids[i, j].item(), 0)

        remapped = remapped.to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            pooled = self.model.encode(remapped, attention_mask)
            logits = self.model.heads[self.task_name](pooled)
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
            confidence = 1.0 - entropy / torch.log(torch.tensor(logits.shape[-1], dtype=torch.float))

        return confidence.cpu().numpy()


class JeffyScorer:
    """Score candidate texts using Jeffy's BGE + logreg pipeline."""

    def __init__(self, pack_path, task_name):
        from jeffy.engine import Engine
        self.engine = Engine(pack_path)
        self.task_name = task_name

    def score_candidates(self, candidates):
        results = self.engine.classify_batch(candidates, self.task_name)
        confidences = np.array([r["confidence"] for r in results])
        return confidences


def build_token_candidates(tokenizer, top_k=50):
    """Get the top-k most common tokens as our generation vocabulary."""
    tokens = []
    for i in range(min(tokenizer.vocab_size, 5000)):
        decoded = tokenizer.decode([i])
        if decoded.strip() and len(decoded) <= 10:
            tokens.append((i, decoded))
    return tokens[:top_k]


def generate_token_by_token(scorer, tokenizer, prompt, gen_length=20,
                            top_k=50, temperature=1.0):
    """Generate text by scoring candidate completions token by token."""
    candidates_pool = build_token_candidates(tokenizer, top_k=top_k)

    text = prompt
    steps = []

    for step in range(gen_length):
        candidate_texts = [text + tok_str for _, tok_str in candidates_pool]
        scores = scorer.score_candidates(candidate_texts)

        scores = scores / max(temperature, 0.01)
        exp_scores = np.exp(scores - scores.max())
        probs = exp_scores / exp_scores.sum()

        chosen_idx = np.random.choice(len(candidates_pool), p=probs)
        chosen_id, chosen_str = candidates_pool[chosen_idx]
        text += chosen_str
        steps.append((step + 1, chosen_str, probs[chosen_idx]))

    return text, steps


def generate_word_by_word(scorer, prompt, gen_length=15, temperature=1.0):
    """Generate text by scoring candidate words (simpler, more readable)."""
    common_words = [
        "the", "a", "an", "is", "was", "are", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "shall",
        "not", "no", "yes", "very", "much", "more", "most",
        "good", "bad", "great", "new", "old", "big", "small",
        "and", "but", "or", "so", "if", "then", "that", "this",
        "it", "I", "you", "he", "she", "we", "they", "my", "your",
        "in", "on", "at", "to", "for", "with", "from", "by",
        "of", "about", "like", "just", "really", "never", "always",
        "happy", "sad", "angry", "love", "hate", "think", "know",
        "want", "need", "feel", "see", "hear", "go", "come",
        "time", "day", "way", "people", "world", "life", "work",
        "money", "market", "stock", "news", "today", "because",
        "movie", "film", "story", "book", "song", "product",
        ".", ",", "!", "?", "...",
    ]

    text = prompt
    steps = []

    for step in range(gen_length):
        candidate_texts = [text + " " + w for w in common_words]
        scores = scorer.score_candidates(candidate_texts)

        scores = scores / max(temperature, 0.01)
        exp_scores = np.exp(scores - scores.max())
        probs = exp_scores / exp_scores.sum()

        chosen_idx = np.random.choice(len(common_words), p=probs)
        chosen_word = common_words[chosen_idx]
        text += " " + chosen_word
        steps.append((step + 1, chosen_word, probs[chosen_idx]))

    return text, steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["hybrid", "jeffy"], default="hybrid")
    parser.add_argument("--prompt", type=str, default="The weather today is")
    parser.add_argument("--length", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--task", type=str, default="sst2",
                        help="Classification task to use as scorer")
    parser.add_argument("--method", choices=["token", "word"], default="word",
                        help="token=BPE tokens, word=common English words")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--checkpoint", type=str,
                        default="experiments/scratch_classifier/checkpoints/best_hybrid.pt")
    parser.add_argument("--pack", type=str, default=None,
                        help="Path to Jeffy pack (for --mode jeffy)")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.mode == "hybrid":
        print(f"Mode: Hybrid classifier scorer (task={args.task})")
        model, config, vocab_map = load_hybrid(args.checkpoint, device)
        if args.task not in model.heads:
            print(f"Available tasks: {list(model.heads.keys())}")
            return
        scorer = HybridScorer(model, tokenizer, vocab_map, args.task, device)
    else:
        print(f"Mode: Jeffy BGE+logreg scorer (task={args.task})")
        if not args.pack:
            print("Error: --pack required for jeffy mode")
            return
        scorer = JeffyScorer(args.pack, args.task)

    print(f"Method: {args.method}, Temperature: {args.temperature}")
    print(f"\nPrompt: \"{args.prompt}\"")
    print("=" * 60)

    prompts = [
        args.prompt,
        "I love this movie because",
        "The stock market today",
        "Dear customer, your account",
        "This product is absolutely",
    ]

    for prompt in prompts:
        t0 = time.time()
        if args.method == "word":
            result, steps = generate_word_by_word(
                scorer, prompt,
                gen_length=args.length, temperature=args.temperature,
            )
        else:
            result, steps = generate_token_by_token(
                scorer, tokenizer, prompt,
                gen_length=args.length, top_k=args.top_k,
                temperature=args.temperature,
            )
        elapsed = time.time() - t0

        print(f"\n\"{prompt}\"")
        print(f"  -> {result}")
        print(f"  ({elapsed:.1f}s, {len(steps)} steps)")
        for step, word, conf in steps[:5]:
            print(f"     step {step}: '{word}' (p={conf:.3f})")
        if len(steps) > 5:
            print(f"     ... ({len(steps) - 5} more steps)")


if __name__ == "__main__":
    main()
