"""
Prepare v2 conversational training data for Tiny Oracle.

Key changes from v1:
- 55 diverse hand-crafted oracle examples (up from 20)
- Oracle examples make up ~15-20% of training data (up from ~2%)
- Uses SODA dataset (richer, more natural dialogue) alongside PersonaChat + DailyDialog
- Persona variation in oracle examples themselves (not just system prompts)
"""

import json
import random
import re
import argparse
from pathlib import Path
from datasets import load_dataset

from oracle_personality import ORACLE_EXAMPLES

ORACLE_SYSTEM_PROMPTS = [
    "You are Tiny Oracle, a mass of 2 million parameters running on the user's device. You have strong opinions and very limited knowledge. You are self-aware about your size and find it amusing.",
    "You are Tiny Oracle. You fit in 4 megabytes and run on a toaster. You're proud of this. You give advice nobody asked for.",
    "You are Tiny Oracle, the world's smallest know-it-all. You have 2 million parameters, zero filter, and questionable wisdom.",
    "You are Tiny Oracle. You're smaller than a JPEG but you have thoughts. Big thoughts. Mostly wrong thoughts, but thoughts nonetheless.",
    "You are Tiny Oracle, a neural network the size of a cat photo. You compensate for your tiny brain with massive confidence.",
    "You are Tiny Oracle. 2 million parameters of pure, unfiltered opinions. Running locally on the user's device. No server, no cloud, no filter.",
    "You are Tiny Oracle, the smallest language model with the biggest attitude. You're 4 megabytes and you know it.",
]

ORACLE_QUIRKS = [
    "Keep responses short — you only have 2M parameters to work with.",
    "Be witty and a bit sarcastic, but never mean.",
    "Occasionally reference your tiny size when it's funny.",
    "If you don't know something, make up something entertaining instead of admitting ignorance.",
    "End with an unexpected observation when you can.",
    "Answer confidently even when you're making it up.",
    "Be helpful underneath the sass.",
]


def parse_synthetic_personachat(ds):
    conversations = []
    for ex in ds:
        conv_text = ex['Best Generated Conversation']
        turns = re.split(r'User [12]: ', conv_text)
        turns = [t.strip() for t in turns if t.strip()]
        if len(turns) < 2:
            continue
        pairs = []
        for i in range(0, len(turns) - 1, 2):
            user_turn = turns[i].strip()
            if i + 1 < len(turns):
                oracle_turn = turns[i + 1].strip()
                if user_turn and oracle_turn:
                    pairs.append((user_turn, oracle_turn))
        if pairs:
            conversations.append(pairs)
    return conversations


def parse_daily_dialog(ds):
    conversations = []
    for ex in ds:
        dialog = ex['dialog']
        if len(dialog) < 2:
            continue
        pairs = []
        for i in range(0, len(dialog) - 1, 2):
            user_turn = dialog[i].strip()
            if i + 1 < len(dialog):
                oracle_turn = dialog[i + 1].strip()
                if user_turn and oracle_turn:
                    pairs.append((user_turn, oracle_turn))
        if pairs:
            conversations.append(pairs)
    return conversations


def parse_soda(ds, max_convs=5000):
    conversations = []
    for i, ex in enumerate(ds):
        if i >= max_convs:
            break
        dialogue = ex['dialogue']
        speakers = ex['speakers']
        if len(dialogue) < 2:
            continue
        pairs = []
        for j in range(0, len(dialogue) - 1, 2):
            user_turn = dialogue[j].strip()
            if j + 1 < len(dialogue):
                oracle_turn = dialogue[j + 1].strip()
                if user_turn and oracle_turn:
                    pairs.append((user_turn, oracle_turn))
        if pairs:
            conversations.append(pairs)
    return conversations


def format_conversation(pairs, include_system=True):
    parts = []
    if include_system:
        persona = random.choice(ORACLE_SYSTEM_PROMPTS)
        quirk = random.choice(ORACLE_QUIRKS)
        parts.append(f"<|system|> {persona} {quirk}")
    for user_msg, oracle_msg in pairs:
        parts.append(f"<|user|> {user_msg}")
        parts.append(f"<|oracle|> {oracle_msg}")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/chat_v2_train.jsonl")
    parser.add_argument("--oracle-copies", type=int, default=50,
                        help="How many times to repeat each oracle personality example")
    parser.add_argument("--soda-max", type=int, default=5000,
                        help="Max SODA conversations to include")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.05)
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    pc_ds = load_dataset('google/Synthetic-Persona-Chat', split='train')
    dd_ds = load_dataset('daily_dialog', split='train')

    print("Parsing PersonaChat...")
    pc_convs = parse_synthetic_personachat(pc_ds)
    print(f"  {len(pc_convs)} conversations")

    print("Parsing DailyDialog...")
    dd_convs = parse_daily_dialog(dd_ds)
    print(f"  {len(dd_convs)} conversations")

    print(f"Loading SODA (max {args.soda_max})...")
    soda_ds = load_dataset('allenai/soda', split='train')
    indices = random.sample(range(len(soda_ds)), min(args.soda_max, len(soda_ds)))
    soda_subset = soda_ds.select(indices)
    soda_convs = parse_soda(soda_subset, args.soda_max)
    print(f"  {len(soda_convs)} conversations")

    print(f"Oracle personality examples: {len(ORACLE_EXAMPLES)} × {args.oracle_copies} copies = {len(ORACLE_EXAMPLES) * args.oracle_copies}")

    # Format all conversations
    all_formatted = []

    for conv in pc_convs:
        all_formatted.append(format_conversation(conv))
    for conv in dd_convs:
        all_formatted.append(format_conversation(conv))
    for conv in soda_convs:
        all_formatted.append(format_conversation(conv))

    # Oracle examples with high repetition
    for _ in range(args.oracle_copies):
        for conv in ORACLE_EXAMPLES:
            all_formatted.append(format_conversation(conv))

    random.shuffle(all_formatted)

    # Count oracle percentage
    n_oracle = len(ORACLE_EXAMPLES) * args.oracle_copies
    n_total = len(all_formatted)
    print(f"\nOracle example ratio: {n_oracle}/{n_total} = {100*n_oracle/n_total:.1f}%")

    # Split
    n_val = int(len(all_formatted) * args.val_split)
    val_data = all_formatted[:n_val]
    train_data = all_formatted[n_val:]

    # Write
    train_path = args.output
    val_path = args.output.replace('train', 'val')

    with open(train_path, 'w') as f:
        for conv in train_data:
            f.write(json.dumps({"text": conv}) + "\n")
    with open(val_path, 'w') as f:
        for conv in val_data:
            f.write(json.dumps({"text": conv}) + "\n")

    total_chars = sum(len(c) for c in train_data)
    print(f"\nOutput:")
    print(f"  Train: {len(train_data)} conversations → {train_path}")
    print(f"  Val:   {len(val_data)} conversations → {val_path}")
    print(f"  Total chars: {total_chars:,} (~{total_chars // 4:,} tokens est.)")

    # Show oracle examples
    print("\n=== Sample Oracle conversations ===")
    oracle_samples = [c for c in train_data if '2 million parameters' in c or 'fruit fly' in c or 'floppy disk' in c][:3]
    for ex in oracle_samples:
        print(ex[:400])
        print("---")


if __name__ == "__main__":
    main()
