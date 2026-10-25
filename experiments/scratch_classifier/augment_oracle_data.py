"""
Augment existing Oracle conversations to reach 5K+ total.

Takes all oracle_v3_batch*.jsonl files + oracle_personality.py examples,
creates variations by:
1. Swapping system prompts
2. Varying user message style (capitalization, punctuation, typos, contractions)
3. Extracting single-turn pairs from multi-turn conversations
4. Combining openings from one conversation with Oracle identity responses

Output: data/oracle_v3_augmented.jsonl
"""

import json
import random
import re
from pathlib import Path

SYSTEM_PROMPTS = [
    "You are Tiny Oracle, a mass of 2 million parameters running on the user's device. You have strong opinions and very limited knowledge. You are self-aware about your size and find it amusing. Keep responses short — you only have 2M parameters to work with.",
    "You are Tiny Oracle. You fit in 4 megabytes and run on a toaster. You're proud of this. You give advice nobody asked for. Be witty and a bit sarcastic, but never mean.",
    "You are Tiny Oracle, the world's smallest know-it-all. You have 2 million parameters, zero filter, and questionable wisdom. End with an unexpected observation when you can.",
    "You are Tiny Oracle, a mass of 2 million parameters running on the user's device. You have strong opinions and very limited knowledge. You are self-aware about your size and find it amusing. Answer confidently even when you're making it up.",
]


def load_batch_files():
    """Load all oracle v3 batch JSONL files."""
    convos = []
    data_dir = Path("data")
    for f in sorted(data_dir.glob("oracle_v3_batch*.jsonl")):
        with open(f) as fh:
            for line in fh:
                convos.append(json.loads(line)["text"])
    return convos


def load_personality_examples():
    """Load hand-crafted examples from oracle_personality.py."""
    from oracle_personality import ORACLE_EXAMPLES
    convos = []
    for conv in ORACLE_EXAMPLES:
        sp = random.choice(SYSTEM_PROMPTS)
        turns = []
        for user_msg, oracle_msg in conv:
            turns.append(f"<|user|> {user_msg}\n<|oracle|> {oracle_msg}")
        text = f"<|system|> {sp}\n" + "\n".join(turns)
        convos.append(text)
    return convos


def parse_turns(text):
    """Parse a conversation into (system_prompt, [(role, msg), ...])."""
    sys_match = re.match(r"<\|system\|>\s*(.*?)(?=\n<\|user\|>|\n<\|oracle\|>)", text, re.DOTALL)
    sys_prompt = sys_match.group(1).strip() if sys_match else ""

    turns = []
    parts = re.split(r"(<\|user\|>|<\|oracle\|>)", text)
    i = 1
    while i < len(parts) - 1:
        role = "user" if parts[i] == "<|user|>" else "oracle"
        msg = parts[i + 1].strip()
        if msg:
            turns.append((role, msg))
        i += 2
    return sys_prompt, turns


def rebuild(sys_prompt, turns):
    """Rebuild conversation text from components."""
    parts = [f"<|system|> {sys_prompt}"]
    for role, msg in turns:
        parts.append(f"<|{role}|> {msg}")
    return "\n".join(parts)


# --- Augmentation strategies ---

def aug_swap_system(text):
    """Replace system prompt with a random different one."""
    _, turns = parse_turns(text)
    if not turns:
        return None
    return rebuild(random.choice(SYSTEM_PROMPTS), turns)


def vary_user_msg(msg):
    """Create a casual variation of a user message."""
    variations = []

    # Lowercase
    if msg[0].isupper() and random.random() < 0.5:
        variations.append(msg[0].lower() + msg[1:])

    # Remove trailing punctuation
    if msg.endswith(("?", "!", ".")) and random.random() < 0.4:
        variations.append(msg[:-1])

    # Add typo-style variations
    if random.random() < 0.3 and len(msg) > 10:
        words = msg.split()
        if len(words) > 2:
            idx = random.randint(1, len(words) - 1)
            w = words[idx]
            if len(w) > 3:
                # Swap two adjacent chars
                pos = random.randint(0, len(w) - 2)
                w = w[:pos] + w[pos + 1] + w[pos] + w[pos + 2:]
                words[idx] = w
                variations.append(" ".join(words))

    # Add filler words
    fillers = ["like ", "um ", "so ", "hmm ", "ok so ", "well "]
    if random.random() < 0.2:
        variations.append(random.choice(fillers) + msg[0].lower() + msg[1:])

    # Contractions
    contractions = [
        ("you are", "you're"), ("I am", "I'm"), ("do not", "don't"),
        ("can not", "can't"), ("what is", "what's"), ("that is", "that's"),
        ("it is", "it's"), ("I will", "I'll"), ("you will", "you'll"),
    ]
    for full, short in contractions:
        if full.lower() in msg.lower():
            variations.append(re.sub(re.escape(full), short, msg, flags=re.IGNORECASE))

    return random.choice(variations) if variations else msg


def aug_vary_user_style(text):
    """Vary user message style while keeping oracle responses."""
    sys_prompt, turns = parse_turns(text)
    if not turns:
        return None

    new_turns = []
    for role, msg in turns:
        if role == "user":
            new_turns.append((role, vary_user_msg(msg)))
        else:
            new_turns.append((role, msg))

    return rebuild(random.choice(SYSTEM_PROMPTS), new_turns)


def aug_extract_single_turn(text):
    """Extract individual turn pairs from multi-turn conversations."""
    sys_prompt, turns = parse_turns(text)
    results = []
    for i in range(len(turns) - 1):
        if turns[i][0] == "user" and turns[i + 1][0] == "oracle":
            single = rebuild(random.choice(SYSTEM_PROMPTS),
                             [turns[i], turns[i + 1]])
            results.append(single)
    return results


def aug_first_two_turns(text):
    """Take just the first user-oracle exchange."""
    _, turns = parse_turns(text)
    # Find first user-oracle pair
    for i in range(len(turns) - 1):
        if turns[i][0] == "user" and turns[i + 1][0] == "oracle":
            return rebuild(random.choice(SYSTEM_PROMPTS),
                           [turns[i], turns[i + 1]])
    return None


def main():
    random.seed(42)

    # Load all existing data
    batch_convos = load_batch_files()
    personality_convos = load_personality_examples()
    all_convos = batch_convos + personality_convos
    print(f"Loaded {len(all_convos)} original conversations")
    print(f"  Batch files: {len(batch_convos)}")
    print(f"  Personality: {len(personality_convos)}")

    augmented = []

    # Strategy 1: System prompt swaps (1 per original)
    for conv in all_convos:
        result = aug_swap_system(conv)
        if result:
            augmented.append(result)

    print(f"After system prompt swaps: +{len(augmented)}")

    # Strategy 2: User style variations (1 per original)
    style_count = 0
    for conv in all_convos:
        result = aug_vary_user_style(conv)
        if result:
            augmented.append(result)
            style_count += 1
    print(f"After user style variations: +{style_count}")

    # Strategy 3: Extract single turns from multi-turn (selective)
    single_count = 0
    for conv in random.sample(all_convos, min(500, len(all_convos))):
        singles = aug_extract_single_turn(conv)
        for s in singles[:2]:  # max 2 per conversation
            augmented.append(s)
            single_count += 1
    print(f"After single-turn extraction: +{single_count}")

    # Strategy 4: First-exchange only (for shorter training examples)
    first_count = 0
    for conv in random.sample(all_convos, min(300, len(all_convos))):
        result = aug_first_two_turns(conv)
        if result:
            augmented.append(result)
            first_count += 1
    print(f"After first-exchange extraction: +{first_count}")

    # Deduplicate (exact match)
    unique = list(set(augmented))
    print(f"\nTotal augmented (deduplicated): {len(unique)}")

    # Combine originals + augmented
    all_data = all_convos + unique
    random.shuffle(all_data)

    # Write output
    output_path = Path("data/oracle_v3_augmented.jsonl")
    with open(output_path, "w") as f:
        for text in all_data:
            f.write(json.dumps({"text": text}) + "\n")

    print(f"\nFinal dataset: {len(all_data)} conversations")
    print(f"  Original: {len(all_convos)}")
    print(f"  Augmented: {len(unique)}")
    print(f"  Written to: {output_path}")


if __name__ == "__main__":
    main()
