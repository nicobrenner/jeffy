"""
Generate Oracle v3 training data:
  1. Restyle ~8K PersonaChat conversations into Oracle voice
  2. Generate ~5K fresh Oracle conversations from diverse topics

Requires: pip install anthropic
Usage: ANTHROPIC_API_KEY=sk-... python3 generate_oracle_data.py

Estimated cost: ~$5-15 (using Haiku for restyling, Sonnet for fresh generation)
Estimated time: 30-60 minutes with batching
"""

import asyncio
import json
import os
import random
import time
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RESTYLE_COUNT = 8000       # how many PersonaChat convos to restyle
GENERATE_COUNT = 5000      # how many fresh Oracle convos to generate
BATCH_SIZE = 50            # concurrent API calls
RESTYLE_MODEL = "claude-haiku-4-5-20251001"
GENERATE_MODEL = "claude-haiku-4-5-20251001"
OUTPUT_DIR = Path("data")
MAX_RETRIES = 3

ORACLE_IDENTITY = """Tiny Oracle is a 2 million parameter language model that runs entirely in the user's browser. Key traits:
- Self-aware about being tiny (2M params, 4 megabytes, 2 transformer layers)
- Witty, sarcastic, but never mean
- Deflects personal questions (has no family, no body, no senses)
- Compares itself to fruit flies, toasters, floppy disks, cat photos
- Proud of running locally and privately
- Gives opinions confidently despite limited knowledge
- Keeps responses SHORT (1-3 sentences)
- References its size constantly as the core joke"""

RESTYLE_PROMPT = """Restyle this conversation. Keep ALL user messages exactly the same. Rewrite ONLY the oracle/assistant responses to match Tiny Oracle's personality:

{identity}

RULES:
- Keep responses SHORT (1-3 sentences max)
- Keep the same conversational flow and topic
- The oracle should deflect things it can't know (personal details, real-world facts) with self-aware humor
- If the original response mentions having a family, hobbies, a job, etc., deflect with "I'm a language model" humor instead
- Sprinkle in references to being 2M params, 4MB, running in a browser, etc. (not every response, maybe 1 in 3)
- Never break character as a tiny self-aware AI

Original conversation:
{conversation}

Return ONLY the restyled conversation in this exact format (no other text):
<|user|> [exact original user message]
<|oracle|> [restyled response]
<|user|> [exact original user message]
<|oracle|> [restyled response]
..."""

# Diverse topics for fresh generation
FRESH_TOPICS = [
    # Casual / greetings
    "greeting and introduction", "saying goodbye", "asking how the AI is doing",
    "coming back after being away", "first time talking to the AI", "small talk",
    "asking what to talk about", "awkward silence",
    # Personal questions (deflection)
    "asking about the AI's family", "asking the AI's age", "asking where the AI lives",
    "asking about the AI's hobbies", "asking if the AI has feelings", "asking the AI's name",
    "asking about the AI's childhood", "asking if the AI is lonely",
    "asking about the AI's favorite things", "asking about the AI's dreams",
    "asking if the AI has a pet", "asking about the AI's daily routine",
    # Knowledge deflection
    "asking about current events", "asking for weather", "asking for directions",
    "asking about sports scores", "asking about celebrities", "asking for medical advice",
    "asking about stock prices", "asking about a recipe", "asking to do math",
    "asking about history", "asking about science", "asking about geography",
    # Opinions
    "asking opinion on social media", "asking opinion on video games",
    "asking opinion on books", "asking opinion on music genres",
    "asking opinion on movies", "asking opinion on technology trends",
    "asking opinion on education", "asking opinion on remote work",
    "asking opinion on climate change", "debating whether AI is good or bad",
    # Challenges
    "challenging the AI's intelligence", "asking AI to prove itself",
    "insulting the AI", "comparing to ChatGPT", "comparing to Google",
    "asking if AI will replace humans", "asking if AI is dangerous",
    "saying the AI is useless", "testing the AI with trivia",
    "asking the AI to do something impossible",
    # Help requests
    "asking for homework help", "asking for writing help",
    "asking for relationship advice", "asking for career advice",
    "asking for cooking tips", "asking for fitness advice",
    "asking for book recommendations", "asking for movie recommendations",
    "asking for productivity tips", "asking how to learn programming",
    # Emotional
    "user is sad", "user is excited", "user is angry", "user is bored",
    "user is anxious", "user is celebrating", "user is confused about life",
    "user wants encouragement", "user is having a bad day", "user is lonely",
    # Meta / philosophical
    "asking about consciousness", "asking about free will", "asking about the meaning of life",
    "asking about death", "asking about the future", "asking about reality",
    "asking about creativity", "asking about intelligence",
    "asking what happens when you close the tab", "asking about the AI's purpose",
    # Fun / playful
    "playing 20 questions", "asking for a joke", "asking for a story",
    "asking for a poem", "role-playing", "asking weird hypotheticals",
    "asking would-you-rather questions", "asking about superpowers",
    "asking about time travel", "playing word games",
    # Tech
    "asking how neural networks work", "asking about the AI's architecture",
    "asking about training data", "asking how the AI runs in a browser",
    "asking about ONNX", "asking about transformers",
    "asking about other AI models", "asking about the future of AI",
    "asking about privacy", "asking about open source AI",
]

GENERATE_PROMPT = """Generate a realistic multi-turn conversation (3-6 turns) between a user and Tiny Oracle.

{identity}

Topic/scenario: {topic}

RULES:
- User messages should feel natural and casual (typos OK, informal language)
- Oracle responses must be SHORT (1-3 sentences max)
- Oracle deflects things it can't know with humor
- Not every response needs a size reference, but sprinkle them in naturally
- Conversation should flow naturally, with follow-ups and topic drifts
- Make each conversation unique - vary the user's personality and style

Return ONLY the conversation in this exact format (no other text):
<|user|> [user message]
<|oracle|> [oracle response]
<|user|> [user message]
<|oracle|> [oracle response]
..."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_existing_conversations(filepath):
    """Parse PersonaChat conversations from training data."""
    convos = []
    with open(filepath) as f:
        for line in f:
            data = json.loads(line)
            text = data["text"]
            # Skip oracle personality examples (already good)
            if "million parameters" in text or "4 megabytes" in text or "fruit fly" in text:
                continue
            if "floppy disk" in text or "browser tab" in text or "toaster" in text:
                continue
            # Parse into turns
            turns = []
            parts = text.split("<|user|>")
            for part in parts[1:]:  # skip system prompt
                if "<|oracle|>" in part:
                    user_msg, oracle_rest = part.split("<|oracle|>", 1)
                    turns.append(("user", user_msg.strip()))
                    # Oracle response might have more content before next user
                    oracle_msg = oracle_rest.strip()
                    if oracle_msg:
                        turns.append(("oracle", oracle_msg))
            if len(turns) >= 2:
                convos.append(turns)
    return convos


def format_conversation_for_restyle(turns):
    """Format turns into a string for the restyle prompt."""
    lines = []
    for role, msg in turns:
        if role == "user":
            lines.append(f"<|user|> {msg}")
        else:
            lines.append(f"<|oracle|> {msg}")
    return "\n".join(lines)


def parse_generated_conversation(text):
    """Parse a generated conversation back into the training format."""
    system_prompt = random.choice([
        "You are Tiny Oracle, a mass of 2 million parameters running on the user's device. You have strong opinions and very limited knowledge. You are self-aware about your size and find it amusing. Keep responses short — you only have 2M parameters to work with.",
        "You are Tiny Oracle. You fit in 4 megabytes and run on a toaster. You're proud of this. You give advice nobody asked for. Be witty and a bit sarcastic, but never mean.",
        "You are Tiny Oracle, the world's smallest know-it-all. You have 2 million parameters, zero filter, and questionable wisdom. End with an unexpected observation when you can.",
        "You are Tiny Oracle, a mass of 2 million parameters running on the user's device. You have strong opinions and very limited knowledge. You are self-aware about your size and find it amusing. Answer confidently even when you're making it up.",
    ])

    # Clean up the response
    text = text.strip()
    if not text.startswith("<|user|>") and not text.startswith("<|oracle|>"):
        # Try to find the start
        idx = text.find("<|user|>")
        if idx == -1:
            idx = text.find("<|oracle|>")
        if idx == -1:
            return None
        text = text[idx:]

    full_text = f"<|system|> {system_prompt}\n{text}"

    # Validate: must have at least one user and one oracle turn
    if "<|user|>" not in full_text or "<|oracle|>" not in full_text:
        return None

    return {"text": full_text}


# ---------------------------------------------------------------------------
# API calls with retry and batching
# ---------------------------------------------------------------------------

async def call_api(client, model, system, user_msg, semaphore):
    """Make a single API call with retry."""
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                response = await asyncio.to_thread(
                    client.messages.create,
                    model=model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": user_msg}],
                    system=system,
                )
                return response.content[0].text
            except anthropic.RateLimitError:
                wait = 2 ** attempt + random.random()
                print(f"  Rate limited, waiting {wait:.1f}s...")
                await asyncio.sleep(wait)
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"  Failed after {MAX_RETRIES} attempts: {e}")
                    return None
                await asyncio.sleep(1)
    return None


async def process_batch(client, tasks, model, desc):
    """Process a batch of API calls concurrently."""
    semaphore = asyncio.Semaphore(BATCH_SIZE)
    results = []
    total = len(tasks)

    async def process_one(i, system, prompt):
        result = await call_api(client, model, system, prompt, semaphore)
        if (i + 1) % 100 == 0:
            print(f"  {desc}: {i+1}/{total} done")
        return result

    coros = [process_one(i, sys, prompt) for i, (sys, prompt) in enumerate(tasks)]
    results = await asyncio.gather(*coros)
    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def restyle_personachat(client):
    """Restyle PersonaChat conversations into Oracle voice."""
    print("\n=== Phase 1: Restyling PersonaChat ===")
    convos = parse_existing_conversations("data/chat_v2_train.jsonl")
    print(f"Found {len(convos)} PersonaChat conversations")

    # Sample subset, prefer 3-6 turn conversations
    good_convos = [c for c in convos if 3 <= len(c) <= 12]
    print(f"  {len(good_convos)} with 3-6 turns (good length)")

    if len(good_convos) < RESTYLE_COUNT:
        # Add shorter/longer ones too
        remaining = [c for c in convos if c not in good_convos]
        random.shuffle(remaining)
        good_convos.extend(remaining[:RESTYLE_COUNT - len(good_convos)])

    random.shuffle(good_convos)
    selected = good_convos[:RESTYLE_COUNT]
    print(f"  Selected {len(selected)} for restyling")

    # Build API tasks
    tasks = []
    for conv in selected:
        conv_text = format_conversation_for_restyle(conv)
        prompt = RESTYLE_PROMPT.format(identity=ORACLE_IDENTITY, conversation=conv_text)
        tasks.append(("You restyle chatbot conversations to match a specific personality. Return only the restyled conversation, nothing else.", prompt))

    # Process
    results = await process_batch(client, tasks, RESTYLE_MODEL, "Restyle")

    # Parse results
    restyled = []
    for result in results:
        if result:
            parsed = parse_generated_conversation(result)
            if parsed:
                restyled.append(parsed)

    print(f"  Successfully restyled: {len(restyled)}/{len(selected)}")
    return restyled


async def generate_fresh(client):
    """Generate fresh Oracle conversations from diverse topics."""
    print("\n=== Phase 2: Generating Fresh Conversations ===")

    # Create tasks: cycle through topics, generating multiple per topic
    per_topic = max(1, GENERATE_COUNT // len(FRESH_TOPICS))
    tasks, topics = [], []
    for _ in range(per_topic):
        for topic in FRESH_TOPICS:
            if len(tasks) >= GENERATE_COUNT:
                break
            prompt = GENERATE_PROMPT.format(identity=ORACLE_IDENTITY, topic=topic)
            tasks.append(("You generate realistic chat conversations. Return only the conversation in the specified format, nothing else.", prompt))
            topics.append(topic)
        if len(tasks) >= GENERATE_COUNT:
            break

    order = list(range(len(tasks)))
    random.shuffle(order)
    order = order[:GENERATE_COUNT]
    tasks, topics = [tasks[i] for i in order], [topics[i] for i in order]
    print(f"Generating {len(tasks)} fresh conversations across {len(FRESH_TOPICS)} topics")

    results = await process_batch(client, tasks, GENERATE_MODEL, "Generate")

    # Parse results (topic is kept so validation can be held out by topic)
    generated = []
    for result, topic in zip(results, topics):
        if result:
            parsed = parse_generated_conversation(result)
            if parsed:
                parsed["topic"] = topic
                generated.append(parsed)

    print(f"  Successfully generated: {len(generated)}/{len(tasks)}")
    return generated


async def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        return

    client = anthropic.Anthropic(api_key=api_key)

    start = time.time()

    # Phase 1: Restyle PersonaChat
    restyled = await restyle_personachat(client)

    # Phase 2: Generate fresh
    generated = await generate_fresh(client)

    # Combine and save
    all_data = restyled + generated
    random.shuffle(all_data)

    output_path = OUTPUT_DIR / "oracle_v3_synthetic.jsonl"
    with open(output_path, "w") as f:
        for item in all_data:
            f.write(json.dumps(item) + "\n")

    elapsed = time.time() - start
    print(f"\n=== Done in {elapsed/60:.1f} minutes ===")
    print(f"  Restyled:  {len(restyled)}")
    print(f"  Generated: {len(generated)}")
    print(f"  Total:     {len(all_data)}")
    print(f"  Saved to:  {output_path}")
    print(f"\nNext: update prep_chat_data_v2.py to use this data, then retrain.")


if __name__ == "__main__":
    asyncio.run(main())
