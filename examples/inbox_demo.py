"""Inbox demo: classify sample messages, correct, retrain, and verify.

Shows the core Jeffy workflow:
1. Classify messages with a pretrained head
2. User corrects mistakes
3. Retrain on corrections (CPU, seconds)
4. Verify on held-out messages

Run:
    python examples/inbox_demo.py
"""

from jeffy.engine import Engine
from jeffy.train import train_classifier


def main():
    # --- Step 1: Classify sample messages with pretrained heads ---
    print("=" * 60)
    print("STEP 1: Classify messages with pretrained heads")
    print("=" * 60)

    engine = Engine()
    engine.load()

    messages = [
        ("I need to cancel my credit card immediately", "banking77"),
        ("WINNER! You've been selected for a $1000 prize! Call now!", "sms_spam"),
        ("Apple stock surges 5% after record iPhone sales", "ag_news"),
        ("The new restaurant downtown has amazing pasta", "ag_news"),
        ("I can't access my account after changing my password", "banking77"),
        ("Free entry to win a Caribbean cruise! Text YES to 12345", "sms_spam"),
    ]

    print()
    for text, task in messages:
        r = engine.predict(task, text)
        print(f"  [{task}] {r['label']} ({r['confidence']:.0%})")
        print(f"    \"{text[:60]}...\"" if len(text) > 60 else f"    \"{text}\"")
        print()

    # --- Step 2: Train a custom classifier with corrections ---
    print("=" * 60)
    print("STEP 2: Train a custom inbox router from examples")
    print("=" * 60)

    # Imagine a user classifying their own inbox categories
    training_examples = [
        # Work
        ("Q3 budget review meeting tomorrow at 2pm", "work"),
        ("Please review the attached contract by Friday", "work"),
        ("Your pull request has been approved", "work"),
        ("Team standup moved to 10am starting next week", "work"),
        ("Invoice #4521 is ready for your approval", "work"),
        ("New hire orientation schedule attached", "work"),

        # Family
        ("Mom's birthday dinner Saturday at 7", "family"),
        ("Soccer practice canceled due to rain", "family"),
        ("Grandma sent photos from the reunion", "family"),
        ("Pick up kids from school at 3:15 today", "family"),
        ("Family movie night this Friday?", "family"),
        ("Your sister's flight arrives at 6pm", "family"),

        # Promotions
        ("50% off everything this weekend only!", "promo"),
        ("Your loyalty points are about to expire", "promo"),
        ("New arrivals just dropped — shop now", "promo"),
        ("Flash sale: free shipping on orders over $25", "promo"),
        ("Exclusive member discount inside", "promo"),
        ("Don't miss our biggest sale of the year", "promo"),

        # Notifications
        ("Your package has been delivered", "notification"),
        ("Password changed successfully", "notification"),
        ("Your flight check-in is now open", "notification"),
        ("Bank statement for September is ready", "notification"),
        ("Subscription renewed: $9.99 charged", "notification"),
        ("Two-factor authentication code: 847291", "notification"),
    ]

    texts = [t for t, _ in training_examples]
    labels = [l for _, l in training_examples]

    print()
    clf = train_classifier(
        texts=texts,
        labels=labels,
        task_id="inbox_router",
        test_size=0.0,  # use all for training (demo)
        cv_folds=3,
    )

    # --- Step 3: Classify new messages ---
    print()
    print("=" * 60)
    print("STEP 3: Classify new messages (not in training data)")
    print("=" * 60)

    new_messages = [
        "Can you send me the Q4 projections?",
        "Aunt Clara is hosting Thanksgiving this year",
        "Buy 2 get 1 free — today only!",
        "Your order #7832 has shipped",
        "Sprint planning session at 3pm in conf room B",
        "Dad wants to know if you're coming to the cookout",
        "Limited time offer: upgrade your plan for $5/mo",
        "Your credit card statement is available",
    ]

    print()
    for msg in new_messages:
        r = clf.predict(msg)
        top = list(r["probabilities"].items())[:2]
        probs_str = ", ".join(f"{l}: {p:.0%}" for l, p in top)
        print(f"  {r['label']:15s} ({r['confidence']:.0%})  \"{msg}\"")

    # --- Step 4: Save and reload ---
    print()
    print("=" * 60)
    print("STEP 4: Save, reload, and verify")
    print("=" * 60)

    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        clf.save(tmpdir)
        print(f"\n  Saved to {tmpdir}/inbox_router/")
        files = os.listdir(os.path.join(tmpdir, "inbox_router"))
        print(f"  Files: {files}")

        # Reload in a fresh engine
        engine2 = Engine(tmpdir)
        engine2.load()
        print(f"  Reloaded: {list(engine2.capabilities.keys())}")

        r = engine2.predict("inbox_router", "Team lunch at noon tomorrow")
        print(f"  Predict: {r['label']} ({r['confidence']:.0%})")

    print()
    print("=" * 60)
    print("Done. This workflow runs entirely on CPU, no API keys needed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
