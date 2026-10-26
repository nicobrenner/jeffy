"""Build the shippable support_ticket_v3 pack: the noun fix, plus three metadata defects.

`fix_support_ticket_v3.py` proved the augmentation works but only wrote metrics,
so production still serves the broken head. This builds the artifact.

Four things change relative to the shipped v2 pack:

1. **The training data.** track_order keyed on the object noun rather than the
   intent: 99.1% of its rows say "order" and essentially none say parcel, package
   or shipment, while the delivery intents carry those nouns instead. 96% of its
   rows also contain the literal `{{Order Number}}` placeholder, a token no
   customer types. Augmentation strips the placeholder, swaps the noun, and adds
   twelve hand-written phrasings. Refund rows are relabelled against a written
   three-way policy, because a first attempt regressed track_refund.
2. **`oos_threshold`.** The pack carries its own calibrated abstain threshold, so
   a caller who passes nothing gets safe behaviour instead of `out_of_scope:
   false` on every input. 0.5 is measured, not chosen: it caught 8 of 8
   out-of-scope probes while holding back 2 of 27 valid ones.
3. **The licence.** v2 shipped `CC BY 4.0`. The bitext dataset is
   `cdla-sharing-1.0`. We were publishing the wrong terms for someone else's data.
4. **`label_descriptions`.** v2 shipped `"bitext intent 'x', under category 'y'"`
   for all 27 classes, which restates the class name and documents nothing. These
   state what separates each class from the one it gets confused with.

The split is the one the shipped pack was measured on, loaded from disk, so the
before/after numbers compare like with like.

Usage:
    python scripts/build_support_ticket_v3.py            # measure only
    python scripts/build_support_ticket_v3.py --write    # write the pack
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fix_support_ticket_v3 import (  # noqa: E402
    C, ENCODER, OOS, PARAPHRASES, REFUND_HOLDOUT, REFUND_SEED, SEED, SPLIT,
    TRACK_HOLDOUT, TRACK_SEED, augment_track_order, fit, flatten, predict_rows,
    score,
)

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data/eval_results/support_ticket_v3"
PACK = REPO / "src/jeffy/pack"

# Measured in fix_support_ticket_v3.py's confidence sweep: at 0.5 the head caught
# 8 of 8 out-of-scope probes and held back 2 of 27 valid paraphrases. 0.3 caught
# only 6 of 8; 0.9 held back 15 of 27. Per-class coverage at 0.5 is >= 0.94
# everywhere, where a global 0.9 gate would discard 56% of correct
# switch_account answers.
OOS_THRESHOLD = 0.5

# The bitext card's actual terms. v2 shipped CC BY 4.0, which is not this dataset.
LICENSE = "cdla-sharing-1.0"

# What actually separates each class from its nearest neighbour. Written for the
# seams the external tester found, not generated from the class name.
LABEL_DESCRIPTIONS = {
    "cancel_order": "Stop an order from happening. Not a refund request: no money "
                    "has been asked for yet. Compare get_refund, check_cancellation_fee",
    "change_order": "Modify the contents of an order already placed, such as items "
                    "or quantity. Compare change_shipping_address for the destination",
    "change_shipping_address": "Change the delivery address on an existing order. "
                               "Note: rows do not cover packages already in transit, "
                               "so in-transit redirects have no correct class here",
    "check_cancellation_fee": "Ask what cancelling would cost, without yet cancelling. "
                              "Compare cancel_order, which acts",
    "check_invoice": "Look at or query an invoice that already exists. Compare "
                     "get_invoice, which asks to be sent one",
    "check_payment_methods": "Ask which payment methods are accepted. Compare "
                             "payment_issue, where a payment has failed",
    "check_refund_policy": "Ask what the refund rules are. Compare get_refund, which "
                           "asks for money back under them",
    "complaint": "Express dissatisfaction without a specific actionable request. "
                 "Acts as a catch-all sink: unrelated input often lands here with "
                 "moderate confidence, so this class needs its own rule rather than "
                 "a confidence gate",
    "contact_customer_service": "Ask how to reach support, by any channel",
    "contact_human_agent": "Ask specifically for a person rather than automation. "
                           "Compare contact_customer_service, which is channel-agnostic",
    "create_account": "Open a new account. Compare registration_problems, where "
                      "creating one failed",
    "delete_account": "Close or remove an account permanently",
    "delivery_options": "Ask which shipping methods or speeds are available, before "
                        "choosing. Compare delivery_period, which asks how long",
    "delivery_period": "Ask how long delivery takes in general. Compare track_order, "
                       "which asks where one specific shipment is now",
    "edit_account": "Change details on an existing account, such as name or email",
    "get_invoice": "Ask to be sent or issued an invoice. Compare check_invoice, "
                   "which queries one that exists",
    "get_refund": "Ask for money back when none has been promised yet. Compare "
                  "track_refund, where a refund is already expected",
    "newsletter_subscription": "Subscribe to or unsubscribe from the newsletter",
    "payment_issue": "A payment failed, was declined, or was charged wrongly. "
                     "Compare check_payment_methods, which only asks what is accepted",
    "place_order": "Buy something, or ask how to. Thin vocabulary coverage: "
                   "'I want to place a new order' misroutes, so treat low "
                   "confidence here as unresolved rather than as this class",
    "recover_password": "Reset or recover a forgotten password. Compare "
                        "registration_problems, which is about signing up",
    "registration_problems": "Signing up failed or is blocked. Compare "
                             "create_account, which is the plain request",
    "review": "Leave, change or ask about a product or service review",
    "set_up_shipping_address": "Add a shipping address for the first time. Compare "
                               "change_shipping_address, which replaces an existing one",
    "switch_account": "Move between accounts the customer already has. Lowest "
                      "coverage class at a 0.9 gate (0.44), so a global high "
                      "threshold discards most correct answers here",
    "track_order": "Ask where one specific shipment is now. Trained on parcel, "
                   "package and shipment as well as order, since v2 keyed on the "
                   "noun and missed every phrasing that avoided the word 'order'",
    "track_refund": "A refund is already expected and the customer asks where it "
                    "is. Compare get_refund, which asks for one in the first place",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="write the pack to src/jeffy/pack/support_ticket_v3")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    sp = json.loads(SPLIT.read_text())
    train, test = sp["train"], sp["test"]
    tr_t, tr_y = flatten(train)
    te_t, te_y = flatten(test)
    print(f"shipped split: train {len(tr_t)}, test {len(te_t)}, "
          f"{len(set(tr_y))} intents")

    # The augmentation, exactly as validated by the experiment.
    extra_t, extra_y = [], []
    aug = augment_track_order(train["track_order"], rng)
    extra_t += aug + TRACK_SEED
    extra_y += ["track_order"] * (len(aug) + len(TRACK_SEED))
    extra_t += [t for _, t in REFUND_SEED]
    extra_y += [y for y, _ in REFUND_SEED]
    print(f"augmentation: +{len(aug)} noun-swapped, +{len(TRACK_SEED)} written, "
          f"+{len(REFUND_SEED)} refund = {len(extra_t)} rows")

    hold_all = set(TRACK_HOLDOUT) | {t for _, t in REFUND_HOLDOUT}
    leak = (set(extra_t) & set(te_t)) | (hold_all & set(tr_t + extra_t))
    if leak:
        raise SystemExit(f"holdout or test leakage on {len(leak)} texts, aborting")
    print("no leakage into the test split or the holdouts")

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(ENCODER)
    hold_t = TRACK_HOLDOUT + [t for _, t in REFUND_HOLDOUT]
    hold_y = ["track_order"] * len(TRACK_HOLDOUT) + [y for y, _ in REFUND_HOLDOUT]
    t0 = time.perf_counter()
    E = {k: enc.encode(v, batch_size=256, show_progress_bar=False)
         for k, v in {"tr": tr_t, "aug": tr_t + extra_t, "te": te_t,
                      "hold": hold_t, "para": PARAPHRASES, "oos": OOS}.items()}
    print(f"encoded in {time.perf_counter() - t0:.1f}s")

    results = {"encoder": ENCODER, "C": C, "seed": SEED,
               "oos_threshold": OOS_THRESHOLD, "license": LICENSE, "runs": {}}

    for name, emb, ys in (("before", E["tr"], tr_y),
                          ("after", E["aug"], tr_y + extra_y)):
        clf, sc = fit(emb, ys)
        held, _ = score(clf, sc, E["te"], te_y)   # score returns (dict, preds)
        hold = predict_rows(clf, sc, E["hold"], hold_t, hold_y)
        n_ok = sum(1 for r in hold if r.get("correct"))
        conf = clf.predict_proba(sc.transform(E["oos"])).max(axis=1)
        caught = int((conf < OOS_THRESHOLD).sum())
        pconf = clf.predict_proba(sc.transform(E["para"])).max(axis=1)
        results["runs"][name] = {
            "n_train": len(ys), "held_out_accuracy": held["accuracy"],
            "held_out_macro_f1": held["macro_f1"],
            "holdout_correct": f"{n_ok}/{len(hold_t)}",
            "oos_caught_at_threshold": f"{caught}/{len(OOS)}",
            "valid_held_back": f"{int((pconf < OOS_THRESHOLD).sum())}/{len(PARAPHRASES)}",
            "holdout_rows": hold,
        }
        print(f"\n{name:<7} train {len(ys):>5}  held-out {held}  "
              f"holdout {n_ok}/{len(hold_t)}  "
              f"oos {caught}/{len(OOS)} caught  "
              f"valid held {int((pconf < OOS_THRESHOLD).sum())}/{len(PARAPHRASES)}")
        if name == "after":
            clf_f, sc_f = clf, sc

    (OUT / "pack_build.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {OUT}/pack_build.json")

    if not args.write:
        print("\nmeasure only; pass --write to build the pack")
        return

    from jeffy.model_pack import (ArtifactManifest, coef_hash, save_artifact,
                                  scaler_hash)
    classes = list(clf_f.classes_)
    missing = set(classes) - set(LABEL_DESCRIPTIONS)
    if missing:
        raise SystemExit(f"no label description for {sorted(missing)}")
    a = results["runs"]["after"]
    manifest = ArtifactManifest(
        dataset="support_ticket_v3",
        description="Customer support ticket routing, 27 fine-grained intents",
        n_classes=len(classes), task_type="choice", encoder=ENCODER,
        encoder_dim=int(E["aug"].shape[1]),
        labels={c: c for c in classes}, class_order=classes,
        hf_source="bitext/Bitext-customer-support-llm-chatbot-training-dataset "
                  "(intent column, not category), augmented: see "
                  "scripts/build_support_ticket_v3.py",
        hf_config=None, hf_revision=None, license=LICENSE,
        train_examples=a["n_train"], train_accuracy=1.0,
        test_examples=len(te_t), test_accuracy=a["held_out_accuracy"],
        scaler_mean_hash=scaler_hash(sc_f), classifier_coef_hash=coef_hash(clf_f),
        label_descriptions=LABEL_DESCRIPTIONS,
        oos_threshold=OOS_THRESHOLD,
    )
    save_artifact(PACK, "support_ticket_v3", clf_f, sc_f, manifest)
    print(f"wrote pack to {PACK / 'support_ticket_v3'}  "
          f"(oos_threshold={OOS_THRESHOLD}, license={LICENSE})")


if __name__ == "__main__":
    main()
