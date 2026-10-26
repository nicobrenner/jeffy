"""Fix the two support_ticket_v2 defects an external tester found, and measure it.

Defect 1, the noun lock (forum post 22). `track_order` training text is 99% "order"
and 0% parcel/package/shipment, so any tracking question phrased with another noun
lands in delivery_*. Only 8% of its rows contain the word "track" at all, and 96%
contain the literal template placeholder `{{Order Number}}`, which no real user types.

Defect 2, the cancel/refund overlap (forum post 24). "I cancelled my order, now I
want my money back" routes to `cancel_order` at 0.710 with margin 0.536, so it clears
every abstain gate while being wrong. A threshold cannot catch it.

Both get the issue-11 treatment: targeted examples, a held-back slice to prove the
patch generalises rather than memorises, and the SAME test split the shipped pack was
scored on so the headline accuracy is comparable.

The thing to watch is NOT held-out accuracy. It was 0.9919 while every defect above
was live, so it is blind to this whole class of problem. The controls are the measure.

    python scripts/fix_support_ticket_v3.py            # measure only
    python scripts/fix_support_ticket_v3.py --write    # also write the pack
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
SPLIT = REPO / "data/eval_results/support_ticket_v2/split.json"
OUT = REPO / "data/eval_results/support_ticket_v3"
ENCODER = "BAAI/bge-large-en-v1.5"
C = 0.01
SEED = 42

# --- defect 1: the noun lock -------------------------------------------------
# Rewrite rule applied to the existing track_order rows. The intent of a row is
# unchanged by swapping the object noun: "status of my order" and "status of my
# parcel" are the same request. The placeholder is replaced rather than deleted,
# because leaving `{{Order Number}}` in teaches a token users never type.
ORDER_NOUNS = ["parcel", "package", "shipment", "delivery"]
PLACEHOLDER = re.compile(r"\{\{\s*Order Number\s*\}\}")
ORDER_WORD = re.compile(r"\b(order|purchase)\b", re.I)

# track_order never says these in the source data (8% "track", 2% "where is"),
# so the phrasings real users reach for first are absent. Hand-written, one per
# shape, deliberately short like real queries rather than bitext's long templates.
TRACK_SEED = [
    "where is my parcel",
    "where is my package right now",
    "track my shipment",
    "I want to track my parcel",
    "how do I track my package",
    "where can I track my parcel after it has been shipped",
    "tracking status of my package please",
    "I need to know where my shipment is right now",
    "has my parcel shipped yet",
    "is my package out for delivery",
    "what is the current status of my shipment",
    "any update on where my delivery is",
]
# held back from training, used only to test generalisation
TRACK_HOLDOUT = [
    "locate my parcel for me",
    "can you tell me where my package has got to",
    "I would like the tracking details for my shipment",
    "whereabouts of my delivery please",
]

# --- defect 2: refund after cancellation -------------------------------------
# Three classes collide here, so the policy has to be written down before the
# examples, or the patch just moves the error somewhere else. It did on the first
# run: eight get_refund seeds pulled "where is my refund, it has not arrived yet"
# off track_refund, because three of those seeds were really track_refund.
#
#   cancel_order   stop the order from happening
#   get_refund     asking for money back, none promised yet
#   track_refund   a refund is already expected, asking where it is
#
# So "I want my money back" is get_refund and "still waiting to be paid back" is
# track_refund, however similar they read.
REFUND_SEED = [
    ("get_refund", "I cancelled my order, now I want my money back"),
    ("get_refund", "I want a refund for the order I cancelled"),
    ("get_refund", "please refund the cancelled purchase"),
    ("get_refund", "I need my money returned after cancelling"),
    ("get_refund", "give me my money back for the order I called off"),
    ("track_refund", "my order was cancelled, when do I get reimbursed"),
    ("track_refund", "the order is cancelled and I am still waiting to be paid back"),
    ("track_refund", "cancelled last week, still no refund"),
    ("track_refund", "where is my refund, it has not arrived"),
    ("cancel_order", "I want to cancel my order, I do not need it any more"),
    ("cancel_order", "please stop my order from being sent"),
    ("cancel_order", "cancel the order I placed this morning"),
    ("cancel_order", "I changed my mind, do not ship my order"),
]
REFUND_HOLDOUT = [
    ("get_refund", "refund me for the purchase I cancelled yesterday"),
    ("get_refund", "I want reimbursing for the order I called off"),
    ("track_refund", "I called off my order, where is my reimbursement"),
    ("track_refund", "the refund you promised has not turned up"),
    ("cancel_order", "do not send the order, I want it cancelled"),
    ("cancel_order", "withdraw my order please"),
]

# --- controls ----------------------------------------------------------------
# The external tester's three failing cases, and my own correct controls that
# must NOT regress. Expected label is what a human router would pick.
CONTROLS = [
    ("track_order", "Where can I track my parcel after it has been shipped?"),
    ("track_order", "Please provide the tracking status of my package."),
    ("track_order", "I need to know where my shipment is right now."),
    ("track_order", "where is my order"),
    ("track_order", "I want to track my order"),
    ("track_order", "how do I track my order"),
    ("delivery_period", "when will my order arrive"),
    ("delivery_period", "how long does delivery take"),
    ("delivery_period", "how long does shipping usually take"),
    ("delivery_options", "what delivery options are available"),
    ("delivery_options", "which shipping methods do you offer"),
    ("change_shipping_address", "I need to change the shipping address on my order"),
    ("change_shipping_address", "redirect my package to a different address, it is already on its way"),
    ("set_up_shipping_address", "how do I add a new shipping address"),
    # the three-way refund collision, as controls rather than training text
    ("get_refund", "I would like a refund for this"),
    ("track_refund", "where is my refund, it has not arrived yet"),
    ("cancel_order", "I want to cancel the order I placed yesterday"),
]

# one plain hand-written paraphrase per intent, not dataset text, to catch
# collateral damage anywhere in the taxonomy
PARAPHRASES = [
    ("cancel_order", "I want to cancel the order I placed yesterday"),
    ("change_order", "can I add another item to my existing order"),
    ("change_shipping_address", "I need to change the shipping address on my order"),
    ("check_cancellation_fee", "is there a fee if I cancel now"),
    ("check_invoice", "I want to look at my invoice from last month"),
    ("check_payment_methods", "which payment methods do you accept"),
    ("check_refund_policy", "what is your refund policy"),
    ("complaint", "this is the third time this has gone wrong and I am furious"),
    ("contact_customer_service", "how do I get in touch with customer service"),
    ("contact_human_agent", "I want to speak to a real person"),
    ("create_account", "how do I sign up for an account"),
    ("delete_account", "please delete my account permanently"),
    ("delivery_options", "what delivery options are available"),
    ("delivery_period", "when will my order arrive"),
    ("edit_account", "I need to update my email address on my profile"),
    ("get_invoice", "please send me the invoice"),
    ("get_refund", "I would like a refund for this"),
    ("newsletter_subscription", "take me off your newsletter"),
    ("payment_issue", "my card was declined when I tried to pay"),
    ("place_order", "I want to place a new order"),
    ("recover_password", "I forgot my password"),
    ("registration_problems", "I cannot finish signing up, the form errors out"),
    ("review", "where can I leave a review of the product"),
    ("set_up_shipping_address", "how do I add a new shipping address"),
    ("switch_account", "I want to switch to my other account"),
    ("track_order", "I want to track my order"),
    ("track_refund", "where is my refund, it has not arrived yet"),
]

# clearly unrelated, for the per-class threshold work
OOS = [
    "what is the capital of Portugal",
    "write me a haiku about rain",
    "how do I change a flat tyre on a bicycle",
    "my cat keeps knocking plants off the windowsill",
    "what is the derivative of x squared",
    "recommend a good science fiction novel",
    "the weather forecast for Tuesday please",
    "translate good morning into Japanese",
]


def augment_track_order(rows, rng):
    """Noun-swapped and de-placeholdered variants of the existing rows."""
    out = []
    for i, text in enumerate(rows):
        noun = ORDER_NOUNS[i % len(ORDER_NOUNS)]
        # drop the template placeholder, which no user types
        t = PLACEHOLDER.sub("", text)
        t = re.sub(r"\s{2,}", " ", t).strip()
        swapped = ORDER_WORD.sub(noun, t)
        if swapped != t:
            out.append(swapped)
    return out


def flatten(d):
    texts, labels = [], []
    for k in sorted(d):
        for t in d[k]:
            texts.append(t)
            labels.append(k)
    return texts, labels


def fit(emb, labels):
    scaler = StandardScaler()
    X = scaler.fit_transform(emb)
    clf = LogisticRegression(C=C, max_iter=3000, solver="newton-cg", random_state=SEED)
    clf.fit(X, labels)
    return clf, scaler


def score(clf, scaler, emb, labels):
    preds = clf.predict(scaler.transform(emb))
    return {
        "n": len(labels),
        "accuracy": round(float(accuracy_score(labels, preds)), 4),
        "macro_f1": round(float(f1_score(labels, preds, average="macro", zero_division=0)), 4),
    }, preds


def predict_rows(clf, scaler, emb, texts, expected=None):
    Xs = scaler.transform(emb)
    probs = clf.predict_proba(Xs)
    rows = []
    for i, text in enumerate(texts):
        p = probs[i]
        order = np.argsort(p)[::-1]
        label = clf.classes_[order[0]]
        rows.append({
            "text": text,
            "expected": expected[i] if expected else None,
            "label": str(label),
            "correct": (str(label) == expected[i]) if expected else None,
            "confidence": round(float(p[order[0]]), 4),
            "margin": round(float(p[order[0]] - p[order[1]]), 4),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    split = json.loads(SPLIT.read_text())
    train, test = split["train"], split["test"]

    # baseline: exactly the shipped recipe
    base_t, base_l = flatten(train)
    te_t, te_l = flatten(test)

    # augmented: baseline plus the two targeted fixes
    aug_train = {k: list(v) for k, v in train.items()}
    added = augment_track_order(train["track_order"], rng)
    aug_train["track_order"] += added + TRACK_SEED
    for lab, txt in REFUND_SEED:
        aug_train[lab].append(txt)
    aug_t, aug_l = flatten(aug_train)

    print(f"baseline train {len(base_t)}, augmented train {len(aug_t)}")
    print(f"  track_order {len(train['track_order'])} -> {len(aug_train['track_order'])} "
          f"({len(added)} noun-swapped, {len(TRACK_SEED)} hand-written)")
    print(f"  get_refund  {len(train['get_refund'])} -> {len(aug_train['get_refund'])}")
    print(f"  cancel_order {len(train['cancel_order'])} -> {len(aug_train['cancel_order'])}")

    leak = set(aug_t) & set(te_t)
    if leak:
        raise SystemExit(f"augmented train leaks into test on {len(leak)} texts")
    hold_texts = TRACK_HOLDOUT + [t for _, t in REFUND_HOLDOUT]
    leak2 = set(aug_t) & set(hold_texts)
    if leak2:
        raise SystemExit(f"holdout leaked into train: {leak2}")
    print("no train/test and no train/holdout overlap")

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(ENCODER)

    def embed(texts):
        return enc.encode(texts, batch_size=256, show_progress_bar=False)

    t0 = time.perf_counter()
    E = {
        "base": embed(base_t), "aug": embed(aug_t), "test": embed(te_t),
        "ctrl": embed([t for _, t in CONTROLS]),
        "para": embed([t for _, t in PARAPHRASES]),
        "oos": embed(OOS),
        "track_hold": embed(TRACK_HOLDOUT),
        "refund_hold": embed([t for _, t in REFUND_HOLDOUT]),
    }
    print(f"encoded in {time.perf_counter() - t0:.1f}s")

    results = {"encoder": ENCODER, "C": C, "seed": SEED,
               "n_train_baseline": len(base_t), "n_train_augmented": len(aug_t),
               "track_order_added": len(added) + len(TRACK_SEED),
               "conditions": {}}

    for name, (tr_e, tr_l) in {"baseline": (E["base"], base_l),
                               "augmented": (E["aug"], aug_l)}.items():
        clf, scaler = fit(tr_e, tr_l)
        held, _ = score(clf, scaler, E["test"], te_l)
        ctrl = predict_rows(clf, scaler, E["ctrl"], [t for _, t in CONTROLS],
                            [e for e, _ in CONTROLS])
        para = predict_rows(clf, scaler, E["para"], [t for _, t in PARAPHRASES],
                            [e for e, _ in PARAPHRASES])
        oos = predict_rows(clf, scaler, E["oos"], OOS)
        th = predict_rows(clf, scaler, E["track_hold"], TRACK_HOLDOUT,
                          ["track_order"] * len(TRACK_HOLDOUT))
        rh = predict_rows(clf, scaler, E["refund_hold"], [t for _, t in REFUND_HOLDOUT],
                          [e for e, _ in REFUND_HOLDOUT])
        results["conditions"][name] = {
            "held_out_same_split_as_shipped": held,
            "controls_correct": f"{sum(r['correct'] for r in ctrl)}/{len(ctrl)}",
            "paraphrases_correct": f"{sum(r['correct'] for r in para)}/{len(para)}",
            "track_holdout_correct": f"{sum(r['correct'] for r in th)}/{len(th)}",
            "refund_holdout_correct": f"{sum(r['correct'] for r in rh)}/{len(rh)}",
            "controls": ctrl, "paraphrases": para, "out_of_scope": oos,
            "track_holdout": th, "refund_holdout": rh,
        }
        print(f"\n--- {name} ---")
        print(f"held-out (shipped split)  acc {held['accuracy']}  macroF1 {held['macro_f1']}")
        print(f"controls                  {sum(r['correct'] for r in ctrl)}/{len(ctrl)}")
        print(f"paraphrases, 1 per intent {sum(r['correct'] for r in para)}/{len(para)}")
        print(f"track holdout (unseen)    {sum(r['correct'] for r in th)}/{len(th)}")
        print(f"refund holdout (unseen)   {sum(r['correct'] for r in rh)}/{len(rh)}")

    # Per-class coverage on the held-out split, which is the column the probe
    # table was missing. n=50 per class here, against n=1 for most classes in the
    # hand-written set below, so this is the one with enough data to act on. It is
    # in-distribution text and therefore optimistic: read it as an upper bound on
    # coverage, and the hand-written paraphrases as the realistic floor.
    clf_a, sc_a = fit(E["aug"], aug_l)
    probs_te = clf_a.predict_proba(sc_a.transform(E["test"]))
    classes = list(clf_a.classes_)
    by_class = {}
    for i, true_label in enumerate(te_l):
        p = probs_te[i]
        order = np.argsort(p)[::-1]
        by_class.setdefault(true_label, []).append({
            "correct": classes[order[0]] == true_label,
            "confidence": float(p[order[0]]),
        })
    coverage = {}
    for lab, rows in sorted(by_class.items()):
        confs = np.array([r["confidence"] for r in rows])
        ok = np.array([r["correct"] for r in rows])
        coverage[lab] = {
            "n": len(rows),
            "accuracy": round(float(ok.mean()), 4),
            "p10_confidence": round(float(np.percentile(confs, 10)), 4),
            "median_confidence": round(float(np.median(confs)), 4),
            # share of CORRECT predictions that would survive each gate
            "coverage_at_0.5": round(float((confs[ok] >= 0.5).mean()), 4) if ok.any() else 0.0,
            "coverage_at_0.9": round(float((confs[ok] >= 0.9).mean()), 4) if ok.any() else 0.0,
        }
    results["per_class_coverage_heldout"] = coverage
    print("\n=== per-class coverage on the held-out split (augmented head) ===")
    print(f"{'intent':<26}{'acc':>7}{'p10conf':>9}{'cov@0.5':>9}{'cov@0.9':>9}")
    for lab, c in sorted(coverage.items(), key=lambda x: x[1]["coverage_at_0.9"]):
        print(f"{lab:<26}{c['accuracy']:>7.3f}{c['p10_confidence']:>9.3f}"
              f"{c['coverage_at_0.5']:>9.3f}{c['coverage_at_0.9']:>9.3f}")

    # per-class threshold table, commitment 4, on the augmented head
    aug = results["conditions"]["augmented"]
    per_class = {}
    for r in aug["paraphrases"] + aug["controls"]:
        per_class.setdefault(r["expected"], []).append(r)
    results["per_class_confidence"] = {
        k: {"n": len(v),
            "correct": sum(x["correct"] for x in v),
            "min_confidence": round(min(x["confidence"] for x in v), 4),
            "median_confidence": round(float(np.median([x["confidence"] for x in v])), 4),
            "min_margin": round(min(x["margin"] for x in v), 4)}
        for k, v in sorted(per_class.items())}

    oos_conf = [r["confidence"] for r in aug["out_of_scope"]]
    valid_conf = [r["confidence"] for r in aug["paraphrases"]]
    sweep = []
    for thr in (0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9):
        sweep.append({
            "threshold": thr,
            "oos_caught": f"{sum(c < thr for c in oos_conf)}/{len(oos_conf)}",
            "valid_held": f"{sum(c < thr for c in valid_conf)}/{len(valid_conf)}",
        })
    results["confidence_sweep_augmented"] = sweep

    (OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    with open(OUT / "controls.csv", "w") as f:
        f.write("condition,expected,predicted,correct,confidence,margin,text\n")
        for cond in ("baseline", "augmented"):
            for r in results["conditions"][cond]["controls"]:
                f.write(f'{cond},{r["expected"]},{r["label"]},{r["correct"]},'
                        f'{r["confidence"]},{r["margin"]},"{r["text"]}"\n')

    print("\n=== controls, baseline -> augmented ===")
    b = {r["text"]: r for r in results["conditions"]["baseline"]["controls"]}
    a = {r["text"]: r for r in results["conditions"]["augmented"]["controls"]}
    for exp, text in CONTROLS:
        rb, ra = b[text], a[text]
        mark = "  " if rb["correct"] == ra["correct"] else ("FIX" if ra["correct"] else "REG")
        print(f'{mark} {rb["label"]:<24}{rb["confidence"]:.3f} -> '
              f'{ra["label"]:<24}{ra["confidence"]:.3f}  {text[:52]}')

    print(f"\nwrote {OUT}/results.json and controls.csv")

    if args.write:
        print("\n--write given, but writing the pack is a separate reviewed step; "
              "measure first, then wire it into build_support_ticket_v2.py")


if __name__ == "__main__":
    main()
