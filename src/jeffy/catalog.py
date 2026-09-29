"""Pretrained capability catalog.

Each capability maps a human-readable task ID to:
- The dataset and labels used for training
- The learning store task signature (for artifact lookup)
- Evaluation results on held-out test sets
- Encoder dependency and preprocessing
"""

from dataclasses import dataclass, field


@dataclass
class Capability:
    task_id: str
    name: str
    description: str
    dataset: str
    labels: dict[str, str]  # internal_label -> human description
    n_classes: int
    task_type: str  # "choice", "noul"
    hf_source: str
    license: str
    encoder: str
    train_examples: int
    # Filled in by evaluation
    test_accuracy: float | None = None
    test_macro_f1: float | None = None
    test_examples: int | None = None
    train_accuracy: float | None = None
    # Artifact linkage
    task_signature: str | None = None  # hash for learning store lookup
    artifact_path: str | None = None


# Label mappings for each dataset.
# These decode the integer labels stored in the knowledge base.
LABEL_MAPS: dict[str, dict[str, str]] = {
    "banking77": {str(i): name for i, name in enumerate([
        "activate_my_card", "age_limit", "apple_pay_or_google_pay",
        "atm_support", "automatic_top_up", "balance_not_updated_after_bank_transfer",
        "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed",
        "cancel_transfer", "card_about_to_expire", "card_acceptance",
        "card_arrival", "card_delivery_estimate", "card_linking",
        "card_not_working", "card_payment_fee_charged", "card_payment_not_recognised",
        "card_payment_wrong_exchange_rate", "card_swallowed", "cash_withdrawal_charge",
        "cash_withdrawal_not_recognised", "change_pin", "compromised_card",
        "contactless_not_working", "country_support", "declined_card_payment",
        "declined_cash_withdrawal", "declined_transfer", "direct_debit_payment_not_recognised",
        "disposable_card_limits", "edit_personal_details", "exchange_charge",
        "exchange_rate", "exchange_via_app", "extra_charge_on_statement",
        "failed_transfer", "fiat_currency_support", "freeze_card",
        "getting_spare_card", "getting_virtual_card", "lost_or_stolen_card",
        "lost_or_stolen_phone", "order_physical_card", "passcode_forgotten",
        "pending_card_payment", "pending_cash_withdrawal", "pending_top_up",
        "pending_transfer", "pin_blocked", "receiving_money",
        "Refund_not_showing_up", "request_refund", "reverted_card_payment?",
        "supported_cards_and_currencies", "terminate_account", "top_up_by_bank_transfer_charge",
        "top_up_by_card_charge", "top_up_by_cash_or_cheque", "top_up_failed",
        "top_up_limits", "top_up_reverted", "topping_up_by_card",
        "transaction_charged_twice", "transfer_fee_charged", "transfer_into_account",
        "transfer_not_received_by_recipient", "transfer_timing", "unable_to_verify_identity",
        "verify_my_identity", "verify_source_of_funds", "verify_top_up",
        "virtual_card_not_working", "visa_or_mastercard", "why_verify_identity",
        "wrong_amount_of_cash_received", "wrong_exchange_rate_for_cash_withdrawal",
    ])},
    "ag_news": {
        "0": "World news",
        "1": "Sports",
        "2": "Business",
        "3": "Science/Technology",
    },
    "dbpedia": {str(i): name for i, name in enumerate([
        "Company", "EducationalInstitution", "Artist", "Athlete",
        "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace",
        "Village", "Animal", "Plant", "Album", "Film", "WrittenWork",
    ])},
    "sst2": {
        "0": "negative",
        "1": "positive",
    },
    "emotion": {str(i): name for i, name in enumerate([
        "sadness", "joy", "love", "anger", "fear", "surprise",
    ])},
    "imdb": {
        "0": "negative",
        "1": "positive",
    },
    "sms_spam": {
        "0": "ham (not spam)",
        "1": "spam",
    },
    "snli": {
        "0": "entailment",
        "1": "neutral",
        "2": "contradiction",
    },
    "tweet_eval_sentiment": {
        "0": "negative",
        "1": "neutral",
        "2": "positive",
    },
    "tweet_eval_emotion": {str(i): name for i, name in enumerate([
        "anger", "joy", "optimism", "sadness",
    ])},
    "tweet_eval_offensive": {
        "0": "not offensive",
        "1": "offensive",
    },
}

# Known dataset licenses
LICENSES: dict[str, str] = {
    "banking77": "CC BY 4.0",
    "clinc_oos": "CC BY 3.0",
    "massive_intent": "CC BY 4.0",
    "ag_news": "Academic / non-commercial",
    "dbpedia": "CC BY-SA 3.0",
    "sst2": "Stanford academic license",
    "emotion": "Academic",
    "imdb": "Academic / non-commercial",
    "sms_spam": "CC BY 4.0",
    "snli": "CC BY-SA 4.0",
    "tweet_eval_sentiment": "Twitter TOS / academic",
    "tweet_eval_emotion": "Twitter TOS / academic",
    "tweet_eval_offensive": "Twitter TOS / academic",
}

# Datasets we have in the knowledge base (from train_knowledge_base.py DATASETS dict)
# with enough metadata to build capabilities
DATASET_CONFIGS = {
    "banking77": {"description": "Banking customer service intent detection", "n_classes": 77, "task_type": "choice"},
    "clinc_oos": {"description": "Intent detection with out-of-scope", "n_classes": 151, "task_type": "choice"},
    "massive_intent": {"description": "Amazon MASSIVE voice command intents", "n_classes": 60, "task_type": "choice"},
    "ag_news": {"description": "News article topic classification", "n_classes": 4, "task_type": "choice"},
    "dbpedia": {"description": "Wikipedia article ontology classification", "n_classes": 14, "task_type": "choice"},
    "sst2": {"description": "Movie review sentiment (positive/negative)", "n_classes": 2, "task_type": "noul"},
    "emotion": {"description": "Text emotion detection", "n_classes": 6, "task_type": "choice"},
    "imdb": {"description": "Movie review sentiment (positive/negative)", "n_classes": 2, "task_type": "noul"},
    "sms_spam": {"description": "SMS spam detection", "n_classes": 2, "task_type": "noul"},
    "snli": {"description": "Natural language inference", "n_classes": 3, "task_type": "choice"},
    "tweet_eval_sentiment": {"description": "Tweet sentiment analysis", "n_classes": 3, "task_type": "choice"},
    "tweet_eval_emotion": {"description": "Tweet emotion detection", "n_classes": 4, "task_type": "choice"},
    "tweet_eval_offensive": {"description": "Offensive language detection", "n_classes": 2, "task_type": "noul"},
}

ENCODER = "BAAI/bge-large-en-v1.5"
ENCODER_DIM = 1024
