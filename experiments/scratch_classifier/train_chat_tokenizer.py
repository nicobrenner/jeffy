"""Train a BPE tokenizer on the chat data with special chat tokens."""

import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

def main():
    # Collect all text from chat data
    texts = []
    for split in ['data/chat_train.jsonl', 'data/chat_val.jsonl']:
        with open(split) as f:
            for line in f:
                texts.append(json.loads(line)['text'])

    print(f"Training tokenizer on {len(texts)} conversations")

    # Define special tokens
    special_tokens = [
        "<pad>", "<unk>", "<bos>", "<eos>",
        "<|system|>", "<|user|>", "<|oracle|>",
    ]

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=4096,
        special_tokens=special_tokens,
        min_frequency=3,
        show_progress=True,
    )

    tokenizer.train_from_iterator(texts, trainer=trainer)

    out_path = "tokenizer_chat.json"
    tokenizer.save(out_path)
    print(f"Saved to {out_path}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    # Test it
    print("\n=== Tests ===")
    for special in special_tokens[4:]:
        enc = tokenizer.encode(special)
        print(f"{special} -> id={enc.ids[0] if len(enc.ids)==1 else enc.ids}, tokens={enc.tokens}")

    test = "<|system|> You are Tiny Oracle. <|user|> What are you? <|oracle|> I'm 2 million parameters."
    enc = tokenizer.encode(test)
    print(f"\nTest: {test}")
    print(f"Tokens ({len(enc.ids)}): {enc.tokens}")
    print(f"Decoded: {tokenizer.decode(enc.ids)}")

    for phrase in ['How are you?', 'parameters', 'megabytes', 'neural network', 'browser', 'toaster']:
        enc = tokenizer.encode(phrase)
        print(f"{phrase:25s} -> {len(enc.ids)} tokens: {enc.tokens}")


if __name__ == "__main__":
    main()
