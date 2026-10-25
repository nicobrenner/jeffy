"""Side-by-side generation from two Tiny Oracle checkpoints on novel prompts.

val_loss did not predict quality for v4 (lower loss, word salad). This produces
a markdown file for a human to judge instead.

Usage: python3 eval_oracle_side_by_side.py checkpoints/best_lm_oracle_v2.pt checkpoints/best_lm_oracle_v5.pt
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from train_lm import CausalTransformer, CustomTokenizerWrapper

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SYSTEM = (
    "<|system|> You are Tiny Oracle, a mass of 2 million parameters. "
    "You have strong opinions and limited knowledge. Keep responses short.\n"
)
PROMPTS = [
    "What are you?",
    "Tell me something interesting",
    "What do you think about AI?",
    "How smart are you?",
    "What's the meaning of life?",
    "Can you help me with my taxes?",
    "Do you dream?",
    "What's your favorite food?",
    "Explain quantum physics",
    "Are you better than ChatGPT?",
    "What should I have for dinner?",
    "Tell me a joke",
    "What's the weather like?",
    "Do you have feelings?",
    "How many parameters do you have?",
    "What happens when you run out of memory?",
    "Give me some advice",
    "What's 2 plus 2?",
    "Who made you?",
    "What are you afraid of?",
]
SAMPLES_PER_PROMPT = 2


def load(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = dict(ckpt["config"])
    cfg.setdefault("n_iterations", 1)
    model = CausalTransformer(**cfg, dropout=0.0).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg, ckpt.get("best_val_loss", ckpt.get("val_loss"))


@torch.no_grad()
def generate(model, tok, prompt, seed, max_tokens=60, temperature=0.8, top_k=40):
    torch.manual_seed(seed)
    ids = list(tok.encode(SYSTEM + f"<|user|> {prompt}\n<|oracle|>"))
    out = []
    for _ in range(max_tokens):
        x = torch.tensor([ids[-model.max_seq_len:]], dtype=torch.long, device=DEVICE)
        logits = model(x)[0, -1, :] / temperature
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[-1]] = float("-inf")
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
        t = tok._tok.id_to_token(nxt) or ""
        if t in ("<|user|>", "<|system|>", "<eos>") or nxt == tok.eos_token_id:
            break
        ids.append(nxt)
        out.append(nxt)
    return tok.decode(out).strip()


def main():
    path_a, path_b = sys.argv[1], sys.argv[2]
    name_a, name_b = Path(path_a).stem.replace("best_lm_", ""), Path(path_b).stem.replace("best_lm_", "")
    tok = CustomTokenizerWrapper("tokenizer_chat_v2.json")
    model_a, cfg_a, vl_a = load(path_a)
    model_b, cfg_b, vl_b = load(path_b)

    lines = [f"# {name_a} vs {name_b}\n"]
    lines.append(f"| | {name_a} | {name_b} |\n|---|---|---|")
    lines.append(f"| layers x iters | {cfg_a['n_layers']}x{cfg_a['n_iterations']} | {cfg_b['n_layers']}x{cfg_b['n_iterations']} |")
    lines.append(f"| d_model | {cfg_a['d_model']} | {cfg_b['d_model']} |")
    lines.append(f"| best val_loss | {vl_a:.4f} | {vl_b:.4f} |\n")
    lines.append("Same seed per row, so sampling noise is matched across models.\n")

    for p in PROMPTS:
        lines.append(f"## {p}\n")
        lines.append(f"| seed | {name_a} | {name_b} |\n|---|---|---|")
        for s in range(SAMPLES_PER_PROMPT):
            a = generate(model_a, tok, p, seed=s).replace("|", "/").replace("\n", " ")
            b = generate(model_b, tok, p, seed=s).replace("|", "/").replace("\n", " ")
            lines.append(f"| {s} | {a} | {b} |")
        lines.append("")
        print(f"done: {p}", flush=True)

    out = Path(f"pile_runs/rope_byte/oracle_{name_a}_vs_{name_b}.md")
    out.write_text("\n".join(lines))
    print(f"\nWrote {out.resolve()}")


if __name__ == "__main__":
    main()
