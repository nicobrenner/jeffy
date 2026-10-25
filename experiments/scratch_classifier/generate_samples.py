"""Generate text samples from trained RoPE byte-level models.

Shows what these tiny models actually produce at different scales,
demonstrating the "generation quality ladder" from d=64 to d=128.
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train_rope_byte import RoPEByteLM


def load_model(checkpoint_path, device='cpu'):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']
    model = RoPEByteLM(
        vocab_size=ckpt.get('vocab_size', 256),
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        max_seq_len=256,
        tie_weights=True,
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, ckpt.get('val_ppl', '?'), config


@torch.no_grad()
def generate(model, prompt_bytes, max_new_bytes=512, temperature=0.8, top_k=40, device='cpu'):
    tokens = torch.tensor([list(prompt_bytes)], dtype=torch.long, device=device)
    generated = list(prompt_bytes)

    for _ in range(max_new_bytes):
        if tokens.shape[1] >= model.max_seq_len:
            tokens = tokens[:, -model.max_seq_len:]
        logits = model(tokens)
        next_logits = logits[0, -1, :] / temperature

        if top_k > 0:
            v, _ = torch.topk(next_logits, top_k)
            next_logits[next_logits < v[-1]] = float('-inf')

        probs = F.softmax(next_logits, dim=-1)
        next_byte = torch.multinomial(probs, 1).item()
        generated.append(next_byte)
        tokens = torch.cat([tokens, torch.tensor([[next_byte]], device=device)], dim=1)

    return bytes(generated)


def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    models = {}
    model_paths = {
        'd=64 (215K, 92% reasoning, PPL 5.46)': 'pile_runs/rope_byte/run_d64_tp1800/best_model.pt',
        'd=128 (824K, 96% reasoning, PPL 3.70)': 'pile_runs/rope_byte/run_d128_tp1800/best_model.pt',
    }

    for name, path in model_paths.items():
        p = Path(path)
        if p.exists():
            model, ppl, config = load_model(p, device)
            models[name] = model
            print(f"Loaded {name} (val_ppl={ppl})")
        else:
            print(f"SKIP {name}: {path} not found")

    if not models:
        print("No models found!")
        return

    prompts = [
        (b"The ", "General text"),
        (b"In the year ", "Temporal/narrative"),
        (b"def ", "Code/Python"),
        (b"http", "URL/web"),
        (b"1. ", "List/structured"),
        (b"Once upon a time", "Story"),
        (b"import ", "Code import"),
        (b"The president ", "News-like"),
    ]

    temperatures = [0.8, 1.0]

    for temp in temperatures:
        print(f"\n{'='*80}")
        print(f"  TEMPERATURE = {temp}")
        print(f"{'='*80}")

        for prompt_bytes, desc in prompts:
            prompt_str = prompt_bytes.decode('utf-8', errors='replace')
            print(f"\n{'─'*80}")
            print(f"  Prompt: {repr(prompt_str):30s}  ({desc})")
            print(f"{'─'*80}")

            for name, model in models.items():
                output = generate(model, prompt_bytes, max_new_bytes=384,
                                  temperature=temp, top_k=40, device=device)
                text = output.decode('utf-8', errors='replace')
                # Clean up for display: show first 400 chars
                display = text[:400]
                # Replace non-printable chars except newline/tab
                display = ''.join(c if c.isprintable() or c in '\n\t' else '·' for c in display)
                print(f"\n  [{name}]")
                for line in display.split('\n')[:12]:  # limit displayed lines
                    print(f"    {line}")

    # Also do unconditional generation (empty / single-byte prompt)
    print(f"\n{'='*80}")
    print(f"  UNCONDITIONAL GENERATION (temp=0.8)")
    print(f"{'='*80}")

    for name, model in models.items():
        print(f"\n  [{name}]")
        # Start from a single space byte
        output = generate(model, b" ", max_new_bytes=512,
                          temperature=0.8, top_k=40, device=device)
        text = output.decode('utf-8', errors='replace')
        display = ''.join(c if c.isprintable() or c in '\n\t' else '·' for c in text[:500])
        for line in display.split('\n')[:15]:
            print(f"    {line}")


if __name__ == '__main__':
    main()
