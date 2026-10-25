"""Export Tiny Oracle checkpoint to ONNX for browser deployment."""
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import numpy as np
from train_lm import CausalTransformer, CustomTokenizerWrapper


class OnnxFriendlyModel(nn.Module):
    """Rewrites the forward pass to avoid nn.MultiheadAttention's static reshapes."""

    def __init__(self, orig: CausalTransformer):
        super().__init__()
        self.token_emb = orig.token_emb
        self.pos_emb = orig.pos_emb
        self.norm = orig.norm
        self.lm_head = orig.lm_head

        cfg = {
            "d_model": orig.pos_emb.embedding_dim,
            "n_heads": orig.transformer.layers[0].self_attn.num_heads,
            "n_layers": len(orig.transformer.layers),
        }
        self.d_model = cfg["d_model"]
        self.n_heads = cfg["n_heads"]
        self.head_dim = self.d_model // self.n_heads

        self.layers = nn.ModuleList()
        for layer in orig.transformer.layers:
            self.layers.append(layer)

    def forward(self, input_ids):
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        causal_mask = torch.triu(
            torch.full((S, S), float("-inf"), device=input_ids.device), diagonal=1
        )

        for layer in self.layers:
            sa = layer.self_attn

            # Project Q, K, V from x (post-norm: no pre-norm)
            qkv = F.linear(x, sa.in_proj_weight, sa.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)

            # Reshape for multi-head: (B, S, D) -> (B, H, S, Dh)
            q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

            # Scaled dot-product attention
            scale = math.sqrt(self.head_dim)
            attn = torch.matmul(q, k.transpose(-2, -1)) / scale
            attn = attn + causal_mask.unsqueeze(0).unsqueeze(0)
            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(attn, v)

            # Reshape back: (B, H, S, Dh) -> (B, S, D)
            out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)
            out = F.linear(out, sa.out_proj.weight, sa.out_proj.bias)

            # Post-norm: residual then norm
            x = layer.norm1(x + out)

            # Feedforward with GELU (matching original activation)
            ff_out = layer.linear2(F.gelu(layer.linear1(x)))
            x = layer.norm2(x + ff_out)

        x = self.norm(x)
        return self.lm_head(x)


def export(ckpt_path, tokenizer_path, output_path):
    device = "cpu"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    orig = CausalTransformer(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        max_seq_len=cfg["max_seq_len"],
        tie_weights=cfg["tie_weights"],
    )
    orig.load_state_dict(ckpt["model_state"])
    orig.eval()

    model = OnnxFriendlyModel(orig)
    model.eval()

    # Verify outputs match
    test_in = torch.randint(0, cfg["vocab_size"], (1, 20), dtype=torch.long)
    with torch.no_grad():
        orig_out = orig(test_in)
        wrap_out = model(test_in)
    diff = (orig_out - wrap_out).abs().max().item()
    print(f"Wrapper vs original max diff: {diff:.6e}")
    assert diff < 1e-4, f"Outputs diverge: {diff}"

    dummy_input = torch.randint(0, cfg["vocab_size"], (1, 32), dtype=torch.long)

    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch", 1: "seq_len"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Exported: {output_path} ({size_mb:.1f} MB)")
    print(f"  Config: d={cfg['d_model']}, L={cfg['n_layers']}, h={cfg['n_heads']}, ff={cfg['d_ff']}")
    print(f"  Vocab: {cfg['vocab_size']}, Max seq: {cfg['max_seq_len']}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # Verify with onnxruntime
    import onnxruntime as ort
    sess = ort.InferenceSession(output_path)
    test_input = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
    result = sess.run(None, {"input_ids": test_input})
    print(f"  ORT verify: output shape {result[0].shape}, dtype {result[0].dtype}")

    # Quick generation test
    tokenizer = CustomTokenizerWrapper(tokenizer_path)
    prompt = "<|system|> You are Tiny Oracle.\n<|user|> Hi\n<|oracle|>"
    ids = tokenizer.encode(prompt)
    for _ in range(30):
        inp = np.array([ids], dtype=np.int64)
        logits = sess.run(None, {"input_ids": inp})[0]
        next_logits = logits[0, -1, :]
        next_logits = next_logits / 0.8
        probs = np.exp(next_logits - np.max(next_logits))
        probs = probs / probs.sum()
        next_id = np.random.choice(len(probs), p=probs)
        ids.append(int(next_id))
        token = tokenizer.decode([int(next_id)])
        if "<|user|>" in token or "<|system|>" in token:
            break
    response = tokenizer.decode(ids)
    oracle_part = response.split("<|oracle|>")[-1].split("<|user|>")[0].split("<|system|>")[0].strip()
    print(f"  Test generation: \"{oracle_part[:150]}\"")


if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_lm_oracle_v2.pt")
    parser.add_argument("--tokenizer", default="tokenizer_chat_v2.json")
    parser.add_argument("--output", default="oracle_v2.onnx")
    args = parser.parse_args()
    export(args.checkpoint, args.tokenizer, args.output)
