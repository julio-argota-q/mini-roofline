"""
tiny_transformer.py
-------------------
A minimal GPT-style transformer forward pass in pure NumPy,
matching HuggingFace GPT-2's actual parameter structure (with QKV biases).

Purpose: derive the FLOP count for every operation from first principles.
Every function has two components:
    1. The actual computation (NumPy)
    2. A manual FLOP count that returns exactly what the math says

At the end we compare our manual count to the formulas in flops.py.
If they match, the derivation is correct.

Notation used throughout:
    B  : batch size
    S  : sequence length
    d  : model dimension
    H  : number of heads
    Dh : head dimension (d / H)
    Ff : feed-forward hidden dim (usually 4d)
    V  : vocabulary size

RULE: 1 multiply-add = 2 FLOPs = 1 MAC.
So a matmul of shape [M, K] × [K, N] costs approx 2 * M * K * N FLOPs.
Bias adds cost 1 FLOP per element — small but included for completeness.
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
#  Matrix multiplication
# ═══════════════════════════════════════════════════════════════════════════

def matmul(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Multiply A @ B and return (result, flop_count).

    A has shape [..., M, K]
    B has shape [K, N]
    Result has shape [..., M, N]

    QUESTION: how many FLOPs does this cost?

    Think about it:
      - Each output element is a dot product of K values.
      - A dot product of K values is K multiplies + (K-1) adds.
      - We approximate as K multiply-adds → 2*K FLOPs.
      - There are (product of leading dims) × M × N output elements.
    """
    result = A @ B

    leading = int(np.prod(A.shape[:-2])) if A.ndim > 2 else 1
    M = A.shape[-2]
    K = A.shape[-1]
    N = B.shape[-1]

    flops = 2 * M * N * K    # ← YOUR ANSWER

    return result, flops


# ═══════════════════════════════════════════════════════════════════════════
#  Layer normalisation
# ═══════════════════════════════════════════════════════════════════════════

def layer_norm(
    x: np.ndarray,          # [B, S, d]
    gamma: np.ndarray,      # [d]
    beta: np.ndarray,       # [d]
    eps: float = 1e-5,
) -> tuple[np.ndarray, int]:
    """
    LayerNorm per token (a vector of length d):
      1. mean       = sum(x) / d              → d adds, 1 divide
      2. centered   = x - mean                → d subs
      3. variance   = sum(centered^2) / d     → d muls + d adds + 1 divide
      4. std        = sqrt(variance + eps)    → 1 add, 1 sqrt
      5. normalised = centered / std          → d divides
      6. output     = gamma * normalised + beta → d muls + d adds

    """
    mean = x.mean(axis=-1, keepdims=True)
    centered = x - mean
    variance = (centered ** 2).mean(axis=-1, keepdims=True)
    std = np.sqrt(variance + eps)
    normalised = centered / std
    output = gamma * normalised + beta

    B, S, d = x.shape
    per_token = 7 * d + 4   # ← YOUR ANSWER
    flops = B * S * per_token

    return output, flops


# ═══════════════════════════════════════════════════════════════════════════
#  Multi-head self-attention (with QKV biases, GPT-2 style)
# ═══════════════════════════════════════════════════════════════════════════

def multi_head_attention(
    x: np.ndarray,           # [B, S, d]
    W_q: np.ndarray,         # [d, d]
    b_q: np.ndarray,         # [d]        — GPT-2 uses QKV biases
    W_k: np.ndarray,         # [d, d]
    b_k: np.ndarray,         # [d]
    W_v: np.ndarray,         # [d, d]
    b_v: np.ndarray,         # [d]
    W_o: np.ndarray,         # [d, d]
    b_o: np.ndarray,         # [d]
    H: int,                  # number of heads
) -> tuple[np.ndarray, dict]:
    """
    Multi-head self-attention as in GPT-2.

    Note on HuggingFace's implementation:
      HF's GPT-2 uses ONE fused Conv1D of shape [d, 3d] that produces
      Q, K, V concatenated. Mathematically equivalent to three separate
      [d, d] matmuls — same FLOP count — but the profiler will see ONE
      op, not three.
        - fused:     2 * B * S * d * 3d   = 6 * B * S * d²
        - separate: 3 * 2 * B * S * d * d = 6 * B * S * d²
    """
    B, S, d = x.shape
    Dh = d // H

    # ── QKV projections with biases ──
    Q, f_q = matmul(x, W_q); Q = Q + b_q
    K, f_k = matmul(x, W_k); K = K + b_k
    V, f_v = matmul(x, W_v); V = V + b_v
    flops_qkv_matmul = f_q + f_k + f_v
    flops_qkv_bias   = 3 * B * S * d
    flops_qkv        = flops_qkv_matmul + flops_qkv_bias

    # ── reshape into heads (free) ──
    Q = Q.reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
    K = K.reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
    V = V.reshape(B, S, H, Dh).transpose(0, 2, 1, 3)

    # ── attention scores ──
    scores = Q @ K.transpose(0, 1, 3, 2)   # [B, H, S, S]
    flops_scores = 2 * B * S * S * d

    # ── causal mask (autoregressive) ──
    mask = np.triu(np.ones((S, S), dtype=bool), k=1)
    scores = np.where(mask, -1e9, scores)
    flops_mask = B * H * S * S # naive assumsion

    # ── scale + softmax ──
    scores = scores / np.sqrt(Dh)
    flops_scale = B * H * S * S
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    flops_softmax = 3 * B * H * S * S 

    # ── weighted sum ──
    out = attn_weights @ V
    flops_weighted = 2 * B * H * S * S * Dh

    # ── concat heads (free) ──
    out = out.transpose(0, 2, 1, 3).reshape(B, S, d)

    # ── output projection with bias ──
    out, flops_out_matmul = matmul(out, W_o)
    out = out + b_o
    flops_out_bias = B * S * d
    flops_output   = flops_out_matmul + flops_out_bias 

    return out, {
        "qkv_projection": flops_qkv,
        "attention_scores": flops_scores,
        "mask": flops_mask,
        "scale": flops_scale,
        "softmax": flops_softmax,
        "weighted_sum": flops_weighted,
        "output_projection": flops_output,
        "total": (flops_qkv + flops_scores + flops_mask + flops_scale
                  + flops_softmax + flops_weighted + flops_output),
    }


# ═══════════════════════════════════════════════════════════════════════════
# MLP block (with biases)
# ═══════════════════════════════════════════════════════════════════════════

def mlp(
    x: np.ndarray,           # [B, S, d]
    W_1: np.ndarray,         # [d, Ff]
    b_1: np.ndarray,         # [Ff]
    W_2: np.ndarray,         # [Ff, d]
    b_2: np.ndarray,         # [d]
) -> tuple[np.ndarray, dict]:
    """
    Two-layer MLP: x → FC1 → GELU → FC2

    QUESTIONS:
      - flops_fc1: matmul [B*S, d] × [d, Ff] + bias
          matmul: 2 * B * S * d * Ff
          bias:   B * S * Ff
      - flops_gelu: ~8 FLOPs/elem × B*S*Ff  (or 0 if you ignore)
      - flops_fc2: matmul [B*S, Ff] × [Ff, d] + bias
          matmul: 2 * B * S * Ff * d
          bias:   B * S * d

    For GPT-2 (Ff=4d): matmul total = 16*B*S*d²
    """
    B, S, d = x.shape
    Ff = W_1.shape[1]

    h, f1_matmul = matmul(x, W_1)
    h = h + b_1
    f1_bias = B * S * Ff
    flops_fc1 = f1_matmul + f1_bias   # ← YOUR ANSWER

    h = 0.5 * h * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (h + 0.044715 * h ** 3)))
    flops_gelu = 8 * B * S * Ff   # ← YOUR ANSWER (or 0)

    out, f2_matmul = matmul(h, W_2)
    out = out + b_2
    f2_bias = B * S * d
    flops_fc2 = f2_matmul + f2_bias   # ← YOUR ANSWER

    return out, {
        "fc1": flops_fc1,
        "gelu": flops_gelu,
        "fc2": flops_fc2,
        "total": flops_fc1 + flops_gelu + flops_fc2,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  One transformer block (attention + MLP with residuals)
# ═══════════════════════════════════════════════════════════════════════════

def transformer_block(
    x: np.ndarray,
    params: dict,
    H: int,
) -> tuple[np.ndarray, dict]:
    """
    One pre-norm transformer block:
        x = x + attention(layernorm(x))
        x = x + mlp(layernorm(x))
    """
    x_norm, f_ln1 = layer_norm(x, params["ln_1_g"], params["ln_1_b"])
    attn_out, f_attn = multi_head_attention(
        x_norm,
        params["W_q"], params["b_q"],
        params["W_k"], params["b_k"],
        params["W_v"], params["b_v"],
        params["W_o"], params["b_o"],
        H,
    )
    x = x + attn_out
    f_residual_1 = int(np.prod(x.shape))

    x_norm, f_ln2 = layer_norm(x, params["ln_2_g"], params["ln_2_b"])
    mlp_out, f_mlp = mlp(x_norm, params["W_1"], params["b_1"], params["W_2"], params["b_2"])
    x = x + mlp_out
    f_residual_2 = int(np.prod(x.shape))

    return x, {
        "layernorm_1": f_ln1,
        "attention": f_attn["total"],
        "residual_1": f_residual_1,
        "layernorm_2": f_ln2,
        "mlp": f_mlp["total"],
        "residual_2": f_residual_2,
        "total": (f_ln1 + f_attn["total"] + f_residual_1
                  + f_ln2 + f_mlp["total"] + f_residual_2),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Full model
# ═══════════════════════════════════════════════════════════════════════════

def gpt2_forward(
    token_ids: np.ndarray,   # [B, S]
    all_params: list[dict],
    token_emb: np.ndarray,   # [V, d]
    pos_emb: np.ndarray,     # [S_max, d]
    ln_f_g: np.ndarray,
    ln_f_b: np.ndarray,
    H: int,
) -> tuple[np.ndarray, dict]:
    """
    Note on LM head cost:
      matmul [B, S, d] × [d, V] = 2*B*S*d*V FLOPs
      For GPT-2 (V=50257, d=768): ~77 MFLOPs per token.
      Often forgotten in FLOP counts.
    """
    B, S = token_ids.shape

    x = token_emb[token_ids] + pos_emb[:S]
    flops_embed = B * S * token_emb.shape[1]

    total_block_flops = 0
    for params in all_params:
        x, block_flops = transformer_block(x, params, H)
        total_block_flops += block_flops["total"]

    x, flops_ln_f = layer_norm(x, ln_f_g, ln_f_b)

    logits, flops_lm_head = matmul(x, token_emb.T)

    total = flops_embed + total_block_flops + flops_ln_f + flops_lm_head

    return logits, {
        "embeddings": flops_embed,
        "all_blocks": total_block_flops,
        "final_layernorm": flops_ln_f,
        "lm_head": flops_lm_head,
        "total": total,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Compare to flops.py formulas
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """
    Instantiate a tiny GPT-2 with random weights (now with all biases),
    run one forward pass, and compare manual counts to flops.py.
    """
    B, S = 1, 32
    L = 2
    d = 64
    H = 4
    Ff = 4 * d          # 256
    V = 100

    rng = np.random.default_rng(42)

    def rand(*shape):
        return rng.standard_normal(shape).astype(np.float32) * 0.02

    blocks = []
    for _ in range(L):
        blocks.append({
            "ln_1_g": np.ones(d),  "ln_1_b": np.zeros(d),
            "W_q": rand(d, d), "b_q": rand(d),
            "W_k": rand(d, d), "b_k": rand(d),
            "W_v": rand(d, d), "b_v": rand(d),
            "W_o": rand(d, d), "b_o": rand(d),
            "ln_2_g": np.ones(d),  "ln_2_b": np.zeros(d),
            "W_1": rand(d, Ff), "b_1": rand(Ff),
            "W_2": rand(Ff, d), "b_2": rand(d),
        })

    token_emb = rand(V, d)
    pos_emb   = rand(S, d)
    ln_f_g    = np.ones(d)
    ln_f_b    = np.zeros(d)

    token_ids = rng.integers(0, V, size=(B, S))

    logits, flops_breakdown = gpt2_forward(
        token_ids, blocks, token_emb, pos_emb, ln_f_g, ln_f_b, H,
    )

    print("=" * 60)
    print(f"Tiny GPT-2 (L={L}, d={d}, H={H}, Ff={Ff}, V={V}), B={B}, S={S}")
    print("=" * 60)
    print(f"Output logits shape: {logits.shape}")
    print()
    print("FLOP breakdown:")
    for k, v in flops_breakdown.items():
        print(f"  {k:<20} {v/1e6:>8.3f} MFLOPs")

    # From flops.py (block-only, ignoring bias/LN/softmax):
    #   L * (24*B*S*d² + 4*B*S²*d)
    formula_blocks = L * (24 * B * S * d * d + 4 * B * S * S * d)
    manual_blocks = flops_breakdown["all_blocks"]

    print()
    print(f"Manual (blocks only):   {manual_blocks/1e6:.3f} MFLOPs")
    print(f"Formula (blocks only):  {formula_blocks/1e6:.3f} MFLOPs")
    print(f"Ratio manual/formula:   {manual_blocks/formula_blocks:.3f}")
    print()
    print("Expected ratio 1.00–1.05. Small excess in manual comes from:")
    print("  - QKV and output biases   (~0.1% each)")
    print("  - MLP biases              (~0.1%)")
    print("  - LayerNorm               (~1% depending on constant)")
    print("  - Softmax + scale         (~0.5%)")


if __name__ == "__main__":
    main()
