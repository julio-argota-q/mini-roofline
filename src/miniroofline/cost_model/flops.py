"""
flops.py
--------
First-principles FLOPs formulas for GPT-style transformer inference.

All formulas are DERIVED, not looked up. Every factor is explained.
Notation throughout:
    B  : batch size
    S  : sequence length (number of tokens)
    L  : number of transformer layers
    d  : model hidden dimension (d_model)
    H  : number of attention heads
    Dh : head dimension = d // H
    Ff : feed-forward hidden dim (4*d for standard GPT-2)

One multiply-add (MAD) = 2 FLOPs throughout.
All counts are for a single forward pass (inference only).
Training would multiply by ~3 (forward + backward + optimizer).
Bias adds contribute <0.1% of total FLOPs and are omitted for clarity
"""

# ---------------------------------------------------------------------------
# Deliberate omissions
# ---------------------------------------------------------------------------
# The formulas below count only matmul FLOPs. The following operations
# are omitted because their combined contribution is < 0.1% of total:
#
#   - Attention softmax:   3·B·H·S²           (~0.03% at S=128)
#   - Score scaling by √Dh: B·H·S²             (~0.01% at S=128)
#   - Causal mask:          0 FLOPs (bool ops only)
#   - QKV, output, MLP biases: 4·B·S·d + B·S·Ff (~0.05% at S=128)
#   - GELU:                  8·B·S·Ff         (~2% at S=128 — the largest)
#   - LayerNorm:             counted separately in flops_layer_norm
#   - Residual adds:         B·S·d per residual (~0.01%)
#
# The tiny NumPy transformer in notebooks/tiny_transformer.py counts
# these individually for verification. See verify_against_flops.py.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Single-layer operations
# ---------------------------------------------------------------------------

def flops_qkv_projection(B: int, S: int, d: int) -> int:
    """
    Q, K, V projections: three [B,S,d] x [d,d] matmuls.

    Each matmul: B*S output rows, each requiring d multiply-adds.
    One projection = 2*B*S*d^2 FLOPs.
    Three projections (Q, K, V) = 3 * 2*B*S*d^2.

    Shape trace:
        input : [B, S, d]
        weight: [d, d]   (one weight matrix per projection)
        output: [B, S, d]
        FLOPs per projection: B * S * d * d MADs * 2 = 2*B*S*d^2
    """
    return 3 * 2 * B * S * d * d


def flops_attention_scores(B: int, S: int, d: int, H: int) -> int:
    """
    Attention score matrix: Q @ K^T -> [B, H, S, S].

    For each head: [B, S, Dh] x [B, Dh, S] -> [B, S, S]
    Each output element requires Dh multiply-adds.
    FLOPs per head: B * S * S * Dh * 2
    Total across H heads: H * 2*B*S^2*Dh = 2*B*H*S^2*(d/H) = 2*B*S^2*d

    Note: scales as O(S^2 * d) — the quadratic-in-S term.
    At long sequences this dominates over MLP.
    """
    Dh = d // H
    return 2 * B * H * S * S * Dh   # = 2*B*S^2*d


def flops_attention_weighted_sum(B: int, S: int, d: int, H: int) -> int:
    """
    Weighted sum: softmax(QK^T) @ V -> [B, H, S, Dh].

    Same shape as attention scores computation:
    [B, H, S, S] x [B, H, S, Dh] -> [B, H, S, Dh]
    Each output element requires S multiply-adds.
    FLOPs: H * 2*B*S*S*Dh = 2*B*S^2*d

    Identical cost to attention scores — together they are 4*B*S^2*d.
    """
    Dh = d // H
    return 2 * B * H * S * S * Dh   # = 2*B*S^2*d


def flops_output_projection(B: int, S: int, d: int) -> int:
    """
    Output projection after attention: [B,S,d] x [d,d] -> [B,S,d].

    One matmul: B*S rows, each requiring d MADs.
    FLOPs: 2*B*S*d^2
    """
    return 2 * B * S * d * d


def flops_attention_layer(B: int, S: int, d: int, H: int) -> int:
    """
    Total FLOPs for one complete attention layer:
        QKV projections  : 3 * 2*B*S*d^2   = 6*B*S*d^2
        Attention scores :     2*B*S^2*d
        Weighted sum     :     2*B*S^2*d
        Output projection:     2*B*S*d^2
        ----------------------------------------
        Total            : 8*B*S*d^2 + 4*B*S^2*d

    The d^2 terms dominate at short S; the S^2 terms dominate at long S.
    Crossover (d^2 term = S^2 term):
        8*d^2 = 4*S^2 => S = sqrt(2)*d
    For GPT-2 small (d=768): crossover at S ≈ 3072 tokens.
    """
    return (
        flops_qkv_projection(B, S, d)
        + flops_attention_scores(B, S, d, H)
        + flops_attention_weighted_sum(B, S, d, H)
        + flops_output_projection(B, S, d)
    )


def flops_mlp_layer(B: int, S: int, d: int, d_ff: int = None) -> int:
    """
    MLP block: two linear layers with a nonlinearity between them.
    Standard GPT-2 uses d_ff = 4*d (no gating).

    FC1: [B,S,d] x [d, d_ff] -> [B,S,d_ff]   FLOPs: 2*B*S*d*d_ff
    FC2: [B,S,d_ff] x [d_ff,d] -> [B,S,d]    FLOPs: 2*B*S*d_ff*d
    Activation (GELU): 8 flops per element.  FLOPs: 8*B*S*d_ff

    Total: 2 * 2*B*S*d*d_ff + 8*B*S*d_ff = 4*B*S*d_ff*(d+2)
    For d_ff = 4d: 4*B*S*4d*(d+2) = 16*B*S*d*(d+2)

    This is always approx 2x the QKV projection cost.
    """
    if d_ff is None:
        d_ff = 4 * d
    matmul_flops = 2 * 2 * B * S * d * d_ff
    gelu_flops = 8 * B * S * d_ff
    return matmul_flops + gelu_flops


def flops_layer_norm(B: int, S: int, d: int) -> int:
    """
    Layer normalisation: mean, variance, normalise, scale, shift.
    FLOPs: approximately 7*B*S*d (subtract mean, variance, normalise,
    multiply by gamma, add beta).

    This is O(B*S*d) — much smaller than attention or MLP.
    In practice, LayerNorm is often *overhead-sensitive* rather than
    compute-bound: measured time exceeds roofline prediction due to
    memory access patterns and PyTorch dispatch overhead.
    """
    return 7 * B * S * d


# ---------------------------------------------------------------------------
# Whole-model prefill (processing the full prompt)
# ---------------------------------------------------------------------------

def flops_prefill(
    B: int,
    S: int,
    L: int,
    d: int,
    H: int,
    d_ff: int = None,
    include_layernorm: bool = True,
) -> dict:
    """
    Total FLOPs for processing a prompt of S tokens (prefill).

    Returns a breakdown dict so each component can be compared
    separately in the validation experiment.

    Dominant term analysis:
        Attention d^2 terms : L * 8*B*S*d^2
        Attention S^2 terms : L * 4*B*S^2*d
        MLP                 : L * 16*B*S*d^2

    Total (ignoring layernorm):
        L * (24*B*S*d^2 + 4*B*S^2*d)

    For GPT-2 small (L=12, d=768, H=12) at S=128, B=1:
        Attention d^2 : 12 * 8 * 1 * 128 * 768^2 ≈ 0.73 GFLOPs
        Attention S^2 : 12 * 4 * 1 * 128^2 * 768 ≈ 0.10 GFLOPs
        MLP           : 12 * 16 * 1 * 128 * 768^2 ≈ 1.45 GFLOPs
        Total         ≈ 2.28 GFLOPs
    """
    if d_ff is None:
        d_ff = 4 * d

    attn_d2 = L * (flops_qkv_projection(B, S, d) + flops_output_projection(B, S, d))
    attn_s2 = L * (flops_attention_scores(B, S, d, H) + flops_attention_weighted_sum(B, S, d, H))
    mlp     = L * flops_mlp_layer(B, S, d, d_ff)
    ln      = L * 2 * flops_layer_norm(B, S, d) if include_layernorm else 0
    total   = attn_d2 + attn_s2 + mlp + ln

    return {
        "attention_proj_flops": attn_d2,
        "attention_scores_flops": attn_s2,
        "attention_total_flops": attn_d2 + attn_s2,
        "mlp_flops": mlp,
        "layernorm_flops": ln,
        "total_flops": total,
        # fractions — useful for roofline component share analysis
        "attention_frac": (attn_d2 + attn_s2) / total,
        "mlp_frac": mlp / total,
    }


# ---------------------------------------------------------------------------
# Per-token decode (one new token generated with KV cache)
# ---------------------------------------------------------------------------

def flops_decode_step(
    B: int,
    S_ctx: int,
    L: int,
    d: int,
    H: int,
    d_ff: int = None,
) -> dict:
    """
    FLOPs for generating ONE new token, given S_ctx tokens already in KV cache.

    Key insight: the new token only needs to attend to all S_ctx cached tokens.
    QKV projections: only for the NEW token (S=1), not all S_ctx tokens.
    Attention: new token queries against S_ctx keys and values.

    This makes decode O(S_ctx) per step in attention, but O(d^2) for MLP
    — MLP cost is constant per token, attention grows with context.

    QKV for new token:  3 * 2*B*1*d^2 = 6*B*d^2
    Attention (1 vs S_ctx):
        scores:  2*B*H*1*S_ctx*Dh = 2*B*S_ctx*d
        values:  2*B*H*1*S_ctx*Dh = 2*B*S_ctx*d
    Output proj:        2*B*1*d^2
    MLP:                L * 16*B*1*d^2  (independent of S_ctx)
    """
    if d_ff is None:
        d_ff = 4 * d
    Dh = d // H

    qkv       = L * 3 * 2 * B * 1 * d * d
    attn_s    = L * 2 * B * H * 1 * S_ctx * Dh   # scores
    attn_v    = L * 2 * B * H * 1 * S_ctx * Dh   # values
    out_proj  = L * 2 * B * 1 * d * d
    mlp       = L * flops_mlp_layer(B, 1, d, d_ff)
    total     = qkv + attn_s + attn_v + out_proj + mlp

    return {
        "qkv_flops": qkv,
        "attention_score_flops": attn_s,
        "attention_value_flops": attn_v,
        "output_proj_flops": out_proj,
        "mlp_flops": mlp,
        "total_flops": total,
        "attention_frac": (attn_s + attn_v) / total,
        "mlp_frac": mlp / total,
    }


# ---------------------------------------------------------------------------
# Attention/MLP crossover
# ---------------------------------------------------------------------------

def attention_mlp_crossover_seq_len(d: int, d_ff: int = None) -> float:
    """
    Sequence length at which attention FLOPs equal MLP FLOPs.

    Uses only matmul-dominant terms (attention scores + weighted sum
    vs MLP FC1 + FC2). Ignores softmax, GELU, and biases; these shift
    the crossover by <1% and would obscure the scaling law S* ∝ d.

    Attention S^2 term per layer: 4*B*S^2*d
    MLP per layer:                16*B*S*d^2  (for d_ff=4d)

    Setting equal and solving for S:
        4*S^2*d = 16*S*d^2
        S = 4*d

    For GPT-2 small (d=768):  S = 3072
    For GPT-2 medium (d=1024): S = 4096

    This is the sequence length above which attention dominates total FLOPs.
    """
    if d_ff is None:
        d_ff = 4 * d
    # generalised: 4*S^2*d = 2 * 2*S*d*d_ff => S = d_ff / d * d = d_ff
    return float(d_ff)


# ---------------------------------------------------------------------------
# GPT-2 model configs (convenience)
# ---------------------------------------------------------------------------

GPT2_CONFIGS = {
    "distilgpt2": dict(L=6,  d=768,  H=12, d_ff=3072, vocab=50257),
    "gpt2":       dict(L=12, d=768,  H=12, d_ff=3072, vocab=50257),
    "gpt2-medium":dict(L=24, d=1024, H=16, d_ff=4096, vocab=50257),
    "gpt2-large": dict(L=36, d=1280, H=20, d_ff=5120, vocab=50257),
    "gpt2-xl":    dict(L=48, d=1600, H=25, d_ff=6400, vocab=50257),
}


def flops_prefill_for_model(model_name: str, B: int, S: int) -> dict:
    """
    Convenience wrapper: compute prefill FLOPs for a named GPT-2 model.

    Example:
        result = flops_prefill_for_model("gpt2", B=1, S=128)
        print(f"Total: {result['total_flops']/1e9:.2f} GFLOPs")
    """
    cfg = GPT2_CONFIGS[model_name]
    result = flops_prefill(B=B, S=S, **{k: cfg[k] for k in ("L", "d", "H", "d_ff")})

    lm_head_flops = 2 * B * S * cfg["d"] * cfg["vocab"]
    result["lm_head_flops"] = lm_head_flops
    result["total_flops"] += lm_head_flops
    result["attention_frac"] = result["attention_total_flops"] / result["total_flops"]
    result["mlp_frac"] = result["mlp_flops"] / result["total_flops"]

    return result


if __name__ == "__main__":

    print("=== GPT-2 small (B=1, S=128) ===")
    result = flops_prefill_for_model("gpt2", B=1, S=128)
    for k, v in result.items():
        if "flops" in k:
            print(f"  {k:<30} {v/1e9:>8.3f} GFLOPs")
        else:
            print(f"  {k:<30} {v:>8.3f}")

    print()
    crossover = attention_mlp_crossover_seq_len(d=768)
    print(f"Attention/MLP crossover (d=768): S = {crossover:.0f} tokens")

    print()
    print("=== Decode step (S_ctx=128) ===")
    decode = flops_decode_step(B=1, S_ctx=128, L=12, d=768, H=12)
    for k, v in decode.items():
        if "flops" in k:
            print(f"  {k:<30} {v/1e6:>8.3f} MFLOPs")
        else:
            print(f"  {k:<30} {v:>8.3f}")
