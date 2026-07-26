"""
memory.py
---------
First-principles memory traffic model for GPT-style transformer inference.

"Memory traffic" means bytes moved between DRAM and CPU (or LLC).
This is what determines whether an operation is memory-bandwidth-bound.

Key distinction:
  - Memory FOOTPRINT : peak bytes resident at once (sizing question)
  - Memory TRAFFIC   : total bytes read + written (bandwidth question)

Roofline uses TRAFFIC, not footprint.
Both are useful: footprint for OOM analysis, traffic for latency prediction.
"""

DTYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
}


# ---------------------------------------------------------------------------
# Model weight memory
# ---------------------------------------------------------------------------

def bytes_model_weights(
    L: int, d: int, d_ff: int = None, vocab: int = 50257, dtype: str = "float32"
) -> dict:
    """
    Memory footprint of all model parameters.

    Accounting per layer:
        Attention:
          QKV weight  : 3 * d * d
          QKV bias    : 3 * d
          Out weight  : d * d
          Out bias    : d
        MLP:
          FC1 weight  : d * d_ff
          FC1 bias    : d_ff
          FC2 weight  : d_ff * d
          FC2 bias    : d
        LayerNorm (x2 per layer):
          gamma + beta: 2 * 2 * d

    Plus:
        Token embedding : vocab * d
        Position embed  : max_pos * d  (1024 for GPT-2)
        LM head         : vocab * d    (tied with token embed in GPT-2)

    For GPT-2 small (L=12, d=768, d_ff=3072, vocab=50257):
        Per-layer attn params  : 4*768^2 + 4*768    ≈ 2.36M
        Per-layer MLP params   : 2*768*3072 + 3072+768 ≈ 4.72M
        Per-layer LN params    : 4*768 = 3072 ≈ negligible
        Total per layer        ≈ 7.08M
        12 layers              ≈ 84.9M
        Embeddings             : 50257*768*2 ≈ 77.2M (tied, count once)
        Total                  ≈ 117M params  → 468 MB in fp32
    """
    if d_ff is None:
        d_ff = 4 * d
    nb = DTYPE_BYTES[dtype]
    max_pos = 1024  # GPT-2 uses learned positional embeddings up to 1024

    per_layer = (
        3 * d * d + 3 * d     # QKV weight + bias
        + d * d + d           # output projection weight + bias
        + d * d_ff + d_ff     # FC1 weight + bias
        + d_ff * d + d        # FC2 weight + bias
        + 2 * 2 * d           # two LayerNorm (gamma + beta each)
    )
    total_params = L * per_layer + vocab * d + max_pos * d
    # LM head is weight-tied with token embedding in GPT-2 — not added again

    return {
        "params_per_layer": per_layer,
        "total_params": total_params,
        "weight_bytes": total_params * nb,
        "dtype": dtype,
    }


# ---------------------------------------------------------------------------
# Activation memory (prefill)
# ---------------------------------------------------------------------------

def bytes_activations_prefill(
    B: int, S: int, L: int, d: int, H: int, d_ff: int = None, dtype: str = "float32"
) -> dict:
    """
    Peak activation memory during a prefill forward pass.

    We track the tensors alive at each layer. The dominant terms are:
        - Residual stream  : [B, S, d]
        - Attention matrix : [B, H, S, S]   ← grows as S^2
        - MLP intermediate : [B, S, d_ff]

    Peak is typically at the attention matrix (when S is large).

    Derivation:
        residual    = B * S * d * nb           (one per layer, reused)
        attn_matrix = B * H * S * S * nb       (materialised in standard attn)
        mlp_inter   = B * S * d_ff * nb        (intermediate activation)

    Total alive at one layer peak ≈ residual + attn_matrix + mlp_inter
    Plus input activations saved for backward (inference: no backward needed).
    """
    if d_ff is None:
        d_ff = 4 * d
    nb = DTYPE_BYTES[dtype]

    residual    = B * S * d * nb
    attn_matrix = B * H * S * S * nb    # the n^2 term — grows fast
    mlp_inter   = B * S * d_ff * nb

    # peak: all three alive simultaneously at the attention layer
    peak_per_layer = residual + attn_matrix + mlp_inter

    return {
        "residual_bytes": residual,
        "attention_matrix_bytes": attn_matrix,   # this is what Flash Attention avoids
        "mlp_intermediate_bytes": mlp_inter,
        "peak_per_layer_bytes": peak_per_layer,
        "note": (
            "attention_matrix grows as S^2 — at S=1024 it is "
            f"{B*H*1024*1024*nb/1024/1024:.1f} MB per layer. "
        ),
    }


# ---------------------------------------------------------------------------
# KV cache memory (decode)
# ---------------------------------------------------------------------------

def bytes_kv_cache(
    B: int, S_ctx: int, L: int, d: int, dtype: str = "float32"
) -> dict:
    """
    KV cache memory for S_ctx tokens already generated.

    For each layer, we store K and V matrices for all past tokens:
        K cache: [B, L, S_ctx, d]   (sometimes stored as [B, L, H, S_ctx, Dh])
        V cache: [B, L, S_ctx, d]
        Total  : 2 * B * L * S_ctx * d * nb

    This grows LINEARLY with S_ctx — contrast with attention matrix (S^2).
    As generation continues, KV cache eventually exceeds model weight memory.

    KV cache crossover: when does KV cache exceed model weights?
        2 * B * L * S_ctx * d * nb = weight_bytes
    Solving for S_ctx:
        S_ctx* = weight_bytes / (2 * B * L * d * nb)

    For GPT-2 small (fp32, B=1): weight_bytes ≈ 468 MB
        S_ctx* = 468e6 / (2 * 1 * 12 * 768 * 4) ≈ 6,334 tokens
    (GPT-2 max context is 1024, so KV cache never exceeds weights in practice.
     But for larger models with longer contexts this crossover matters a lot.)
    """
    nb = DTYPE_BYTES[dtype]
    kv_bytes = 2 * B * L * S_ctx * d * nb

    return {
        "kv_cache_bytes": kv_bytes,
        "kv_cache_mb": kv_bytes / 1024 / 1024,
        "per_token_bytes": 2 * B * L * d * nb,   # marginal cost per new token
        "note": f"Grows linearly with S_ctx. Current: {kv_bytes/1024/1024:.2f} MB",
    }


def kv_cache_crossover_tokens(
    L: int, d: int, weight_bytes: int, B: int = 1, dtype: str = "float32"
) -> float:
    """
    Token count at which KV cache memory equals model weight memory.
    Above this point, KV cache is the dominant memory consumer.
    """
    nb = DTYPE_BYTES[dtype]
    return weight_bytes / (2 * B * L * d * nb)


# ---------------------------------------------------------------------------
# Memory traffic (for roofline arithmetic intensity)
# ---------------------------------------------------------------------------

def traffic_attention_layer(
    B: int, S: int, d: int, H: int, dtype: str = "float32"
) -> dict:
    """
    Memory traffic for one attention layer (bytes read + written).

    Traffic = weights read + activations read + activations written.
    This is what determines arithmetic intensity, not footprint.

    Weights (read once per forward pass per layer):
        QKV weight  : 3 * d * d * nb
        Out weight  : d * d * nb

    Activations:
        Input  read : B * S * d * nb
        QKV outputs : 3 * B * S * d * nb  (written)
        Attn matrix : B * H * S * S * nb  (written + read for softmax + AV)
        Output      : B * S * d * nb      (written)

    This is a simplified model — actual cache behaviour is more complex.
    """
    nb = DTYPE_BYTES[dtype]
    Dh = d // H

    weight_traffic = (3 * d * d + d * d) * nb
    input_traffic  = B * S * d * nb
    qkv_traffic    = 3 * B * S * d * nb
    attn_traffic   = 2 * B * H * S * S * nb   # write + read of attn matrix
    out_traffic    = B * S * d * nb

    total = weight_traffic + input_traffic + qkv_traffic + attn_traffic + out_traffic

    flops = (
        3 * 2 * B * S * d * d            # QKV
        + 2 * 2 * B * H * S * S * Dh     # scores + values
        + 2 * B * S * d * d              # out proj
    )

    return {
        "weight_traffic_bytes": weight_traffic,
        "activation_traffic_bytes": total - weight_traffic,
        "total_traffic_bytes": total,
        "flops": flops,
        "arithmetic_intensity": flops / total,   # FLOP/byte
    }


def traffic_mlp_layer(
    B: int, S: int, d: int, d_ff: int = None, dtype: str = "float32"
) -> dict:
    """
    Memory traffic for one MLP layer.

    Weights:
        FC1: d * d_ff * nb
        FC2: d_ff * d * nb

    Activations:
        Input  : B * S * d * nb
        FC1 out: B * S * d_ff * nb  (intermediate, written then read)
        Output : B * S * d * nb

    Arithmetic intensity for MLP (d_ff=4d):
        FLOPs  = 16*B*S*d^2
        Traffic ≈ 2*d*d_ff*nb + 3*B*S*d*nb + B*S*d_ff*nb
               ≈ 8*d^2*nb + 3*B*S*d*nb + 4*B*S*d*nb
               = 8*d^2*nb + 7*B*S*d*nb

    At large batch*seq this is memory-bound; at small batch it can be
    weight-dominated (still memory-bound, just a different bottleneck).
    """
    if d_ff is None:
        d_ff = 4 * d
    nb = DTYPE_BYTES[dtype]

    weight_traffic = (d * d_ff + d_ff * d) * nb
    act_in         = B * S * d * nb
    act_inter      = B * S * d_ff * nb
    act_out        = B * S * d * nb
    total          = weight_traffic + act_in + act_inter + act_out

    flops = 2 * 2 * B * S * d * d_ff

    return {
        "weight_traffic_bytes": weight_traffic,
        "activation_traffic_bytes": act_in + act_inter + act_out,
        "total_traffic_bytes": total,
        "flops": flops,
        "arithmetic_intensity": flops / total,
    }

if __name__ == "__main__":
    from miniroofline.cost_model.flops import GPT2_CONFIGS

    print("=== GPT-2 small memory analysis (B=1, S=128, fp32) ===\n")
    cfg = GPT2_CONFIGS["gpt2"]

    weights = bytes_model_weights(**{k: cfg[k] for k in ("L", "d", "d_ff", "vocab")})
    print(f"Model weights    : {weights['weight_bytes']/1e6:.1f} MB")
    print(f"Total params     : {weights['total_params']/1e6:.1f} M")

    acts = bytes_activations_prefill(B=1, S=128, **{k: cfg[k] for k in ("L","d","H","d_ff")})
    print(f"\nPeak activations (S=128):")
    print(f"  Attention matrix : {acts['attention_matrix_bytes']/1e6:.2f} MB")
    print(f"  MLP intermediate : {acts['mlp_intermediate_bytes']/1e6:.2f} MB")
    print(f"  Residual stream  : {acts['residual_bytes']/1e6:.2f} MB")

    kv = bytes_kv_cache(B=1, S_ctx=128, L=cfg["L"], d=cfg["d"])
    print(f"\nKV cache (S_ctx=128): {kv['kv_cache_mb']:.2f} MB")

    crossover = kv_cache_crossover_tokens(
        L=cfg["L"], d=cfg["d"],
        weight_bytes=weights["weight_bytes"]
    )
    print(f"KV=weights crossover: S_ctx = {crossover:.0f} tokens")
