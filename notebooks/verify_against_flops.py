"""
verify_against_flops.py
-----------------------
Cross-check: does the tiny NumPy transformer's manual FLOP count agree
with the closed-form formulas in miniroofline.cost_model.flops?

We run BOTH at the same tiny config and compare component-by-component.
Any single mismatch means one of the two has a bug.

Run:  uv run python notebooks/verify_against_flops.py
"""

import sys
from pathlib import Path

# Make the notebook script able to import the sibling script
sys.path.insert(0, str(Path(__file__).parent))
from tiny_transformer import (
    gpt2_forward, multi_head_attention,
    mlp, layer_norm,
)

import numpy as np
from miniroofline.cost_model.flops import (
    flops_qkv_projection,
    flops_attention_scores,
    flops_attention_weighted_sum,
    flops_output_projection,
    flops_attention_layer,
    flops_mlp_layer,
    flops_layer_norm,
    flops_prefill,
)


# ═══════════════════════════════════════════════════════════════════════════
# Config — must be tiny so cross-check is fast
# ═══════════════════════════════════════════════════════════════════════════

B, S = 1, 32
L = 2
d = 64
H = 4
Ff = 4 * d       # 256
V = 100

rng = np.random.default_rng(42)
def rand(*shape):
    return rng.standard_normal(shape).astype(np.float32) * 0.02


# ═══════════════════════════════════════════════════════════════════════════
# Build the tiny model and run one forward pass
# ═══════════════════════════════════════════════════════════════════════════

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

_, tiny_breakdown = gpt2_forward(
    token_ids, blocks, token_emb, pos_emb, ln_f_g, ln_f_b, H,
)

# Also break out one block to compare per-component
x = token_emb[token_ids] + pos_emb[:S]
x_norm, tiny_ln1 = layer_norm(x, blocks[0]["ln_1_g"], blocks[0]["ln_1_b"])
_, tiny_attn = multi_head_attention(
    x_norm,
    blocks[0]["W_q"], blocks[0]["b_q"],
    blocks[0]["W_k"], blocks[0]["b_k"],
    blocks[0]["W_v"], blocks[0]["b_v"],
    blocks[0]["W_o"], blocks[0]["b_o"],
    H,
)
_, tiny_mlp = mlp(
    x_norm, blocks[0]["W_1"], blocks[0]["b_1"],
    blocks[0]["W_2"], blocks[0]["b_2"],
)


# ═══════════════════════════════════════════════════════════════════════════
# Formula predictions from flops.py
# ═══════════════════════════════════════════════════════════════════════════

f_qkv       = flops_qkv_projection(B, S, d)
f_scores    = flops_attention_scores(B, S, d, H)
f_weighted  = flops_attention_weighted_sum(B, S, d, H)
f_out_proj  = flops_output_projection(B, S, d)
f_attn_full = flops_attention_layer(B, S, d, H)
f_mlp       = flops_mlp_layer(B, S, d, Ff)
f_ln        = flops_layer_norm(B, S, d)
f_prefill   = flops_prefill(B=B, S=S, L=L, d=d, H=H, d_ff=Ff)


# ═══════════════════════════════════════════════════════════════════════════
# Component-by-component comparison
# ═══════════════════════════════════════════════════════════════════════════

def compare(label, tiny, formula, tolerance_pct=5.0):
    """
    Print a comparison. Returns True if within tolerance.
    Tolerance is % of the formula value.
    """
    diff = tiny - formula
    pct = 100 * diff / formula if formula else 0
    status = "OK " if abs(pct) <= tolerance_pct else "!! "
    # Small allowed excess in tiny (bias + softmax + LN etc.) — up to 5%
    if 0 <= pct <= tolerance_pct:
        status = "OK "
    print(
        f"  {status} {label:<28} "
        f"tiny={tiny:>10,.0f}  formula={formula:>10,.0f}  "
        f"diff={diff:>+10,.0f}  ({pct:+.2f}%)"
    )
    return abs(pct) <= tolerance_pct


print("=" * 78)
print(f"Cross-check: tiny transformer vs flops.py at tiny scale")
print(f"Config: L={L}, d={d}, H={H}, Ff={Ff}, V={V}, B={B}, S={S}")
print("=" * 78)

print("\n── Per-component comparison (single layer) ──")
compare("QKV projection",     tiny_attn["qkv_projection"],     f_qkv)
compare("attention scores",   tiny_attn["attention_scores"],   f_scores)
compare("weighted sum",       tiny_attn["weighted_sum"],       f_weighted)
compare("output projection",  tiny_attn["output_projection"],  f_out_proj)
compare("attention block",    tiny_attn["total"],              f_attn_full, tolerance_pct=10)
compare("MLP block",          tiny_mlp["total"],               f_mlp,        tolerance_pct=10)
compare("LayerNorm",          tiny_ln1,                         f_ln,         tolerance_pct=50)

print("\n── Whole-model comparison ──")
tiny_total = tiny_breakdown["total"]
formula_total = f_prefill["total_flops"]
compare("prefill total", tiny_total, formula_total, tolerance_pct=10)

print("\n── Fine-grained breakdown ──")
for name, val in tiny_breakdown.items():
    print(f"  tiny.{name:<20} = {val/1e6:>9.3f} MFLOPs")

print()
for name, val in f_prefill.items():
    if "flops" in name:
        print(f"  formula.{name:<25} = {val/1e6:>9.3f} MFLOPs")
