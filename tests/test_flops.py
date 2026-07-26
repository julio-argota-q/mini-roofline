import pytest
from miniroofline.cost_model.flops import (
    flops_qkv_projection,
    flops_attention_scores,
    flops_attention_weighted_sum,
    flops_output_projection,
    flops_attention_layer,
    flops_mlp_layer,
    flops_layer_norm,
    flops_prefill,
    flops_decode_step,
    attention_mlp_crossover_seq_len,
    flops_prefill_for_model
)

def test_qkv_projection_formula():
    """QKV = 3 · 2 · B · S · d² per layer."""
    B, S, d = 1, 32, 64
    expected = 3 * 2 * B * S * d * d   # 786_432
    assert flops_qkv_projection(B, S, d) == expected


def test_attention_scores_scales_as_s_squared():
    """4× S should give 16× FLOPs (S² scaling)."""
    small = flops_attention_scores(B=1, S=32, d=64, H=4)
    large = flops_attention_scores(B=1, S=128, d=64, H=4)
    assert large == 16 * small


def test_weighted_sum_equals_scores():
    """Attention scores and weighted sum have identical FLOP count."""
    args = dict(B=1, S=128, d=768, H=12)
    assert flops_attention_scores(**args) == flops_attention_weighted_sum(**args)


def test_output_projection_scales_with_d_squared():
    """4× d should give 16× FLOPs (d² scaling)."""
    small = flops_output_projection(B=1, S=128, d=128)
    large = flops_output_projection(B=1, S=128, d=512)
    assert large == 16 * small


def test_mlp_matches_16bsd_squared():
    """MLP with d_ff = 4d costs 16 · B · S · d · (d+ 2. bcause we are counting GELU flops"""
    B, S, d = 1, 128, 768
    assert flops_mlp_layer(B, S, d) == 16 * B * S * d * (d + 2)


def test_gpt2_small_prefill_agrees_with_fvcore():
    """Total FLOPs at S=128 should be within 5% of fvcore's 31.65 G."""
    result = flops_prefill_for_model("gpt2", B=1, S=128)
    total_g = result["total_flops"] / 1e9
    assert 30.0 < total_g < 34.0


def test_lm_head_included_and_positive():
    result = flops_prefill_for_model("gpt2", B=1, S=128)
    assert result["lm_head_flops"] > 0
    assert result["lm_head_flops"] < result["total_flops"]


def test_attention_mlp_crossover_is_4d_for_standard_config():
    """For d_ff = 4d, crossover S = 4d."""
    d = 768
    assert attention_mlp_crossover_seq_len(d) == 4 * d