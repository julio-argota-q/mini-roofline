import pytest

from miniroofline.cost_model.flops import flops_attention_layer
from miniroofline.cost_model.memory import (
    DTYPE_BYTES,
    bytes_activations_prefill,
    bytes_kv_cache,
    bytes_model_weights,
    kv_cache_crossover_tokens,
    traffic_attention_layer,
    traffic_mlp_layer,
)


@pytest.mark.parametrize(
    ("dtype", "expected_bytes"),
    [("float32", 4), ("float16", 2), ("bfloat16", 2), ("int8", 1)],
)
def test_dtype_byte_widths(dtype: str, expected_bytes: int) -> None:
    assert DTYPE_BYTES[dtype] == expected_bytes


def test_model_weight_count_matches_manual_tiny_model() -> None:
    L, d, d_ff, vocab = 2, 4, 10, 20
    per_layer = (
        3 * d * d + 3 * d
        + d * d + d
        + d * d_ff + d_ff
        + d_ff * d + d
        + 4 * d
    )
    total_params = L * per_layer + vocab * d + 1024 * d

    result = bytes_model_weights(L=L, d=d, d_ff=d_ff, vocab=vocab, dtype="float32")

    assert result["params_per_layer"] == per_layer
    assert result["total_params"] == total_params
    assert result["weight_bytes"] == total_params * 4
    assert result["dtype"] == "float32"


def test_model_weights_default_to_four_x_mlp_dimension() -> None:
    assert bytes_model_weights(L=1, d=8)["params_per_layer"] == bytes_model_weights(
        L=1, d=8, d_ff=32
    )["params_per_layer"]


def test_model_weight_bytes_scale_with_dtype_width() -> None:
    fp32 = bytes_model_weights(L=2, d=8, dtype="float32")
    fp16 = bytes_model_weights(L=2, d=8, dtype="float16")
    int8 = bytes_model_weights(L=2, d=8, dtype="int8")

    assert fp32["total_params"] == fp16["total_params"] == int8["total_params"]
    assert fp32["weight_bytes"] == 2 * fp16["weight_bytes"]
    assert fp32["weight_bytes"] == 4 * int8["weight_bytes"]


def test_unknown_dtype_raises_key_error() -> None:
    with pytest.raises(KeyError):
        bytes_model_weights(L=1, d=8, dtype="float64")


def test_prefill_activation_breakdown_matches_tensor_shapes() -> None:
    B, S, L, d, H, d_ff = 2, 5, 3, 12, 3, 40
    nb = 4
    result = bytes_activations_prefill(B=B, S=S, L=L, d=d, H=H, d_ff=d_ff)

    expected_residual = B * S * d * nb
    expected_attention = B * H * S * S * nb
    expected_mlp = B * S * d_ff * nb

    assert result["residual_bytes"] == expected_residual
    assert result["attention_matrix_bytes"] == expected_attention
    assert result["mlp_intermediate_bytes"] == expected_mlp
    assert result["peak_per_layer_bytes"] == expected_residual + expected_attention + expected_mlp


def test_prefill_activation_memory_scales_quadratically_for_attention() -> None:
    short = bytes_activations_prefill(B=1, S=8, L=2, d=16, H=4)
    long = bytes_activations_prefill(B=1, S=16, L=2, d=16, H=4)

    assert long["attention_matrix_bytes"] == 4 * short["attention_matrix_bytes"]
    assert long["residual_bytes"] == 2 * short["residual_bytes"]
    assert long["mlp_intermediate_bytes"] == 2 * short["mlp_intermediate_bytes"]


def test_kv_cache_breakdown_and_megabyte_conversion() -> None:
    B, S_ctx, L, d = 2, 7, 3, 12
    expected = 2 * B * L * S_ctx * d * 4
    expected_per_token = 2 * B * L * d * 4

    result = bytes_kv_cache(B=B, S_ctx=S_ctx, L=L, d=d)

    assert result["kv_cache_bytes"] == expected
    assert result["per_token_bytes"] == expected_per_token
    assert result["kv_cache_mb"] == pytest.approx(expected / 1024**2)


def test_kv_cache_grows_linearly_with_context() -> None:
    short = bytes_kv_cache(B=1, S_ctx=10, L=2, d=16)
    long = bytes_kv_cache(B=1, S_ctx=30, L=2, d=16)

    assert long["kv_cache_bytes"] == 3 * short["kv_cache_bytes"]
    assert long["per_token_bytes"] == short["per_token_bytes"]


def test_kv_cache_crossover_inverts_cache_formula() -> None:
    B, L, d, dtype = 2, 3, 12, "float16"
    target_tokens = 125
    weight_bytes = 2 * B * L * target_tokens * d * DTYPE_BYTES[dtype]

    assert kv_cache_crossover_tokens(L=L, d=d, weight_bytes=weight_bytes, B=B, dtype=dtype) == target_tokens


def test_attention_traffic_matches_manual_accounting() -> None:
    B, S, d, H = 2, 5, 12, 3
    nb = 4
    result = traffic_attention_layer(B=B, S=S, d=d, H=H)

    expected_weight = 4 * d**2 * nb
    expected_activation = (
        B * S * d * nb
        + 3 * B * S * d * nb
        + 2 * B * H * S**2 * nb
        + B * S * d * nb
    )
    expected_total = expected_weight + expected_activation

    assert result["weight_traffic_bytes"] == expected_weight
    assert result["activation_traffic_bytes"] == expected_activation
    assert result["total_traffic_bytes"] == expected_total
    assert result["flops"] == flops_attention_layer(B, S, d, H)
    assert result["arithmetic_intensity"] == pytest.approx(result["flops"] / expected_total)


def test_attention_traffic_scales_with_dtype_size() -> None:
    fp32 = traffic_attention_layer(B=1, S=8, d=16, H=4, dtype="float32")
    fp16 = traffic_attention_layer(B=1, S=8, d=16, H=4, dtype="float16")

    assert fp32["flops"] == fp16["flops"]
    assert fp32["total_traffic_bytes"] == 2 * fp16["total_traffic_bytes"]
    assert fp16["arithmetic_intensity"] == pytest.approx(2 * fp32["arithmetic_intensity"])


def test_mlp_traffic_matches_manual_accounting() -> None:
    B, S, d, d_ff = 2, 5, 12, 40
    nb = 4
    result = traffic_mlp_layer(B=B, S=S, d=d, d_ff=d_ff)

    expected_weight = 2 * d * d_ff * nb
    expected_activation = (B * S * d + B * S * d_ff + B * S * d) * nb
    expected_flops = 4 * B * S * d * d_ff

    assert result["weight_traffic_bytes"] == expected_weight
    assert result["activation_traffic_bytes"] == expected_activation
    assert result["total_traffic_bytes"] == expected_weight + expected_activation
    assert result["flops"] == expected_flops
    assert result["arithmetic_intensity"] == pytest.approx(expected_flops / result["total_traffic_bytes"])


def test_mlp_traffic_defaults_to_four_x_expansion() -> None:
    assert traffic_mlp_layer(B=1, S=8, d=16) == traffic_mlp_layer(B=1, S=8, d=16, d_ff=64)
