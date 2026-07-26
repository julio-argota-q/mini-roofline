import pytest

from miniroofline.cost_model.flops import GPT2_CONFIGS, flops_attention_layer, flops_mlp_layer
from miniroofline.cost_model.hardware import HardwareSpec
from miniroofline.cost_model.memory import traffic_attention_layer, traffic_mlp_layer
from miniroofline.cost_model.roofline import (
    OVERHEAD_THRESHOLD,
    RooflineResult,
    analyze,
    analyze_model,
    print_roofline_table,
)


@pytest.fixture
def hw() -> HardwareSpec:
    # Ridge point = 10 FLOP/byte.
    return HardwareSpec(name="Test accelerator", peak_flops_fp32=1_000.0, memory_bw=100.0)


def test_analyze_computes_intensity_latency_and_memory_classification(hw: HardwareSpec) -> None:
    result = analyze(operation="op", flops=2_000, traffic_bytes=400, hw=hw)

    assert result.operation == "op"
    assert result.arithmetic_intensity == 5.0
    assert result.predicted_latency_s == 4.0  # throughput = 100 B/s * 5 FLOP/B = 500 FLOP/s
    assert result.classification == "memory_bound"
    assert result.hw is hw


def test_analyze_computes_compute_bound_latency(hw: HardwareSpec) -> None:
    result = analyze(operation="op", flops=2_000, traffic_bytes=100, hw=hw)

    assert result.arithmetic_intensity == 20.0
    assert result.predicted_latency_s == 2.0
    assert result.classification == "compute_bound"


def test_zero_traffic_produces_zero_intensity_and_infinite_latency(hw: HardwareSpec) -> None:
    result = analyze(operation="empty", flops=100, traffic_bytes=0, hw=hw)

    assert result.arithmetic_intensity == 0.0
    assert result.predicted_latency_s == float("inf")
    assert result.classification == "memory_bound"


def test_measurement_above_threshold_is_overhead_sensitive(hw: HardwareSpec) -> None:
    predicted_only = analyze(operation="op", flops=2_000, traffic_bytes=400, hw=hw)
    measured = predicted_only.predicted_latency_s * OVERHEAD_THRESHOLD + 0.001

    predicted_only.add_measurement(measured)

    assert predicted_only.classification == "overhead_sensitive"
    assert predicted_only.prediction_error_pct == pytest.approx(
        (measured - predicted_only.predicted_latency_s) / predicted_only.predicted_latency_s * 100
    )


def test_measurement_exactly_at_threshold_is_not_overhead_sensitive(hw: HardwareSpec) -> None:
    result = analyze(operation="op", flops=2_000, traffic_bytes=400, hw=hw)
    result.add_measurement(OVERHEAD_THRESHOLD * result.predicted_latency_s)

    assert result.classification == "memory_bound"


def test_analyze_can_add_measurement_during_construction(hw: HardwareSpec) -> None:
    result = analyze(
        operation="op",
        flops=2_000,
        traffic_bytes=400,
        measured_latency_s=5.0,
        hw=hw,
    )

    assert result.measured_latency_s == 5.0
    assert result.prediction_error_pct == pytest.approx(25.0)


def test_custom_hardware_controls_classification() -> None:
    # Custom ridge is 1 FLOP/byte; the default M4 Pro ridge is roughly 14.5.
    custom = HardwareSpec(name="Low-ridge device", peak_flops_fp32=100.0, memory_bw=100.0)
    result = analyze(operation="op", flops=200, traffic_bytes=100, hw=custom)

    assert result.arithmetic_intensity == 2.0
    assert result.classification == "compute_bound"


def test_summary_uses_result_hardware_ridge_point() -> None:
    custom = HardwareSpec(name="Low-ridge device", peak_flops_fp32=100.0, memory_bw=100.0)
    result = analyze(operation="op", flops=200, traffic_bytes=100, hw=custom)

    assert "ridge: 1.0" in result.summary()


def test_result_summary_contains_measurement_and_error(hw: HardwareSpec) -> None:
    # Use an intensity whose classification agrees under both custom and default hardware.
    result = analyze(
        operation="attention",
        flops=2_000,
        traffic_bytes=400,
        measured_latency_s=5.0,
        hw=hw,
    )
    summary = result.summary()

    assert "Operation  : attention" in summary
    assert "Measured   : 5000.00 ms" in summary
    assert "Error      : +25.0%" in summary
    assert "Class      : memory_bound" in summary


def test_analyze_model_returns_all_expected_components(hw: HardwareSpec) -> None:
    results = analyze_model("distilgpt2", B=1, S=8, hw=hw)

    assert set(results) == {"attention", "mlp", "layernorm", "lm_head"}
    assert all(isinstance(result, RooflineResult) for result in results.values())
    assert all(result.hw is hw for result in results.values())


def test_analyze_model_attention_and_mlp_totals_match_source_models(hw: HardwareSpec) -> None:
    B, S = 1, 8
    cfg = GPT2_CONFIGS["distilgpt2"]
    results = analyze_model("distilgpt2", B=B, S=S, hw=hw)

    expected_attention_flops = flops_attention_layer(B, S, cfg["d"], cfg["H"]) * cfg["L"]
    expected_attention_traffic = (
        traffic_attention_layer(B, S, cfg["d"], cfg["H"])["total_traffic_bytes"] * cfg["L"]
    )
    expected_mlp_flops = flops_mlp_layer(B, S, cfg["d"], cfg["d_ff"]) * cfg["L"]
    expected_mlp_traffic = (
        traffic_mlp_layer(B, S, cfg["d"], cfg["d_ff"])["total_traffic_bytes"] * cfg["L"]
    )

    assert results["attention"].flops == expected_attention_flops
    assert results["attention"].traffic_bytes == expected_attention_traffic
    assert results["mlp"].flops == expected_mlp_flops
    assert results["mlp"].traffic_bytes == expected_mlp_traffic


def test_analyze_model_passes_component_measurements(hw: HardwareSpec) -> None:
    measurements = {
        "attention": 0.010,
        "mlp": 0.020,
        "layernorm": 0.003,
        "lm_head": 0.015,
    }
    results = analyze_model("distilgpt2", B=1, S=8, measured_latencies=measurements, hw=hw)

    for component, latency in measurements.items():
        assert results[component].measured_latency_s == latency
        assert results[component].prediction_error_pct is not None


def test_analyze_model_rejects_unknown_model(hw: HardwareSpec) -> None:
    with pytest.raises(KeyError):
        analyze_model("not-a-model", B=1, S=8, hw=hw)


def test_print_roofline_table_outputs_headers_and_rows(hw: HardwareSpec, capsys) -> None:
    results = {
        "attention": analyze("attention", flops=2_000, traffic_bytes=400, hw=hw),
        "mlp": analyze("mlp", flops=2_000, traffic_bytes=100, measured_latency_s=2.5, hw=hw),
    }

    print_roofline_table(results)
    output = capsys.readouterr().out

    assert "Component" in output
    assert "Predicted" in output
    assert "attention" in output
    assert "mlp" in output
    assert "memory_bound" in output
    assert "compute_bound" in output
