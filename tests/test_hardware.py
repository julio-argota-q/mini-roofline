import pytest

from miniroofline.cost_model.hardware import HardwareSpec


@pytest.fixture
def hw() -> HardwareSpec:
    return HardwareSpec(name="Test accelerator", peak_flops_fp32=1_000.0, memory_bw=100.0)


def test_ridge_point_is_peak_divided_by_bandwidth(hw: HardwareSpec) -> None:
    assert hw.ridge_point == 10.0


@pytest.mark.parametrize(
    ("intensity", "expected"),
    [
        (None, "overhead_sensitive"),
        (0.0, "memory_bound"),
        (9.999, "memory_bound"),
        (10.0, "compute_bound"),
        (100.0, "compute_bound"),
    ],
)
def test_classify_uses_ridge_point(hw: HardwareSpec, intensity: float | None, expected: str) -> None:
    assert hw.classify(intensity) == expected


def test_predicted_throughput_uses_bandwidth_ceiling_below_ridge(hw: HardwareSpec) -> None:
    assert hw.predicted_throughput(4.0) == 400.0


def test_predicted_throughput_caps_at_peak(hw: HardwareSpec) -> None:
    assert hw.predicted_throughput(25.0) == 1_000.0


def test_predicted_latency_uses_roofline_throughput(hw: HardwareSpec) -> None:
    assert hw.predicted_latency_s(total_flops=2_000.0, arithmetic_intensity=4.0) == 5.0
    assert hw.predicted_latency_s(total_flops=2_000.0, arithmetic_intensity=25.0) == 2.0


@pytest.mark.parametrize("intensity", [0.0, -1.0])
def test_predicted_latency_is_infinite_for_non_positive_throughput(
    hw: HardwareSpec, intensity: float
) -> None:
    assert hw.predicted_latency_s(total_flops=100.0, arithmetic_intensity=intensity) == float("inf")
