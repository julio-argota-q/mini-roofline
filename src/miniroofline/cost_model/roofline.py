"""
roofline.py
-----------
Roofline model: classifies each transformer operation as
compute-bound, memory-bound, or overhead-sensitive.

The roofline model (Williams et al., 2009) says the achievable
performance of any operation is bounded by:

    perf(I) = min(P_peak, BW * I)

where:
    I       = arithmetic intensity (FLOP / byte)
    P_peak  = peak FLOP/s of the hardware
    BW      = memory bandwidth (bytes/s)
    perf    = achievable FLOP/s

The ridge point I* = P_peak / BW separates the two regimes:
    I < I*  → memory-bound   (bandwidth limits performance)
    I >= I* → compute-bound  (peak FLOP/s limits performance)

A third class — overhead-sensitive — is needed when measured
latency far exceeds the roofline prediction. This happens for
small operations where PyTorch dispatch, kernel launch, or
cache effects dominate over actual compute or memory traffic.
"""

from dataclasses import dataclass
from typing import Optional
from miniroofline.cost_model.hardware import HardwareSpec, DEFAULT_HW


# ---------------------------------------------------------------------------
# Threshold for overhead-sensitive classification
# ---------------------------------------------------------------------------
# If measured_latency > OVERHEAD_THRESHOLD * predicted_latency,
# we classify the operation as overhead-sensitive.
OVERHEAD_THRESHOLD = 3.0


@dataclass
class RooflineResult:
    """Stores the roofline analysis for one operation."""
    operation: str
    flops: int
    traffic_bytes: int
    arithmetic_intensity: float      # FLOP / byte
    predicted_latency_s: float       # from roofline formula
    measured_latency_s: Optional[float] = None
    classification: str = ""         # set after measurement
    prediction_error_pct: Optional[float] = None
    hw: HardwareSpec = None

    def __post_init__(self):
        if not self.classification:
            self.classification = self._classify()
        if self.hw is None:
            self.hw = DEFAULT_HW

    def _classify(self) -> str:
        if self.measured_latency_s is None:
            # Before measurement: classify from intensity alone
            if self.arithmetic_intensity >= self.hw.ridge_point:
                return "compute_bound"
            return "memory_bound"

        # After measurement: check if overhead dominates
        if self.measured_latency_s > OVERHEAD_THRESHOLD * self.predicted_latency_s:
            return "overhead_sensitive"
        if self.arithmetic_intensity >= self.hw.ridge_point:
            return "compute_bound"
        return "memory_bound"

    def add_measurement(self, measured_latency_s: float) -> None:
        self.measured_latency_s = measured_latency_s
        self.classification = self._classify()
        if self.predicted_latency_s > 0:
            self.prediction_error_pct = (
                (measured_latency_s - self.predicted_latency_s)
                / self.predicted_latency_s * 100
            )

    def summary(self) -> str:
        lines = [
            f"Operation  : {self.operation}",
            f"FLOPs      : {self.flops/1e9:.3f} GFLOPs",
            f"Traffic    : {self.traffic_bytes/1e6:.2f} MB",
            f"Intensity  : {self.arithmetic_intensity:.2f} FLOP/byte  "
            f"(ridge: {self.hw.ridge_point:.1f})",
            f"Predicted  : {self.predicted_latency_s*1000:.2f} ms",
        ]
        if self.measured_latency_s is not None:
            lines.append(f"Measured   : {self.measured_latency_s*1000:.2f} ms")
            lines.append(f"Error      : {self.prediction_error_pct:+.1f}%")
        lines.append(f"Class      : {self.classification}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyze(
    operation: str,
    flops: int,
    traffic_bytes: int,
    measured_latency_s: float = None,
    hw: HardwareSpec = None,
) -> RooflineResult:
    """
    Run roofline analysis for one operation.

    Args:
        operation       : human-readable name (e.g. "attention layer 0")
        flops           : total FLOPs (from flops.py)
        traffic_bytes   : total bytes read+written (from memory.py)
        measured_latency_s : wall-clock time in seconds (optional)
        hw              : hardware spec (defaults to M4 Pro)

    Returns:
        RooflineResult with classification and optional error analysis.
    """
    if hw is None:
        hw = DEFAULT_HW

    intensity = flops / traffic_bytes if traffic_bytes > 0 else 0.0
    predicted  = hw.predicted_latency_s(flops, intensity)

    result = RooflineResult(
        operation=operation,
        flops=flops,
        traffic_bytes=traffic_bytes,
        arithmetic_intensity=intensity,
        predicted_latency_s=predicted,
        hw=hw,
    )
    if measured_latency_s is not None:
        result.add_measurement(measured_latency_s)

    return result


# ---------------------------------------------------------------------------
# Batch analysis for a full model sweep
# ---------------------------------------------------------------------------

def analyze_model(
    model_name: str,
    B: int,
    S: int,
    measured_latencies: dict = None,
    hw: HardwareSpec = None,
) -> dict[str, RooflineResult]:
    """
    Run roofline analysis for all components of a GPT-2 model.
    Explicitly omits:
      - Embeddings: <0.001% of FLOPs.
      - Residual adds: <0.01% of FLOPs.
      
    Args:
        model_name       : e.g. "gpt2", "distilgpt2"
        B, S             : batch size and sequence length
        measured_latencies: dict mapping component name -> latency_s
                           (from profiler/hooks.py output)
        hw               : hardware spec

    Returns:
        Dict of component name -> RooflineResult
    """
    from miniroofline.cost_model.flops import GPT2_CONFIGS
    from miniroofline.cost_model.memory import (
        traffic_attention_layer,
        traffic_mlp_layer,
    )
    from miniroofline.cost_model.flops import (
        flops_attention_layer,
        flops_mlp_layer,
    )

    if hw is None:
        hw = DEFAULT_HW

    cfg = GPT2_CONFIGS[model_name]
    L, d, H, d_ff = cfg["L"], cfg["d"], cfg["H"], cfg["d_ff"]
    nb = 4  # float32

    results = {}

    # Attention
    attn_flops   = flops_attention_layer(B, S, d, H) * L
    attn_traffic = traffic_attention_layer(B, S, d, H)
    results["attention"] = analyze(
        "attention (all layers)",
        attn_flops,
        attn_traffic["total_traffic_bytes"] * L,
        measured_latency_s=(measured_latencies or {}).get("attention"),
        hw=hw,
    )

    # MLP
    mlp_flops   = flops_mlp_layer(B, S, d, d_ff) * L
    mlp_traffic = traffic_mlp_layer(B, S, d, d_ff)
    results["mlp"] = analyze(
        "mlp (all layers)",
        mlp_flops,
        mlp_traffic["total_traffic_bytes"] * L,
        measured_latency_s=(measured_latencies or {}).get("mlp"),
        hw=hw,
    )

    # LayerNorm (rough: 7*B*S*d FLOPs, traffic ≈ 3*B*S*d bytes read+write)
    ln_flops   = 7 * B * S * d * 2 * L    # 2 LN per layer
    ln_traffic = 3 * B * S * d * nb * 2 * L
    results["layernorm"] = analyze(
        "layernorm (all layers)",
        ln_flops,
        ln_traffic,
        measured_latency_s=(measured_latencies or {}).get("layernorm"),
        hw=hw,
    )

    # LM head (single [B, S, d] × [d, V] matmul, no bias in GPT-2)
    V = cfg["vocab"]
    lmh_flops = 2 * B * S * d * V
    lmh_traffic = (B * S * d + d * V + B * S * V) * nb
    results["lm_head"] = analyze(
        "lm_head",
        lmh_flops,
        lmh_traffic,
        measured_latency_s=(measured_latencies or {}).get("lm_head"),
        hw=hw,
    )

    return results


def print_roofline_table(results: dict[str, RooflineResult]) -> None:
    """Print a formatted comparison table."""
    header = f"{'Component':<22} {'Intensity':>10} {'Predicted':>12} {'Measured':>12} {'Error':>8} {'Class':<20}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        meas = f"{r.measured_latency_s*1000:.2f} ms" if r.measured_latency_s else "—"
        err  = f"{r.prediction_error_pct:+.0f}%" if r.prediction_error_pct is not None else "—"
        print(
            f"{name:<22} "
            f"{r.arithmetic_intensity:>9.2f}x "
            f"{r.predicted_latency_s*1000:>10.2f} ms "
            f"{meas:>12} "
            f"{err:>8} "
            f"{r.classification:<20}"
        )


if __name__ == "__main__":
    print("=== Roofline analysis — GPT-2 small, B=1, S=128 ===\n")
    results = analyze_model("gpt2", B=1, S=128)
    print_roofline_table(results)

    print(f"\nM4 Pro ridge point: {DEFAULT_HW.ridge_point:.1f} FLOP/byte")
    print("Operations with intensity below this are memory-bound.")

    print("\n=== Single operation example ===")
    r = analyze(
        operation="attention layer 0 (S=512)",
        flops=2 * 1 * 12 * 512 * 512 * 64 * 2,   # scores + values
        traffic_bytes=1 * 12 * 512 * 512 * 4,      # attn matrix
    )
    print(r.summary())
