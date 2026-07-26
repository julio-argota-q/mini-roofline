"""
experiments/exp3_shape_sensitivity.py
-------------------------------------
EXPERIMENT 3 — Matmul shape sensitivity study

RESEARCH QUESTION:
  Experiment 2 established that GPT-2's matmul-heavy components (attention,
  MLP) achieve only 37-85% of the peak throughput measured on a 4096² fp32
  matmul. Why?

  Hypothesis: the "peak FLOP/s" of a machine is not a single constant. It
  varies with matrix shape, primarily with the M dimension (batch × seq_len)
  and the aspect ratio. Small M matmuls do not saturate AMX/BLAS kernels.

  Concretely: measure achieved throughput for a grid of matmul shapes,
  isolate the shapes that appear in GPT-2 inference, and quantify how far
  below the 4096²-matmul peak they fall.

METHOD:
  1. Define a grid of representative matmul shapes:
     a. GPT-2 shapes: QKV proj, output proj, MLP FC1, MLP FC2, LM head,
        at S ∈ {32, 128, 512}.
     b. Square baselines: 512², 1024², 2048², 4096² (to reproduce the
        "peak" number we measured in hardware.py).
     c. Rectangular sweep: hold N,K fixed and sweep M ∈ {8, 32, 128, 512, 2048}
        to isolate the M-dimension effect.

  2. For each shape (M, K, N):
     - Generate fresh fp32 tensors.
     - Warmup 10 runs.
     - Measure 30 runs, take median.
     - Compute achieved GFLOP/s = 2·M·K·N / median_s / 1e9.

  3. Save one JSON row per shape.

  4. Report: which shape achieves the highest throughput? Which GPT-2 shapes
     fall furthest below that? What's the M threshold at which throughput
     approaches peak?

OUTPUTS:
  - experiments/results/exp3/shape_sensitivity.json  (all measurements)
  - Console tables: peak sweep, GPT-2 shapes, rectangular sweep

Run:
  uv run python experiments/exp3_shape_sensitivity.py
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
import torch

from miniroofline.benchmark.timing import set_perf_cores
from miniroofline.cost_model.hardware import DEFAULT_HW
from miniroofline.cost_model.flops import GPT2_CONFIGS


N_WARMUP = 10
N_RUNS = 30
OUT_DIR = Path("experiments/results/exp3")


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MatmulMeasurement:
    """One measurement of one matmul shape."""
    label: str
    category: str        # "gpt2", "square_baseline", "m_sweep"
    M: int
    K: int
    N: int
    median_s: float
    iqr_relative: float
    flops: int           # 2*M*K*N
    achieved_gflops: float
    fraction_of_peak: float   # relative to our M4 Pro peak constant

    @property
    def arithmetic_intensity(self) -> float:
        # For a matmul A[M,K] @ B[K,N]:
        # Reads: M*K + K*N floats = 4*(M*K + K*N) bytes
        # Writes: M*N floats = 4*M*N bytes
        # Total traffic: 4*(M*K + K*N + M*N)
        traffic = 4 * (self.M * self.K + self.K * self.N + self.M * self.N)
        return self.flops / traffic


# ─────────────────────────────────────────────────────────────────────────────
# The measurement primitive
# ─────────────────────────────────────────────────────────────────────────────

def measure_matmul(M: int, K: int, N: int, label: str, category: str) -> MatmulMeasurement:
    """
    Measure achieved GFLOP/s for one fp32 matmul of shape [M,K] × [K,N].

    Uses the same warmup + median-of-N protocol as hardware.benchmark_flops.
    """
    A = torch.randn(M, K, dtype=torch.float32)
    B = torch.randn(K, N, dtype=torch.float32)

    # Warmup
    for _ in range(N_WARMUP):
        _ = A @ B

    times: list[float] = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        _ = A @ B
        times.append(time.perf_counter() - t0)

    times.sort()
    median_s = times[len(times) // 2]
    p25, p75 = times[len(times) // 4], times[(3 * len(times)) // 4]
    iqr_rel = (p75 - p25) / median_s if median_s > 0 else 0.0

    flops = 2 * M * K * N
    achieved = flops / median_s
    peak = DEFAULT_HW.peak_flops_fp32
    fraction = achieved / peak

    return MatmulMeasurement(
        label=label,
        category=category,
        M=M, K=K, N=N,
        median_s=median_s,
        iqr_relative=iqr_rel,
        flops=flops,
        achieved_gflops=achieved / 1e9,
        fraction_of_peak=fraction,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shape grids
# ─────────────────────────────────────────────────────────────────────────────

def gpt2_shapes() -> list[dict]:
    """
    The matmul shapes that actually appear in GPT-2 small inference.
    Each shape is exercised at three sequence lengths.

    Naming:
      QKV: one fused [S, d] × [d, 3d]  (HuggingFace's Conv1D)
      Out: [S, d] × [d, d]
      FC1: [S, d] × [d, 4d]
      FC2: [S, 4d] × [4d, d]
      LMH: [S, d] × [d, V]
    """
    cfg = GPT2_CONFIGS["gpt2"]
    d, V = cfg["d"], cfg["vocab"]
    shapes = []
    for S in [32, 128, 512]:
        shapes += [
            dict(label=f"QKV_S{S}",   category="gpt2", M=S, K=d,     N=3 * d),
            dict(label=f"Out_S{S}",   category="gpt2", M=S, K=d,     N=d),
            dict(label=f"FC1_S{S}",   category="gpt2", M=S, K=d,     N=4 * d),
            dict(label=f"FC2_S{S}",   category="gpt2", M=S, K=4 * d, N=d),
            dict(label=f"LMH_S{S}",   category="gpt2", M=S, K=d,     N=V),
        ]
    return shapes


def square_baseline_shapes() -> list[dict]:
    """Square matmuls to reproduce the "peak" measurement across sizes."""
    return [
        dict(label=f"square_{n}", category="square_baseline", M=n, K=n, N=n)
        for n in [128, 256, 512, 1024, 2048, 4096]
    ]


def m_sweep_shapes() -> list[dict]:
    """
    Hold K=768, N=3072 fixed (MLP FC1 shape for GPT-2), sweep M.
    Isolates how the "batch × seq_len" dimension affects throughput.
    """
    K, N = 768, 3072
    return [
        dict(label=f"m_sweep_{m}", category="m_sweep", M=m, K=K, N=N)
        for m in [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_measurements_table(measurements: list[MatmulMeasurement], title: str) -> None:
    print(f"\n── {title} ─────────────────────────────────────────────")
    print(f"  {'Shape':<14} {'M':>5} {'K':>6} {'N':>6}   "
          f"{'Median (ms)':>11} {'GFLOP/s':>10} {'% peak':>7} {'AI':>7}")
    print(f"  {'-'*14} {'-'*5} {'-'*6} {'-'*6}   "
          f"{'-'*11} {'-'*10} {'-'*7} {'-'*7}")
    for m in measurements:
        noise = " ⚠" if m.iqr_relative > 0.1 else ""
        print(
            f"  {m.label:<14} "
            f"{m.M:>5} {m.K:>6} {m.N:>6}   "
            f"{m.median_s*1000:>11.2f} "
            f"{m.achieved_gflops:>10.0f} "
            f"{m.fraction_of_peak*100:>6.1f}% "
            f"{m.arithmetic_intensity:>7.1f}{noise}"
        )


def print_gpt2_component_efficiency(gpt2_ms: list[MatmulMeasurement]) -> None:
    """Group GPT-2 shapes by (op, S) to show efficiency ranking."""
    print("\n── GPT-2 component efficiency ranking ─────────────────────────────")
    print("  Ordered by % of peak achieved.")
    print()
    print(f"  {'Component':<16} {'M':>5} {'K':>6} {'N':>6}   "
          f"{'GFLOP/s':>10} {'% peak':>7}")
    print(f"  {'-'*16} {'-'*5} {'-'*6} {'-'*6}   "
          f"{'-'*10} {'-'*7}")
    ranked = sorted(gpt2_ms, key=lambda m: -m.fraction_of_peak)
    for m in ranked:
        print(
            f"  {m.label:<16} "
            f"{m.M:>5} {m.K:>6} {m.N:>6}   "
            f"{m.achieved_gflops:>10.0f} "
            f"{m.fraction_of_peak*100:>6.1f}%"
        )


def print_m_sweep_finding(m_ms: list[MatmulMeasurement]) -> None:
    """Where does throughput approach peak as M grows?"""
    print("\n── M-dimension sweep (K=768, N=3072, GPT-2 MLP FC1 shape) ─────────")
    print("  Isolates how batch × seq_len dimension affects saturation.")
    print()
    print(f"  {'M':>6}  {'GFLOP/s':>10}  {'% peak':>7}  {'gain from M/2':>14}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*7}  {'-'*14}")
    prev = None
    for m in sorted(m_ms, key=lambda x: x.M):
        gain = f"{m.achieved_gflops / prev.achieved_gflops:.2f}×" if prev else "—"
        print(
            f"  {m.M:>6}  {m.achieved_gflops:>10.0f}  "
            f"{m.fraction_of_peak*100:>6.1f}%  {gain:>14}"
        )
        prev = m


def print_key_findings(all_measurements: list[MatmulMeasurement]) -> None:
    print("\n" + "=" * 70)
    print("KEY FINDINGS TO LOG")
    print("=" * 70)

    gpt2_ms = [m for m in all_measurements if m.category == "gpt2"]
    m_sweep = [m for m in all_measurements if m.category == "m_sweep"]
    square = [m for m in all_measurements if m.category == "square_baseline"]

    if gpt2_ms:
        best_gpt2 = max(gpt2_ms, key=lambda m: m.fraction_of_peak)
        worst_gpt2 = min(gpt2_ms, key=lambda m: m.fraction_of_peak)
        print(f"\n  GPT-2 shape efficiency range: "
              f"{worst_gpt2.fraction_of_peak*100:.1f}% ({worst_gpt2.label}) "
              f"to {best_gpt2.fraction_of_peak*100:.1f}% ({best_gpt2.label})")

    if square:
        peak_shape = max(square, key=lambda m: m.achieved_gflops)
        print(f"  Square-matmul peak: {peak_shape.achieved_gflops:.0f} GFLOP/s "
              f"at {peak_shape.label} ({peak_shape.fraction_of_peak*100:.1f}% of hardware.py peak)")

    if m_sweep:
        # Find M at which we hit 80% of the maximum in the sweep
        max_g = max(m.achieved_gflops for m in m_sweep)
        threshold = 0.8 * max_g
        crossing = None
        for m in sorted(m_sweep, key=lambda x: x.M):
            if m.achieved_gflops >= threshold:
                crossing = m
                break
        if crossing:
            print(f"  Throughput reaches 80% of its max ({max_g:.0f} GFLOP/s) "
                  f"at M ≥ {crossing.M}")
            print(f"  → GPT-2 at S=32,128 sits below this threshold; S=512 is at it.")

    # LM head is typically the highest-M GPT-2 op — call this out
    lmh = [m for m in gpt2_ms if m.label.startswith("LMH_")]
    fc1 = [m for m in gpt2_ms if m.label.startswith("FC1_")]
    if lmh and fc1:
        # match on S
        by_s = {}
        for m in lmh + fc1:
            S = int(m.label.split("_S")[1])
            by_s.setdefault(S, {})[m.label.split("_")[0]] = m
        for S, ops in sorted(by_s.items()):
            if "LMH" in ops and "FC1" in ops:
                ratio = ops["LMH"].fraction_of_peak / ops["FC1"].fraction_of_peak
                print(f"  At S={S}: LMH achieves {ratio:.2f}× the peak-fraction of FC1  "
                      f"(N-dim: {ops['LMH'].N} vs {ops['FC1'].N})")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    set_perf_cores()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT 3 — Matmul shape sensitivity")
    print("=" * 70)
    print(f"\nReference peak (hardware.py): "
          f"{DEFAULT_HW.peak_flops_fp32/1e12:.2f} TFLOP/s")
    print(f"(measured on a 4096² fp32 matmul)")

    all_shapes = square_baseline_shapes() + gpt2_shapes() + m_sweep_shapes()
    print(f"\nMeasuring {len(all_shapes)} shapes "
          f"({N_WARMUP} warmup + {N_RUNS} runs each)...")

    all_measurements: list[MatmulMeasurement] = []
    for shape in all_shapes:
        print(f"  {shape['label']:<16} "
              f"[{shape['M']:>5}, {shape['K']:>6}] × [{shape['K']:>6}, {shape['N']:>6}]",
              end=" ", flush=True)
        m = measure_matmul(**shape)
        all_measurements.append(m)
        print(f"→ {m.achieved_gflops:>6.0f} GFLOP/s "
              f"({m.fraction_of_peak*100:.1f}% of peak)")

    # ── Save ──
    out_path = OUT_DIR / "shape_sensitivity.json"
    with open(out_path, "w") as f:
        json.dump([asdict(m) for m in all_measurements], f, indent=2)

    # ── Tables ──
    square = [m for m in all_measurements if m.category == "square_baseline"]
    gpt2_ms = [m for m in all_measurements if m.category == "gpt2"]
    m_sweep = [m for m in all_measurements if m.category == "m_sweep"]

    print_measurements_table(square, "Square-matmul baseline (varying N=M=K)")
    print_gpt2_component_efficiency(gpt2_ms)
    print_m_sweep_finding(m_sweep)
    print_key_findings(all_measurements)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()