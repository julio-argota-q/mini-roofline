"""
hardware.py
-----------
Hardware specifications for the roofline model.

Each spec contains:
  - peak_flops_fp32  : theoretical peak FLOP/s (float32)
  - memory_bw        : memory bandwidth in bytes/sec
  - ridge_point      : peak_flops / memory_bw  (FLOP/byte)
                       operations with arithmetic intensity below this
                       are memory-bound; above it are compute-bound.
"""

from dataclasses import dataclass
import torch
import time

@dataclass(frozen=True)
class HardwareSpec:
    name: str
    peak_flops_fp32: float   # FLOP/s
    memory_bw: float         # bytes/s
    unified_memory: bool = False

    @property
    def ridge_point(self) -> float:
        """
        Ridge point in FLOP/byte.
        Ops with arithmetic intensity < ridge_point are memory-bound.
        Ops with arithmetic intensity > ridge_point are compute-bound.
        """
        return self.peak_flops_fp32 / self.memory_bw

    def classify(self, arithmetic_intensity: float) -> str:
        """
        Classify an operation given its arithmetic intensity (FLOP/byte).
        Returns one of: 'compute_bound', 'memory_bound', 'overhead_sensitive'.
        Pass arithmetic_intensity=None to get 'overhead_sensitive' directly.
        """
        if arithmetic_intensity is None:
            return "overhead_sensitive"
        if arithmetic_intensity >= self.ridge_point:
            return "compute_bound"
        return "memory_bound"

    def predicted_throughput(self, arithmetic_intensity: float) -> float:
        """
        Roofline performance ceiling in FLOP/s.
        perf = min(peak_flops, memory_bw * arithmetic_intensity)
        """
        return min(self.peak_flops_fp32, self.memory_bw * arithmetic_intensity)

    def predicted_latency_s(self, total_flops: float, arithmetic_intensity: float) -> float:
        """
        Predicted latency in seconds for an operation with known
        total_flops and arithmetic_intensity.
        latency = total_flops / predicted_throughput
        """
        throughput = self.predicted_throughput(arithmetic_intensity)
        if throughput <= 0:
            return float("inf")
        return total_flops / throughput

    def summary(self) -> str:
        lines = [
            f"Hardware : {self.name}",
            f"Peak     : {self.peak_flops_fp32/1e12:.2f} TFLOP/s (fp32)",
            f"BW       : {self.memory_bw/1e9:.0f} GB/s",
            f"Ridge pt : {self.ridge_point:.1f} FLOP/byte",
            f"Unified  : {self.unified_memory}",
        ]

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# M4 Pro — measured spect on the machine
# ---------------------------------------------------------------------------

M4_PRO = HardwareSpec(
    name="Apple M4 Pro (14-core, 24 GB) — measured",
    peak_flops_fp32=3.33e12,
    memory_bw=240e9,
    unified_memory=True
)

# Default used throughout the project
DEFAULT_HW = M4_PRO

# ---------------------------------------------------------------------------
# Runtime measurement helper
# ---------------------------------------------------------------------------

# Mesure the floap peaks
def benchmark_flops(n: int = 4096, reps: int = 30) -> float:
    """
    Measure achieved peak FLOP/s on the current machine using a large matmul.
    Returns FLOP/s as a float.

    Call this once and update M4_PRO.peak_flops_fp32 with the result.

    Usage:
        from miniroofline.cost_model.hardware import benchmark_flops
        measured = benchmark_flops()
        print(f"Achieved: {measured/1e12:.2f} TFLOP/s")
    """

    torch.set_num_threads(10)  # P-cores only on M4 Pro
    A = torch.randn(n, n)
    B = torch.randn(n, n)

    # warmup
    for _ in range(10):
        _ = A @ B

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _ = A @ B
        times.append(time.perf_counter() - t0)

    times.sort()
    median_s = times[len(times) // 2]
    flops = 2 * n ** 3          # one matmul = 2n³ FLOPs
    achieved = flops / median_s

    spec = M4_PRO.peak_flops_fp32
    print(f"Matrix   : {n}×{n}")
    print(f"Median   : {median_s * 1000:.2f} ms")
    print(f"Achieved : {achieved / 1e12:.2f} TFLOP/s")
    print(f"Spec     : {spec / 1e12:.2f} TFLOP/s")
    print(f"Efficiency: {achieved / spec * 100:.1f}%")
    print(f"Ridge point (achieved): {achieved / M4_PRO.memory_bw:.1f} FLOP/byte")

    return achieved

# Measure the memory bandwidth
def benchmark_memory():

    torch.set_num_threads(10)

    n_bytes = 1024 * 1024 * 1024  # 1 GB
    A = torch.randn(n_bytes // 4)  # 1 GB in float32

    # Warmup
    for _ in range(5):
        _ = A.clone()

    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        _ = A.clone()   # reads 1 GB + writes 1 GB = 2 GB traffic
        times.append(time.perf_counter() - t0)

    times.sort()
    median_s = times[len(times) // 2]
    gbs = (2 * n_bytes) / median_s / 1e9
    print(f"Median: {median_s*1000:.2f} ms")
    print(f"Achieved: {gbs:.0f} GB/s")
    print(f"Apple spec: 273 GB/s")
    print(f"Efficiency: {gbs/273*100:.1f}%")
    print(f"Ridge point (with measured BW): {3.33e12 / (gbs*1e9):.1f} FLOP/byte")    