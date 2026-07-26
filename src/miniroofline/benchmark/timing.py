"""
timing.py
---------
Reliable CPU timing for ML workloads.

Why this matters:
  CPU timings are noisy. A single time.perf_counter() call around a
  forward pass can vary by 5-20% run to run due to thermal throttling,
  OS scheduling, BLAS thread contention, and page-cache warmup.

  This module enforces:
    1. Warmup runs (discarded) — let JIT compilation and caches settle
    2. Repeated measurement runs — collect a distribution
    3. Report median (robust to outliers), not mean (sensitive to spikes)
    4. Report IQR alongside median — characterise the noise
    5. Synchronisation barriers where needed (for MPS/CUDA, no-ops on CPU)

For the M4 Pro:
  torch.set_num_threads(10) pins to performance cores. Without this,
  the OS schedules threads on efficiency cores and timings vary by 2-3x.

Reference: PyTorch profiler documentation, section on benchmarking
  https://pytorch.org/tutorials/recipes/recipes/benchmark.html
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Callable, Any
import torch


# Always set this once at process start — call set_perf_cores() to do it
_THREADS_SET = False


def set_perf_cores(n_threads: int = 10) -> None:
    """
    Pin PyTorch to performance cores on Apple Silicon.
    Idempotent — safe to call multiple times.
    M4 Pro has 10 P-cores + 4 E-cores. Default PyTorch uses all 14,
    but the E-cores are 3-4x slower at FLOP-heavy work, so the OS
    can park threads on them and inflate timings.

    Call this once at program start, before any benchmark.
    """
    global _THREADS_SET
    if _THREADS_SET:
        # torch.set_num_threads is safe to call repeatedly;
        # torch.set_num_interop_threads is NOT — only call it once.
        torch.set_num_threads(n_threads)
        return
    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(1)   # avoid contention
    _THREADS_SET = True


def synchronize(device: str = "cpu") -> None:
    """
    Force pending work to complete before stopping the timer.

    On CPU this is a no-op (work is synchronous). Included so the same
    timing code works if you later run on MPS or CUDA without rewriting.
    """
    if device == "cpu":
        return
    if device == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Timing results — what every benchmark returns
# ---------------------------------------------------------------------------

@dataclass
class TimingResult:
    """
    Statistics from a repeated timing measurement.

    Use median for reporting — it is robust to OS jitter.
    Use iqr / median as a noise indicator: if > 0.1, results are unreliable
    and you should rerun with more warmup or check for background load.
    """
    label: str
    n_runs: int
    n_warmup: int
    times_s: list[float]                # raw individual measurements
    median_s: float = 0.0
    mean_s: float = 0.0
    min_s: float = 0.0
    max_s: float = 0.0
    p25_s: float = 0.0
    p75_s: float = 0.0
    iqr_s: float = 0.0                  # p75 - p25
    iqr_relative: float = 0.0           # iqr / median — noise indicator

    def __post_init__(self):
        if self.times_s:
            s = sorted(self.times_s)
            n = len(s)
            self.min_s = s[0]
            self.max_s = s[-1]
            self.median_s = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
            self.mean_s = sum(s) / n
            self.p25_s = s[n // 4]
            self.p75_s = s[(3 * n) // 4]
            self.iqr_s = self.p75_s - self.p25_s
            self.iqr_relative = self.iqr_s / self.median_s if self.median_s > 0 else 0.0

    def summary(self) -> str:
        noise_warning = ""
        if self.iqr_relative > 0.1:
            noise_warning = "  ⚠ noisy (IQR/median > 10%)"
        return (
            f"{self.label}: "
            f"median {self.median_s*1000:.2f} ms  "
            f"IQR [{self.p25_s*1000:.2f}, {self.p75_s*1000:.2f}] "
            f"(rel {self.iqr_relative*100:.1f}%){noise_warning}"
        )

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "n_runs": self.n_runs,
            "n_warmup": self.n_warmup,
            "median_s": self.median_s,
            "mean_s": self.mean_s,
            "min_s": self.min_s,
            "max_s": self.max_s,
            "p25_s": self.p25_s,
            "p75_s": self.p75_s,
            "iqr_s": self.iqr_s,
            "iqr_relative": self.iqr_relative,
            "times_s": self.times_s,
        }


# ---------------------------------------------------------------------------
# Core timing primitive
# ---------------------------------------------------------------------------

def time_callable(
    fn: Callable[[], Any],
    label: str = "",
    n_warmup: int = 5,
    n_runs: int = 30,
    device: str = "cpu",
) -> TimingResult:
    """
    Time a no-argument callable with warmup + repeated runs.

    Args:
        fn        : zero-arg callable to time (use functools.partial or lambda)
        label     : human-readable name for the measurement
        n_warmup  : runs to discard before measurement (default 5)
        n_runs    : measurement runs (default 30 — gives stable median)
        device    : "cpu", "mps", or "cuda" — controls synchronisation

    Returns:
        TimingResult with median, IQR, and full distribution.

    Example:
        result = time_callable(
            lambda: model(input_ids),
            label="gpt2-prefill-s128",
            n_warmup=5,
            n_runs=30,
        )
        print(result.summary())
    """
    if not _THREADS_SET:
        set_perf_cores()

    # Warmup — discarded
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = fn()
        synchronize(device)

    # Measurement
    times: list[float] = []
    for _ in range(n_runs):
        synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = fn()
        synchronize(device)
        times.append(time.perf_counter() - t0)

    return TimingResult(
        label=label or fn.__name__,
        n_runs=n_runs,
        n_warmup=n_warmup,
        times_s=times,
    )


# ---------------------------------------------------------------------------
# Convenience: time a model forward pass
# ---------------------------------------------------------------------------

def time_forward(
    model: torch.nn.Module,
    inputs: dict | torch.Tensor,
    label: str = "forward",
    n_warmup: int = 5,
    n_runs: int = 30,
    device: str = "cpu",
) -> TimingResult:
    """
    Time a single forward pass of a model with the given inputs.

    Accepts either a tensor (positional arg) or a dict of kwargs
    (e.g. {"input_ids": ..., "attention_mask": ...}).
    """
    model.eval()

    if isinstance(inputs, dict):
        fn = lambda: model(**inputs)
    else:
        fn = lambda: model(inputs)

    return time_callable(fn, label=label, n_warmup=n_warmup, n_runs=n_runs, device=device)


# ---------------------------------------------------------------------------
# Hardware metadata — record alongside every measurement
# ---------------------------------------------------------------------------

def get_hardware_metadata() -> dict:
    """
    Capture everything needed to reproduce the measurement.

    Record this with every benchmark result. Reviewers need to know
    which CPU produced the numbers; numbers are meaningless without it.
    """
    import platform
    import sys

    meta = {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
    }

    # Try to get more detail on macOS
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            meta["cpu_brand"] = result.stdout.strip()

        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            meta["ram_bytes"] = int(result.stdout.strip())
    except Exception:
        pass

    return meta


if __name__ == "__main__":
    # Quick test — time a small matmul
    set_perf_cores()
    print("Hardware metadata:")
    for k, v in get_hardware_metadata().items():
        print(f"  {k}: {v}")

    print("\nTiming a 1024x1024 matmul:")
    A = torch.randn(1024, 1024)
    B = torch.randn(1024, 1024)
    result = time_callable(lambda: A @ B, label="matmul-1024", n_runs=50)
    print(result.summary())

    # Sanity check: predicted FLOPs / measured time = achieved GFLOP/s
    flops = 2 * 1024 ** 3
    gflops = flops / result.median_s / 1e9
    print(f"  achieved: {gflops:.0f} GFLOP/s")
