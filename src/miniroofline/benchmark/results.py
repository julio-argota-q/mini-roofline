"""
results.py
----------
Structured output for benchmark runs.

Every experiment writes one BenchmarkResult per configuration. These get
loaded back in notebooks/02_results.ipynb to produce plots and tables.

Design rules:
  - Everything serialisable to JSON — no torch tensors, no numpy arrays.
    Convert lists if needed.
  - Hardware metadata recorded with every result so a single file is
    self-describing.
  - Predicted and measured fields are sibling fields so that
    error analysis is trivial: error = (measured - predicted) / predicted.
  - One file per experiment configuration. Aggregation happens in analysis.
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any
from miniroofline.benchmark.timing import TimingResult, get_hardware_metadata


@dataclass
class BenchmarkResult:
    """One full benchmark measurement of one model configuration."""

    # Configuration
    model: str                            # e.g. "gpt2", "distilgpt2"
    batch_size: int
    seq_len: int
    dtype: str = "float32"
    device: str = "cpu"
    mode: str = "prefill"                 # "prefill", "decode", "generate"

    # Timing (raw measurement)
    prefill: dict | None = None           # TimingResult.to_dict() output
    decode_per_token: dict | None = None  # for decode mode

    # Predicted (from cost model)
    predicted_flops: int = 0
    predicted_memory_bytes: int = 0
    predicted_arithmetic_intensity: float = 0.0
    predicted_latency_s: float = 0.0
    predicted_class: str = ""             # compute_bound / memory_bound / ...

    # Measured (from profiler + fvcore)
    measured_flops: int | None = None     # from fvcore
    measured_peak_memory_bytes: int | None = None   # from memory_profiler

    # Per-component (from hooks)
    component_times_s: dict[str, float] = field(default_factory=dict)
    component_classes: dict[str, str] = field(default_factory=dict)

    # Metadata
    hardware: dict = field(default_factory=dict)
    timestamp: str = ""
    notes: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        if not self.hardware:
            self.hardware = get_hardware_metadata()

    # ------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------

    @property
    def median_latency_s(self) -> float:
        """Median latency from the appropriate timing field for this mode."""
        if self.mode == "prefill" and self.prefill:
            return self.prefill["median_s"]
        if self.mode == "decode" and self.decode_per_token:
            return self.decode_per_token["median_s"]
        return 0.0

    @property
    def flops_error_pct(self) -> float | None:
        """Prediction error: 100 * (measured - predicted) / predicted."""
        if self.measured_flops is None or self.predicted_flops == 0:
            return None
        return 100 * (self.measured_flops - self.predicted_flops) / self.predicted_flops

    @property
    def latency_error_pct(self) -> float | None:
        """Latency prediction error."""
        m = self.median_latency_s
        p = self.predicted_latency_s
        if m == 0 or p == 0:
            return None
        return 100 * (m - p) / p

    @property
    def achieved_gflops(self) -> float | None:
        """Effective throughput (measured FLOPs / measured latency)."""
        m = self.median_latency_s
        if m == 0:
            return None
        flops = self.measured_flops or self.predicted_flops
        return flops / m / 1e9

    # ------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------

    def to_dict(self) -> dict:
        d = asdict(self)
        # add derived fields for convenience
        d["derived"] = {
            "median_latency_s": self.median_latency_s,
            "flops_error_pct": self.flops_error_pct,
            "latency_error_pct": self.latency_error_pct,
            "achieved_gflops": self.achieved_gflops,
        }
        return d

    def filename(self) -> str:
        """Canonical filename for this configuration."""
        return f"{self.model}_b{self.batch_size}_s{self.seq_len}_{self.mode}.json"


def save_result(result: BenchmarkResult, out_dir: str | Path) -> Path:
    """Write a benchmark result to JSON. Returns the path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / result.filename()
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    return path


def load_result(path: str | Path) -> BenchmarkResult:
    """Load a benchmark result from JSON."""
    with open(path) as f:
        d = json.load(f)
    d.pop("derived", None)  # derived fields are recomputed
    return BenchmarkResult(**d)


def load_all_results(results_dir: str | Path) -> list[BenchmarkResult]:
    """Load every JSON file in a results directory."""
    results_dir = Path(results_dir)
    return [load_result(p) for p in sorted(results_dir.glob("*.json"))]


if __name__ == "__main__":
    # Round-trip test
    from miniroofline.benchmark.timing import TimingResult

    timing = TimingResult(
        label="test",
        n_runs=10,
        n_warmup=2,
        times_s=[0.01, 0.011, 0.012, 0.011, 0.010, 0.012, 0.011, 0.010, 0.011, 0.012],
    )

    result = BenchmarkResult(
        model="gpt2",
        batch_size=1,
        seq_len=128,
        prefill=timing.to_dict(),
        predicted_flops=22_359_000_000,
        predicted_latency_s=0.011,
        predicted_class="compute_bound",
        measured_flops=22_400_000_000,
    )

    print("Filename:", result.filename())
    print("Median latency:", f"{result.median_latency_s*1000:.2f} ms")
    print("FLOPs error:", f"{result.flops_error_pct:.2f}%")
    print("Latency error:", f"{result.latency_error_pct:.2f}%")
    print("Achieved:", f"{result.achieved_gflops:.0f} GFLOP/s")

    path = save_result(result, "/tmp/test_results")
    print(f"\nSaved: {path}")

    loaded = load_result(path)
    print(f"Loaded: {loaded.filename()}")
    print(f"Match: {loaded.predicted_flops == result.predicted_flops}")
