# MiniRoofline

*A first-principles cost model for transformer inference on CPU, validated three ways, showing why standard roofline predictions miss by 1.4–7× on Apple M4 Pro.*

---

## What this is

MiniRoofline is a Python package and set of four self-contained experiments that measure where transformer inference on CPU fails to hit its theoretical performance ceiling. It provides analytical cost formulas validated against a real HuggingFace GPT-2, a reproducible benchmarking harness for Apple Silicon, and empirical characterisation of the M4 Pro's compute-saturation behaviour. The intended reader is an ML systems engineer or researcher curious about the mechanistic gap between roofline predictions and observed CPU inference latency.

## Author

Julio Argota — mathematics PhD, physics MSc, computer science BSc, nine years as a software engineer. This project is my transition into machine learning research.

[LinkedIn](https://www.linkedin.com/in/julio-narciso-argota-quiroz-a6167824/) · [Github](https://github.com/julio-argota-q/)

## Key findings

- **The standard roofline model over-predicts transformer inference throughput
  by 1.7–7.9×** on M4 Pro, with the gap shrinking monotonically as sequence
  length grows (S=32: 7.9×; S=128: 2.8×; S=512: 1.7×). At the machine's
  measured peak of 3.33 TFLOP/s and 240 GB/s memory bandwidth (ridge point
  I* ≈ 13.9 FLOP/byte), the naive roofline predicts latencies far below
  what GPT-2 actually achieves.

- **Apple's AMX unit saturates at M ≈ 512** for fp32 matmuls. Holding
  K=768 and N=3072 fixed (the MLP FC1 shape), achieved throughput rises
  from 615 GFLOP/s at M=8 to 3238 GFLOP/s at M=512, then plateaus. GPT-2
  inference at typical sequence lengths sits below this knee.

- **Prefill and decode achieve dramatically different throughput** on the
  same hardware with identical per-token FLOPs — the clearest case of the
  compute-bound / memory-bound divide in transformer inference.

- **Per-shape roofline calibration predicts throughput above what is
  achievable.** Applying per-shape peaks from isolated microbenchmarks
  yields a latency prediction below measured at long sequences — the
  roofline model becomes too fast, not too slow. In-workload throughput
  is systematically suppressed relative to isolated microbenchmarks.

- **fvcore undercounts attention linearly with sequence length**, missing
  all attention-internal matmul operations. Undercount grows from 0.6%
  at S=32 to 7.7% at S=512, extrapolating to 25%+ at S=2048 — a real
  limitation for long-context inference cost estimation.

## Validation

The cost model is validated three independent ways:

1. **NumPy reference implementation** counts every FLOP operation individually; ratio to closed-form formulas: 1.042.
2. **Component cross-check**: all Linear-module operations match closed-form formulas to 6 significant figures.
3. **fvcore comparison** at three sequence lengths (S=32, 128, 512): total FLOPs agree within 2–8%, with every discrepancy quantitatively explained.

See `notebooks/verify_against_flops.py` for the reproducible validation harness.

## Hardware constants

All predictions in this repository use empirically measured, not
spec-sheet, hardware constants:

| Quantity          | Measured value   | % of spec |
|-------------------|------------------|-----------|
| Peak FLOP/s (fp32) | 3.33 TFLOP/s     | 72%       |
| Memory bandwidth   | 240 GB/s         | 88%       |
| Ridge point I*     | 13.9 FLOP/byte   | —         |

Peak measurement uses a 4096² matmul with warmup + median-of-30
protocol; memory bandwidth uses a 1 GB tensor clone with the same
protocol. Run-to-run variance is approximately 15%, so peak values
should be understood with an implicit uncertainty band. See
`src/miniroofline/cost_model/hardware.py` for the measurement helpers.

## Repository structure

```
mini-roofline/
├── src/miniroofline/                   package
│   ├── cost_model/                     Analytical FLOP and memory formulas, validated against fvcore
│   ├── benchmark/                      Timing infrastructure with warmup + median-of-N protocol
│   └── profiler/                       Hook-based per-component timing
├── experiments/                        Four self-contained experiments producing JSON output
├── notebooks/                          NumPy reference implementation + figure generation
├── tests/                              Unit tests (matmul FLOPs, roofline classification)
├── figures/                            Four figures as PDF and PNG
├── report.md                           Full report of findings
└── README.md                           This file
```

## Quickstart

```bash
git clone https://github.com/<you>/mini-roofline.git
cd mini-roofline
uv sync          # install
make test        # verify install
make f_peak      # run flops peak benchmark
make m_peak      # run memory bandwith benchmark
make exp1        # run experiment 1
make exp2        # run experiment 2
make exp3        # run experiment 3 
make exp4        # run experiment 4
make figs        # the create figures
uv run pytest    # run tests
```

## Reproducing findings

All experiments write JSON to `experiments/results/exp{N}/`. The report
and figures read from these JSONs; nothing is hardcoded. Total runtime
to reproduce every figure from a clean state is approximately 4 minutes
on an M4 Pro.

| Experiment | Measures | Runtime | Figure |
|---|---|---|---|
| exp1_baseline | Component timing at S∈{32,128,512} | ~1 min | Fig 1, Fig 4 |
| exp2_validation | Per-component roofline classification | <1 s (reads Exp 1) | — |
| exp3_shape_sensitivity | Matmul throughput by shape | ~2 min | Fig 2, Fig 3 |
| exp4_calibrated_roofline | Shape-calibrated prediction | <1 s (reads Exp 1+3) | Fig 1 |

All numbers above are reproduced by running make exp1 exp3 on Apple M4 Pro. Actual values may vary within ~15% between runs due to measurement noise.

## Requirements

- **Machine**: Apple M4 Pro (or similar Apple Silicon); other hardware will show different constants but the same qualitative patterns
- **OS**: macOS 14+
- **Python**: 3.11+
- **Package manager**: `uv` (or `pip install -e .`)
- **Disk**: ~500 MB (mostly HuggingFace model cache)
- **Time to reproduce all figures from clean environment**: ~4 minutes

## Design decisions

- CPU-only; GPU extension in Future Work
- fp32 throughout; bf16/int8 would shift ridge point
- GPT-2 as the primary case; findings extend to modern architectures with GQA
- Convention: 1 MAD = 2 FLOPs (Kaplan et al., 2020)
- Softmax, causal mask, and biases omitted from FLOP counts (<0.2% of total)

## Read next

- [`report.md`](docs/report.md) — full writeup with methodology, results, and discussion
- [`experiments/`](./experiments/) — four self-contained experiment scripts

## License

MIT
