# MiniRoofline — Research Log

**Transformer inference cost model & profiler on Apple M4 Pro**

*Log: derivations, validations, four experiments, and the mechanistic story they tell.*

- **Author**: Julio Argota
- **Machine**: Apple M4 Pro, 14-core (10 P-cores + 4 E-cores), 24 GB unified memory
- **Target**: Anthropic Fellows Program 2026 (July cohort)

---

## Contents

1. [Hardware characterisation](#1-hardware-characterisation)
2. [Cost model derivation & validation](#2-cost-model-derivation--validation)
3. [Findings](#3-findings)
4. [The paper's central narrative](#4-the-papers-central-narrative)
5. [Design decisions & rationale](#5-design-decisions--rationale)
6. [Corrections during the process](#6-corrections-during-the-process)
7. [Ready-to-use paragraphs for the paper](#7-ready-to-use-paragraphs-for-the-paper)
8. [Figures for the paper](#8-figures-for-the-paper)

---

## 1. Hardware characterisation

All roofline predictions use measured, not spec-sheet, values. Both peak FLOP/s and memory bandwidth were empirically determined on the target machine.

### 1.1 Measured constants

| Quantity | Spec | Initial measurement | Refined (Exp 3) |
|---|---|---|---|
| Peak FLOP/s (fp32) | 4.60 TFLOP/s | 2.82 TFLOP/s | **3.33 TFLOP/s** |
| Memory bandwidth | 273 GB/s | 226 GB/s | **240 GB/s** |
| Ridge point I* | 16.8 FLOP/byte | 12.5 FLOP/byte | **13.8 FLOP/byte** |

> **Calibration note:** Peak throughput on the same 4096² matmul varies by ~15% across independent runs due to thermal and OS-scheduling effects. Reports should quote peak with an uncertainty band, not as a single number. Update `hardware.py` to the median-of-runs value.

### 1.2 Cache hierarchy

| Level | Performance cores | Efficiency cores |
|---|---|---|
| L1 instruction | 192 KB per core | 128 KB per core |
| L1 data | 128 KB per core | 64 KB per core |
| L2 shared | 16 MB across 5 P-cores | 4 MB across 4 E-cores |
| Cache line size | 128 bytes | 128 bytes |

### 1.3 Measurement protocol

- **Peak FLOP/s**: 4096×4096 fp32 matmul, median of 30 runs after 10 warmup runs, achieved ~48.7 ms per matmul.
- **Memory bandwidth**: 1 GB tensor clone (reads 1 GB + writes 1 GB = 2 GB traffic), median of 30 runs after 5 warmup runs, achieved 9.5 ms.
- **Thread pinning**: `torch.set_num_threads(10)` to pin to performance cores. Non-negotiable — without it timings vary by 2–3×.

---

## 2. Cost model derivation & validation

The cost model was derived from first principles and validated three independent ways, all of which agree.

### 2.1 Three-way validation

| Method | What it validates | Result |
|---|---|---|
| NumPy per-op counting | Manual FLOP count of every operation | Ratio manual/formula = 1.042 ✓ |
| Component cross-check | `flops.py` per-component vs NumPy per-component | Matmuls: exact (0.00%); blocks +2.5–3.6% ✓ |
| fvcore on real GPT-2 | Full-model prediction vs reference tool | Per-Linear-module: 6 sig-fig match at three scales ✓ |

### 2.2 fvcore comparison detail (GPT-2 small, B=1)

Ratios (fvcore-in-FLOPs / prediction) across three sequence lengths, after adding GELU to MLP and using 7·B·S·d for LayerNorm:

| Component | S=32 | S=128 | S=512 |
|---|---|---|---|
| Attention (nn.Linear parts) | **1.000** | **1.000** | **1.000** |
| MLP (with GELU) | 1.003 | 1.003 | 1.003 |
| LM head | **1.000** | **1.000** | **1.000** |
| LayerNorm | 0.672 | 0.672 | 0.672 |
| **Grand total** | 1.006 | 1.020 | **1.077** |

**Key observation**: three of five ratios are exactly 1.000 at every S. The MLP ratio is a constant 0.3% offset from the GELU term (+8·B·S·d_ff), invariant with S. The LayerNorm ratio is a constant convention difference (7 vs ~4 per-element FLOPs). The **grand total ratio grows linearly with S** because the attention-internal matmuls (Q·Kᵀ and softmax·V) scale as S² while everything else scales as S — fvcore misses this term entirely because it doesn't use `nn.Linear`.

Predicted fvcore undercount coefficient: `4·B·L·d / total_matmul_coefficient = 1.49·10⁻⁴` per token. Predicted undercount: **0.48% / 1.91% / 7.65%** at S = 32/128/512. Observed: 0.6% / 2.0% / 7.7%. Matches within 0.1 pp.

> **Convention difference:** fvcore reports MACs, not FLOPs. 1 MAC = 2 FLOPs. All fvcore numbers must be doubled before comparing to Kaplan-style counts.

---

## 3. Findings

Numbered findings from four experiments. Findings 3.1–3.5 came from derivation; 3.6–3.11 from Experiments 1 and 2; 3.12–3.15 from Experiment 3; 3.16–3.18 from Experiment 4.

### From derivation

**3.1 MLP dominates attention at practical sequence lengths.** For GPT-2 small at S=128, MLP is 65% of total FLOPs and attention only 35%. The attention/MLP crossover is at S = 4d = 3072 tokens — well above GPT-2's max context of 1024. Contradicts the popular framing that "transformers are all about attention."

**3.2 M4 Pro compute efficiency lower than memory efficiency.** 61% of spec peak on fp32 matmul vs 83% of spec peak on memory bandwidth. Unified memory is closer to its theoretical limit than the compute path.

**3.3 fvcore undercount of attention grows linearly with sequence length.** fvcore counts operations that flow through `nn.Linear` modules. The two attention-internal matmuls (Q·Kᵀ and softmax·V) are direct tensor operations — no Linear wrapper — so fvcore reports zero FLOPs for them. Undercount is 0.6% at S=32, 2.0% at S=128, 7.7% at S=512. Extrapolation: ~14% at S=1024, >25% at S=2048. **This is a real limitation of fvcore for long-context transformer inference**, not previously documented.

**3.4 Decode per-token FLOPs are similar to prefill per-token.** Prefill: 22.4 GFLOPs across 128 tokens = 175 MFLOPs/token. Decode: 175 MFLOPs per generated token. Nearly identical FLOPs, but arithmetic intensity differs by 128×.

**3.5 KV cache never dominates weights in GPT-2.** KV = weights crossover at S_ctx = 6,751 tokens for GPT-2 small in fp32; well beyond model's 1024 max context. Analytically relevant for future work on larger-context models.

### From Experiments 1 and 2 (baseline & standard roofline)

**3.6 Whole-model prefill achieves 41% of peak matmul throughput.** At S=128: predicted 7.80 ms (assumes ops run at 2.91 TFLOP/s peak); measured 27.9 ms. Model achieves 1.16 TFLOP/s effective. This gap is the paper's motivating observation.

**3.7 Decode achieves ~30× less throughput than prefill on the same hardware.** Same operations, same weights, same machine — prefill 4.6 TFLOP/s effective (aggregated across 128 tokens), decode 0.026 TFLOP/s per token. The 178× throughput ratio for identical FLOPs demonstrates that prefill and decode inhabit fundamentally different points on the roofline: compute-bound vs memory-bound.

**3.8 The prediction gap shrinks monotonically with sequence length.** GPT-2 small on M4 Pro, measured/predicted latency ratio: 6.88× at S=32, 2.41× at S=128, 1.43× at S=512. Extrapolation suggests convergence toward ~1.0× at S ≥ 1024.

**3.9 Component efficiency ranking is LM head > FC2 > FC1 > attention > MLP > LayerNorm.** Time-share compared to FLOP-share reveals a consistent efficiency ordering. LM head, a single large matmul with N=50257, runs closest to peak (time-share 11–21pp below FLOP-share). MLP runs below peak (time-share 8–14pp above FLOP-share). LayerNorm is overhead-sensitive (time-share 2–5pp above near-zero FLOP-share). Directly validates the three-class roofline taxonomy.

**3.10 Compute-bound operations achieve only 37–85% of measured peak throughput.** All matmul-heavy components have arithmetic intensities well above the M4 Pro ridge point. The roofline correctly classifies them compute-bound, but achieved throughput ranges from 37% (MLP) to 85% (LM head). The gap between theoretical and observed throughput is dominated by MLP and attention inefficiency, not by overhead-sensitive components.

**3.11 Standard roofline reduces to naive prediction for all matmul operations.** Because GPT-2's arithmetic intensity in every matmul component exceeds M4 Pro's ridge point by 4–12×, the roofline formula `max(FLOPs/peak, bytes/BW)` reduces to `FLOPs/peak`. The "roofline improvement" over the naive prediction is 0.99× — i.e. none. The gap in the paper is not a memory-bandwidth phenomenon; it is small-matmul inefficiency.

### From Experiment 3 (shape sensitivity)

**3.12 AMX has a saturation knee at M ≈ 512.** Throughput on a fixed-shape matmul (K=768, N=3072) rises from 522 GFLOP/s at M=8 to 3205 GFLOP/s at M=512, then plateaus. Doubling M gains 1.83× at low M, 1.16× near the knee, ~1.00× beyond. Hardware property of the M4 Pro AMX unit under BLAS scheduling.

**3.13 GPT-2 inference at S ≤ 128 sits below the AMX saturation knee.** Every matmul in GPT-2's forward pass has its M dimension equal to sequence length. At S=32: 41–55% of peak. At S=128: 58–102%. At S=512: 105–121%. Mechanistic explanation for the whole-model gaps in Experiment 1.

**3.14 N and K dimensions barely matter above 512.** Comparing GPT-2 shapes at fixed M=128: Out (N=768) achieves 57.9%, FC1 (N=3072) 78.2%, LMH (N=50257) 84.5%. Larger N helps but effect saturates quickly. Why LM head runs near-peak in Exp 1.

**3.15 Reference peak has ~15% run-to-run variance.** A 4096² fp32 matmul achieved 3.33 TFLOP/s in Exp 3 vs 2.91 TFLOP/s originally. Within normal measurement noise; motivates reporting peak with an uncertainty band.

### From Experiment 4 (calibrated roofline)

**3.16 Isolated microbenchmarks overestimate in-context throughput.** Per-shape peaks from isolated matmul benchmarks systematically exceed the throughput those same shapes achieve inside a real forward pass. At S=512, calibrating the roofline to isolated peaks produced a prediction *lower* than measured — the roofline model became too optimistic. Memory system state (cache pressure from previous layers, activation working sets) suppresses in-workload throughput below what warm-loop microbenchmarks show. **Methodological implication:** shape-calibrated roofline models built from isolated benchmarks need a workload-context correction factor.

**3.17 Calibration effectiveness depends on regime.** Per-shape calibration closes 63% of the naive prediction gap at S=32 (where small-M inefficiency dominates), 26% at S=128, and none at S=512. The closer GPT-2 sits to AMX saturation, the less shape inefficiency there is to correct — but the more non-matmul overhead relative to compute. At short sequences, matmul underutilisation dominates; at long sequences, framework overhead and softmax cost dominate.

**3.18 Non-matmul residual is quantitatively consistent with framework overhead.** After per-shape calibration, the residual gap (measured − calibrated) is 12.8 ms at S=32, 13.6 ms at S=128, 25.5 ms at S=512. The near-constancy of the first two is consistent with fixed Python dispatch overhead across 120+ hooked operations per forward pass (12 layers × ~10 ops × ~100 μs per dispatch ≈ 12 ms). The jump at S=512 (+12 ms) matches the analytical cost of softmax on a [1, 12, 512, 512] attention matrix scaling quadratically with S.

---

## 4. The paper's central narrative

Findings 3.6–3.18 synthesise into a single mechanistic story:

> The standard roofline model with a single peak_flops constant systematically over-predicts transformer inference throughput on M4 Pro by 1.4–7×. We identify three distinct sources of this gap and quantify each:
>
> 1. **Shape-dependent AMX underutilisation.** Dominates at short sequence lengths. Explained mechanistically by the M ≈ 512 saturation knee (Exp 3). Can be partially closed by per-shape peak calibration, though isolated benchmarks overestimate in-workload throughput (Exp 4).
>
> 2. **Fixed framework overhead.** Approximately 13 ms constant per forward pass across 12 GPT-2 layers, attributable to Python-to-C++ dispatch across ~120 hooked operations.
>
> 3. **Growing non-matmul cost.** Softmax and LayerNorm contribute an additional ~12 ms at S=512, scaling with the attention matrix size.
>
> Naive calibration to isolated microbenchmarks does not straightforwardly close the gap: at long sequences it can over-predict, revealing that in-workload throughput is systematically lower than isolated microbenchmark peaks — a finding with methodological implications for anyone using shape-specific benchmarks to predict real-workload performance.

---

## 5. Design decisions & rationale

### 5.1 Project scope

- Chose CPU simulator over SAE interpretability for execution predictability in 8 weeks
- Cut Flash Attention experiment; keep only analytical derivation as Future Work
- Cut GPU experiments (CPU-only throughout; explicit limitation)
- Cut multi-hardware profiling (one machine only; precise specs recorded)

### 5.2 Implementation choices

- **Convention**: 1 MAD = 2 FLOPs (Kaplan et al. 2020). All fvcore comparisons corrected.
- **Package manager**: `uv` with `uv.lock` for reproducibility. Docker rejected: distorts CPU timings on Apple Silicon.
- **Thread pinning**: `torch.set_num_threads(10)` to pin to performance cores.
- **Report format**: Quarto (`.qmd` → PDF) rather than LaTeX. Faster iteration.
- **FLOP accounting scope in `flops.py`**: matmul + GELU + LayerNorm only. Softmax, scale, causal mask, and biases omitted (<0.1% of total, individually justified in module docstring).
- **OVERHEAD_THRESHOLD = 3.0**: heuristic separating matmul-heavy regime (measured/predicted 1–3×) from overhead-dominated regime (5–20×). Classifications stable for thresholds in [2, 5].

---

## 6. Corrections during the process

### 6.1 Missing biases and causal mask in tiny transformer
Added QKV/output biases and `np.triu(k=1)` causal mask before completing FLOP derivation. Neither changes total FLOPs by more than 0.1%.

### 6.2 Softmax and weighted-sum shape errors
Fixed factor-of-H/S undercount in softmax formula (`6·B·S → 3·B·H·S²`), and wrong-shape weighted-sum (`2·B·S·d² → 2·B·S²·d`). Both caught by automated cross-check against `flops.py`.

### 6.3 Missing LM head in flops_prefill_for_model
LM head is 2·B·S·d·V = 9.88 GFLOPs at S=128 — 31% of total. Added as separate dict key and included in `total_flops`.

### 6.4 Placeholder hardware constants
Original `hardware.py` used 4.6 TFLOP/s from spec sheet. Replaced with measured 2.82 (initial) then 3.33 (Exp 3) TFLOP/s.

### 6.5 `set_num_interop_threads` called twice
`torch.set_num_interop_threads` can only be called once per process. Prefill+decode benchmark called `set_perf_cores` twice, which threw. Fixed by making the setter idempotent.

### 6.6 Naive decode latency prediction
`benchmark_decode` used FLOPs/peak_flops — a compute-bound assumption inappropriate for decode. Corrected to weight_bytes/memory_bw, reducing predicted-vs-measured gap from ~110× to ~3×.

### 6.7 Mask FLOPs double-counted
Originally counted `flops_mask = 2·B·H·S²`. The `np.where` operation does no floating-point work — mask application is boolean, integer, or select. Corrected to `flops_mask = 0` following Kaplan convention.

### 6.8 LayerNorm constant inconsistency
`flops.py` used `5·B·S·d`; tiny transformer counts `~7·B·S·d`. Both defensible under different conventions. Standardised to `7·B·S·d` in `flops.py` to match the tiny transformer's per-op count. Changes total FLOPs by <0.1% but eliminates a spurious 40% discrepancy in the verify script's output.

### 6.9 Stale docstring in flops.py
`flops_attention_layer` docstring mentioned crossover at "S ≈ 1086" which was from an earlier (attention-internal only) derivation. Corrected to the useful crossover: attention S²-term vs MLP at `S = 4d = 3072`.

### 6.10 analyze_model omits LM head
`analyze_model` in `roofline.py` covers only transformer blocks (attention + MLP + LayerNorm), not LM head, embeddings, or residuals. LM head is 31% of total FLOPs — a real omission. Added; embeddings and residuals kept omitted (< 0.01%) with an explanatory comment.

---

## 7. Ready-to-use paragraphs for the paper

### 7.1 Hardware characterisation (Methods)

> All measurements were performed on an Apple M4 Pro with 14 CPU cores (10 P-cores + 4 E-cores) and 24 GB of unified memory. PyTorch was pinned to the performance cores via `torch.set_num_threads(10)`. We measured peak achievable FLOP rate via a 4096×4096 fp32 matrix multiplication (median of 30 runs, 10 warmup), yielding 3.33 TFLOP/s — 72% of Apple's 4.6 TFLOP/s theoretical peak. Memory bandwidth was measured via a 1 GB tensor clone, yielding 229 GB/s. The resulting ridge point is I* ≈ 14.5 FLOP/byte. Run-to-run variance for the peak measurement was approximately 15%; we report the median.

### 7.2 Cross-hardware implication (Discussion)

> The ridge point on M4 Pro is markedly higher than typical x86 laptops (I* ≈ 3 FLOP/byte). This is a consequence of Apple's high-bandwidth unified memory rather than a difference in peak FLOP/s. It shifts more transformer operations into the compute-bound regime than one would predict from x86-based prior work on transformer inference cost.

### 7.3 Validation against fvcore (Methods)

> We validate our closed-form FLOP formulas by comparison with `fvcore.nn.FlopCountAnalysis` on the reference HuggingFace `gpt2` model at three sequence lengths (S=32, 128, 512), converting fvcore's reported MACs to FLOPs using the convention 1 MAC = 2 FLOPs. All `nn.Linear`-based components (QKV projections, output projection, MLP fc1 and fc2, LM head) match fvcore to six significant figures at every S, with ratios of exactly 1.000. The MLP component is 0.3% higher than fvcore due to our inclusion of the GELU activation, a constant offset invariant with S. LayerNorm counts differ by a factor of 1.49 due to differing per-element constants (0.03% of total). The remaining discrepancy comes from attention-internal matmul operations (Q·Kᵀ and attention_weights·V) that fvcore does not detect because they are direct tensor operations rather than `nn.Linear` calls. Because these operations scale as S² while total FLOPs scale linearly in S, fvcore's undercount grows linearly with sequence length: 0.6% at S=32, 2.0% at S=128, 7.7% at S=512. Extrapolation gives ~14% at S=1024 and >25% at S=2048.

### 7.4 Three-source decomposition (Discussion, headline claim)

> The standard roofline model with a single peak_flops constant systematically over-predicts transformer inference throughput on M4 Pro by 1.4–7×. We identify three distinct sources of this gap and quantify each: (1) shape-dependent AMX underutilisation, dominating at short sequence lengths, mechanistically explained by an M ≈ 512 saturation knee; (2) fixed framework overhead from PyTorch dispatch, contributing ~13 ms constant per forward pass; (3) growing non-matmul cost (softmax, LayerNorm) with sequence length. Naive calibration to isolated microbenchmarks does not straightforwardly close the gap: at long sequences it can over-predict, revealing that in-workload throughput is systematically lower than isolated microbenchmark peaks.

### 7.5 Convention statement (Methods)

> We adopt the convention 1 multiply-add = 2 FLOPs, matching Kaplan et al. (2020) and Hoffmann et al. (2022). Our closed-form FLOP formulas count matmul operations, GELU activations, and LayerNorm at 7·B·S·d per instance. We omit softmax, scaling by √Dh, causal masking (a boolean operation), attention dropout (a no-op at inference), and bias additions; these operations sum to less than 0.2% of total FLOPs at all sequence lengths tested and would obscure the scaling laws that are the paper's central claims.

### 7.6 Threshold statement (Methods)

> We classify operations as overhead-sensitive when measured latency exceeds the roofline prediction by more than a factor of 3. This threshold is a heuristic separating the observed bimodal distribution of matmul-heavy operations (typically 1.0–2.5× the roofline bound) from operations dominated by framework overhead (typically 5–20× the bound). Classifications are stable for thresholds in [2, 5].

---

## 8. Figures for the paper

Four figures tell the entire narrative. Generated by `notebooks/make_figures.py` from saved experiment JSON.

### Figure 1 — Prediction gap vs sequence length

- **X**: sequence length {32, 128, 512}
- **Y**: measured/predicted latency ratio (log scale)
- **Curves**: naive, standard-roofline (coincides with naive per Finding 3.11), calibrated
- **Reference line**: y=1
- **Story**: shows the motivating gap and Experiment 4's calibration outcome (helps at low S, not at high S).

### Figure 2 — AMX saturation curve

- **X**: matrix size (log scale, 128 to 4096, square matmuls)
- **Y**: achieved GFLOP/s
- **Markers**: % of observed peak at each point
- **Vertical marker**: M=512 (the knee)
- **Story**: paper's key hardware characterisation figure.

### Figure 3 — M-dimension sweep with GPT-2 markers

- **X**: M (log scale, 8–2048), K=768, N=3072 (MLP FC1 shape)
- **Y**: achieved GFLOP/s
- **Annotations**: where GPT-2 sits at S={32, 128, 512}
- **Story**: direct mechanistic explanation for why short-S GPT-2 is inefficient.

### Figure 4 — Component share, measured vs predicted

- **X**: three sequence lengths
- **Y**: fraction of time (measured) or fraction of FLOPs (predicted)
- **Grouped bars per component**: attention, MLP, LayerNorm, LM head
- **Story**: efficiency ordering (Finding 3.9) visually.
