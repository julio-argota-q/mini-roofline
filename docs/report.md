# MiniRoofline: A Cost Model for Transformer Inference on Apple M4 Pro

**Author**: Julio Narciso Argota-Quiroz
**Date**: 2026-07
**Status**: Interim technical report (v0.1).

---

## 1. Summary

Standard roofline predictions of transformer inference latency, using a single peak-throughput constant for the target hardware, are systematically incorrect on CPU. On Apple M4 Pro, running fp32 inference of GPT-2 small, the roofline model over-predicts throughput by 1.7–7.9× depending on sequence length. This gap has been observed anecdotally in the ML systems community but not decomposed into its causes.

This report presents MiniRoofline, a first-principles analytical cost model for transformer inference and a set of four experiments characterising the gap between roofline predictions and observed latency on Apple M4 Pro. The cost model was derived from scratch and validated three independent ways: a NumPy reference implementation that counts every operation individually (ratio to closed-form formulas: 1.042), a component cross-check that verifies matmul formulas match to six significant figures, and a full-model comparison against `fvcore.nn.FlopCountAnalysis` at three sequence lengths (agreement within 2–8% across a 16× range in S, with every discrepancy quantitatively explained).

The four experiments produce a coherent mechanistic story. Experiment 1 measures the whole-model prediction gap at 7.9× / 2.8× / 1.7× for S = 32 / 128 / 512. Experiment 2 shows the standard roofline model correctly classifies every matmul-heavy component as compute-bound, but the "compute-bound" prediction still overshoots — the gap is not memory-related. Experiment 3 identifies the mechanism: Apple's AMX unit saturates at M ≈ 512, and GPT-2's M dimension is the sequence length, so short-sequence inference sits below the saturation knee. Experiment 4 attempts to close the gap by calibrating the roofline to per-shape peaks measured in isolation; this closes most of the gap at S=32 but produces an *optimistic* prediction at S=512, revealing that in-workload throughput is systematically suppressed relative to isolated microbenchmarks.

---

## 2. Findings

The eighteen findings from four experiments can be organised into three groups: what characterises the hardware, where the roofline model fails, and why it fails.

### 2.1 Hardware characterisation

**M4 Pro peak throughput is 3.33 TFLOP/s on fp32 matmul**, measured on a 4096² multiplication with a warmup + median-of-30 protocol. This is 72% of Apple's 4.6 TFLOP/s spec-sheet peak. Memory bandwidth is 240 GB/s, 88% of the 273 GB/s specification. The resulting ridge point is I* ≈ 13.9 FLOP/byte — about 5× higher than typical x86 laptops (≈ 3 FLOP/byte), a direct consequence of Apple's unified memory architecture rather than a difference in peak compute.

**The peak throughput exhibits ~15% run-to-run variance**, motivating reporting peak values with an uncertainty band. A single measurement is not reliable to more than two significant figures.

**AMX has a compute-saturation knee at M ≈ 512** for fp32 matmuls. Holding K=768 and N=3072 fixed (the shape of GPT-2's MLP FC1 layer), achieved throughput rises from 615 GFLOP/s at M=8 to 3238 GFLOP/s at M=512, then plateaus. Above the knee, doubling M gains approximately 0%. This is a hardware property of the AMX unit under BLAS scheduling.

### 2.2 Where the roofline model fails

**The naive prediction gap shrinks monotonically with sequence length**. Whole-model measured/predicted latency ratios are 7.9× at S=32, 2.8× at S=128, and 1.7× at S=512. Extrapolation to S=1024 suggests convergence toward ~1.0×. The gap exists because the naive prediction assumes every operation runs at peak throughput; observed throughput ranges from below 20% (small-M matmuls at S=32) to over 85% (LM head at S=128) of the reference peak.

**The standard roofline model reduces to the naive prediction for all matmul-heavy components of GPT-2**. Every component has arithmetic intensity 4–12× above the ridge point (I* ≈ 13.9 FLOP/byte), so the roofline formula max(FLOPs/peak, bytes/BW) reduces to FLOPs/peak. The "roofline improvement" over naive is exactly 0.99× — i.e. none. The gap is not a memory-bandwidth phenomenon.

**LayerNorm is overhead-sensitive**: measured time exceeds the roofline prediction by more than 3× (the threshold for classification) despite doing near-zero FLOPs. Time-share is 2–5% of the forward pass; FLOP-share is <0.1%. This validates our extension of the standard roofline taxonomy with a third class (compute-bound / memory-bound / overhead-sensitive).

**Prefill and decode achieve very different throughputs on the same hardware**. GPT-2 small produces the same 175 MFLOPs per token in both modes, yet prefill runs at multi-TFLOP/s effective throughput while per-token decode runs at tens of GFLOP/s. Same operations, same weights, same machine — but prefill amortises weight loading across all tokens (compute-bound regime), while decode reloads weights for each token (memory-bound regime). This is the classic prefill/decode divergence, quantified in a controlled setup.

### 2.3 The mechanism

**GPT-2 at S ≤ 128 sits below the AMX saturation knee**. Every matmul in the forward pass has M equal to the sequence length. At S=32 the GPT-2 shapes achieve roughly 40–55% of reference peak throughput; at S=128 they achieve 60–100%; at S=512 they reach the plateau. This is the mechanistic explanation for the whole-model gap: the naive roofline assumed the matmuls would hit reference peak; instead they run below it, with efficiency directly determined by sequence length.

**The N and K dimensions matter much less than M above the saturation threshold**. Comparing GPT-2 shapes at fixed M=128, larger N dimensions (LM head with N=50257 vs Out projection with N=768) help, but the effect saturates quickly. This is why the LM head — a single large matmul with enormous N — runs near-peak while the smaller MLP and attention matmuls do not, even at the same sequence length.

**Per-shape calibration closes the roofline gap at short sequences but not long ones**. Applying per-shape peaks from isolated microbenchmarks to the roofline prediction closes most of the naive gap at S=32 and partially at S=128, but at S=512 the calibrated prediction becomes optimistic (predicted latency below measured). Isolated microbenchmarks systematically overestimate the throughput a shape achieves inside a real forward pass, likely because cache state from preceding layers suppresses in-workload throughput. This is a methodological finding worth stating explicitly: shape-calibrated roofline models built from isolated benchmarks need a workload-context correction factor.

**The residual gap after per-shape calibration decomposes into fixed framework overhead and growing non-matmul cost**. The near-constancy of the residual at short and moderate sequence lengths is consistent with fixed PyTorch dispatch overhead across ~120 hooked operations per forward pass (12 layers × ~10 ops × ~100 μs per dispatch ≈ 12 ms). The additional growth at S=512 matches the analytical cost of softmax on a [1, 12, 512, 512] attention matrix, which scales quadratically with S.

**Together, these findings yield a three-source decomposition of the roofline gap**: (1) shape-dependent AMX underutilisation, dominant at short sequence lengths, mechanistically explained by the M ≈ 512 knee; (2) fixed PyTorch dispatch overhead of approximately 13 ms per forward pass, invariant with S; (3) non-matmul cost that grows with sequence length, primarily softmax at long S.

### 2.4 A methodological finding on fvcore

Independent of the main results, our validation revealed a systematic limitation of `fvcore.nn.FlopCountAnalysis`. Because fvcore counts operations flowing through `nn.Linear` modules with weight tensors, it does not detect the two attention-internal matmuls (Q·Kᵀ and softmax·V), which are direct tensor operations between activations. These operations scale as S² while total FLOPs scale linearly with S, so fvcore's relative undercount grows linearly with sequence length: 0.6% at S=32, 2.0% at S=128, 7.7% at S=512. Extrapolation predicts undercount exceeding 25% at S=2048. This is a real concern for anyone using trace-based FLOP counting to estimate long-context transformer inference cost, and to our knowledge has not been previously documented.

---

## 3. Methodology

### 3.1 Hardware and measurement

All measurements were performed on an Apple M4 Pro (14-core: 10 P-cores + 4 E-cores, 24 GB unified memory) running macOS. PyTorch was pinned to the performance cores via `torch.set_num_threads(10)`; without this pinning, timings varied by 2–3× as the OS scheduled threads onto efficiency cores. All timings use `torch.inference_mode()` to disable gradient tracking and tensor version counting. Peak FLOP throughput was measured via a 4096² fp32 matrix multiplication (10 warmup runs discarded, median of 30 measurement runs), yielding 3.33 TFLOP/s. Memory bandwidth was measured via a 1 GB tensor clone (reads 1 GB + writes 1 GB = 2 GB traffic per operation), with the same warmup and repetition protocol, yielding 240 GB/s. The resulting ridge point is I* ≈ 13.9 FLOP/byte. Component-level timing used forward hooks registered on each transformer submodule; hook overhead was verified at 2–5% of total forward-pass time across sequence lengths.

### 3.2 Cost model

The cost model derives FLOP counts and memory-traffic estimates from first principles for each transformer component: QKV projections, attention scores, weighted sum, output projection, MLP (both linear layers plus GELU activation), LayerNorm, and the LM head. We adopt the convention 1 multiply-add = 2 FLOPs, matching Kaplan et al. (2020) and standard systems literature. The formulas count matmul operations, GELU activation, and LayerNorm (using 7·B·S·d per LayerNorm following per-op counting). Softmax, scaling by √Dₕ, causal masking (a boolean operation with zero FLOPs), attention dropout (no-op at inference), bias additions, and residual adds are omitted; each contributes less than 0.2% of total FLOPs individually. The memory-traffic model counts bytes read and written under a worst-case DRAM assumption (no cache reuse modelled), giving upper bounds on required bandwidth.

### 3.3 Validation

The cost model is validated three independent ways. First, a NumPy reference implementation (`notebooks/tiny_transformer.py`) counts every operation individually at a scaled-down configuration (L=2, d=64, H=4, S=32, V=100); the ratio of manually-counted FLOPs to closed-form formulas is 1.042, with the small excess attributable to biases and LayerNorm. Second, a component cross-check (`notebooks/verify_against_flops.py`) compares closed-form formulas to the NumPy counts operation by operation; all matmul components match to 0.00%, with attention and MLP blocks differing by 2.5–3.6% (biases, softmax, GELU counted in tiny but omitted from closed form). Third, a full-model comparison against `fvcore.nn.FlopCountAnalysis` at S ∈ {32, 128, 512} shows all `nn.Linear` components matching to six significant figures at every sequence length; total FLOP predictions agree within 2–8%, with the discrepancy fully accounted for by three named operations (attention-internal matmuls, GELU, LayerNorm constant difference).

### 3.4 Experiments

Four experiments were run, each self-contained and producing JSON output that downstream analysis reads. Experiment 1 measures whole-model latency and per-component timing via hooks for GPT-2 small at S ∈ {32, 128, 512}, comparing to naive roofline predictions. Experiment 2 re-analyses Experiment 1's data using per-component arithmetic intensity to apply the standard roofline classification (compute-bound / memory-bound). Experiment 3 measures achieved throughput for 30 matmul shapes: a size sweep from 128² to 4096², the shapes appearing in GPT-2 (five shapes at three sequence lengths), and an M-dimension sweep at fixed K=768 and N=3072. Experiment 4 re-analyses Experiment 1's data using the per-shape peaks measured in Experiment 3, producing shape-calibrated latency predictions. Total runtime for all four experiments from a clean environment is approximately four minutes on M4 Pro.

---

## 4. Status

### 4.1 Complete

The cost model is fully derived and validated by three independent methods, with quantitative agreement across a 16× range in sequence length. All four experiments have executed successfully on Apple M4 Pro with reproducible JSON output. Four publication-quality figures have been generated. The repository is packaged with `uv` for one-command reproduction from a clean environment. A minimal unit test suite covers the FLOP formulas, memory-traffic model, roofline classifier, and hardware specification.

### 4.2 Known limitations

This work characterises inference on a single machine (Apple M4 Pro), single dtype (fp32), single model family (GPT-2), single batch size (B=1), and single random seed. All findings are conditional on this scope. The three-source decomposition of the roofline gap is expected to transfer qualitatively to other hardware and models, but the specific numerical values (saturation knee at M ≈ 512, framework overhead of ~13 ms) are hardware- and framework-specific.

---

## Appendix A: Reproducibility

All findings in Section 2 are reproducible with the following commands, executed from a clean checkout of the repository, in approximately four minutes on Apple M4 Pro:

```bash
uv sync
make exp1   # ~1 min
make exp3   # ~2 min
make exp2   # <1 s (reads Exp 1)
make exp4   # <1 s (reads Exp 1+3)
make figs   # <30 s
```

All experimental data is written as JSON to `experiments/results/exp{N}/`. Figure generation reads exclusively from these JSON files; no numerical values are hardcoded. See the repository README for a full quickstart.

## Appendix B: Full findings list

A complete enumeration of eighteen numbered findings, including derivations, supporting numerical evidence, and cross-references to source code, is available in `docs/research_log.md` in the accompanying repository.
