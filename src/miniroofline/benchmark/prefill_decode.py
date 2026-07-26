"""
prefill_decode.py
-----------------
Separate measurement of prefill (prompt processing) vs decode (per-token generation).

Why separate them?
  Prefill processes S tokens in one forward pass — O(S²) attention,
  large matmuls, throughput-friendly.

  Decode processes 1 token at a time using the KV cache. Per-token cost
  is dominated by MLP (always 1 token through 4d * d * d matmul) plus
  attention against the growing cache.

  Mixing them in one measurement (e.g. timing `model.generate()`) hides
  the structural difference. The roofline analysis depends on telling
  them apart.

Derivation reminder (from flops.py):
  prefill FLOPs ≈ L * (12*B*S*d^2 + 4*B*S^2*d)
  decode FLOPs per token ≈ L * (24*B*d^2 + 4*B*S_ctx*d)

  Ratio prefill / (S * decode_per_token):
    ≈ S * (12*d^2 + 4*S*d) / (S * (24*d^2 + 4*S*d))
    For S << d: ratio ≈ 0.5 (per-token decode is 2x prefill-per-token)
    For S >> d: ratio ≈ 1 (attention dominates both, similar cost)

  In practice on M4 Pro the ratio is closer to 1 — small batched matmul
  in decode hits framework overhead. This is a finding worth measuring.

Output:
  Each function returns a BenchmarkResult with prefill or decode_per_token
  populated. Save with save_result() and aggregate in the analysis step.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from miniroofline.benchmark.timing import (
    time_callable, set_perf_cores, TimingResult,
)
from miniroofline.benchmark.results import BenchmarkResult
from miniroofline.cost_model.flops import flops_prefill_for_model, flops_decode_step, GPT2_CONFIGS
from miniroofline.cost_model.roofline import analyze_model, DEFAULT_HW


# ---------------------------------------------------------------------------
# Prefill
# ---------------------------------------------------------------------------

def benchmark_prefill(
    model,
    tokenizer,
    model_name: str,
    batch_size: int = 1,
    seq_len: int = 128,
    n_warmup: int = 5,
    n_runs: int = 30,
    device: str = "cpu",
) -> BenchmarkResult:
    """
    Measure latency of a single forward pass on a prompt of length seq_len.

    This is the "prefill" phase — the model processes all S tokens at once,
    producing logits for the next token but generating nothing.

    Returns BenchmarkResult with predicted and measured fields populated.
    """
    set_perf_cores()
    model.eval()

    # Build a synthetic prompt of exactly seq_len tokens
    # Using random token IDs avoids cache effects from common tokens
    vocab_size = tokenizer.vocab_size if tokenizer else 50000
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Measure
    timing = time_callable(
        lambda: model(input_ids),
        label=f"{model_name}-prefill-b{batch_size}-s{seq_len}",
        n_warmup=n_warmup,
        n_runs=n_runs,
        device=device,
    )

    # Predict
    flops_breakdown = flops_prefill_for_model(model_name, B=batch_size, S=seq_len)
    roofline = analyze_model(model_name, B=batch_size, S=seq_len)

    # Aggregate predictions across components — sum predicted latencies
    predicted_latency = sum(r.predicted_latency_s for r in roofline.values())
    # Total bytes traffic is approximate — sum across components
    total_traffic = sum(r.traffic_bytes for r in roofline.values())
    avg_intensity = (
        flops_breakdown["total_flops"] / total_traffic if total_traffic > 0 else 0
    )

    # Determine majority class
    classes = [r.classification for r in roofline.values()]
    predicted_class = max(set(classes), key=classes.count) if classes else "unknown"

    return BenchmarkResult(
        model=model_name,
        batch_size=batch_size,
        seq_len=seq_len,
        mode="prefill",
        prefill=timing.to_dict(),
        predicted_flops=flops_breakdown["total_flops"],
        predicted_memory_bytes=total_traffic,
        predicted_arithmetic_intensity=avg_intensity,
        predicted_latency_s=predicted_latency,
        predicted_class=predicted_class,
    )


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def benchmark_decode(
    model,
    tokenizer,
    model_name: str,
    prompt_len: int = 128,
    n_generate: int = 16,
    batch_size: int = 1,
    n_warmup: int = 3,
    n_runs: int = 10,
    device: str = "cpu",
) -> BenchmarkResult:
    """
    Measure per-token generation latency with KV cache.

    Process:
      1. Run prefill to build up the KV cache for `prompt_len` tokens.
      2. Generate `n_generate` tokens one at a time, timing each step.
      3. Report median per-token latency.

    HuggingFace passes the KV cache via `past_key_values` — first call
    has no past_key_values, subsequent calls reuse them.
    """
    set_perf_cores()
    model.eval()

    vocab_size = tokenizer.vocab_size if tokenizer else 50000

    def one_decode_run() -> list[float]:
        """One decode run: warmup prefill, then n_generate timed steps."""
        import time

        input_ids = torch.randint(0, vocab_size, (batch_size, prompt_len))

        # Prefill — build the cache (untimed)
        with torch.no_grad():
            out = model(input_ids, use_cache=True)
        past = out.past_key_values
        next_token = out.logits[:, -1:, :].argmax(-1)

        step_times: list[float] = []
        for _ in range(n_generate):
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(next_token, past_key_values=past, use_cache=True)
            step_times.append(time.perf_counter() - t0)
            past = out.past_key_values
            next_token = out.logits[:, -1:, :].argmax(-1)
        return step_times

    # Warmup
    for _ in range(n_warmup):
        _ = one_decode_run()

    # Measure: collect per-token times across n_runs runs
    all_step_times: list[float] = []
    for _ in range(n_runs):
        all_step_times.extend(one_decode_run())

    timing = TimingResult(
        label=f"{model_name}-decode-prompt{prompt_len}",
        n_runs=len(all_step_times),
        n_warmup=n_warmup * n_generate,
        times_s=all_step_times,
    )

    # Predict (decode step at S_ctx = prompt_len)
    cfg = GPT2_CONFIGS[model_name]
    decode_flops = flops_decode_step(
        B=batch_size,
        S_ctx=prompt_len,
        L=cfg["L"], d=cfg["d"], H=cfg["H"], d_ff=cfg["d_ff"],
    )

    # Predicted latency: roughly decode_flops / peak_flops for compute-bound,
    # or flops / (BW * intensity) for memory-bound. Conservatively use peak.
    # predicted_latency = decode_flops["total_flops"] / DEFAULT_HW.peak_flops_fp32
    
    # Decode is memory-bound: dominated by loading all model weights for 1 token
    from miniroofline.cost_model.memory import bytes_model_weights
    weight_bytes = bytes_model_weights(**{k: cfg[k] for k in ("L", "d", "d_ff", "vocab")})["weight_bytes"]
    predicted_latency = weight_bytes / DEFAULT_HW.memory_bw

    return BenchmarkResult(
        model=model_name,
        batch_size=batch_size,
        seq_len=prompt_len,
        mode="decode",
        decode_per_token=timing.to_dict(),
        predicted_flops=decode_flops["total_flops"],
        predicted_latency_s=predicted_latency,
        predicted_class="overhead_sensitive",   # decode is usually overhead-dominated
        notes=f"prompt={prompt_len}, generated={n_generate} per run, {n_runs} runs",
    )


# ---------------------------------------------------------------------------
# Combined ratio analysis
# ---------------------------------------------------------------------------

def prefill_decode_ratio(prefill: BenchmarkResult, decode: BenchmarkResult) -> dict:
    """
    Compute the empirical vs analytical prefill/decode ratio.

    Hypothesis (from spec): ratio ≈ S (sequence length).
    Verify whether your measurement supports or refutes this.
    """
    if prefill.seq_len != decode.seq_len:
        raise ValueError("Prefill and decode must have matching seq_len for ratio")

    S = prefill.seq_len
    measured_ratio = prefill.median_latency_s / decode.median_latency_s
    analytical_ratio = float(S)

    return {
        "seq_len": S,
        "prefill_latency_ms": prefill.median_latency_s * 1000,
        "decode_per_token_ms": decode.median_latency_s * 1000,
        "measured_ratio": measured_ratio,
        "analytical_ratio": analytical_ratio,
        "discrepancy_pct": 100 * (measured_ratio - analytical_ratio) / analytical_ratio,
        "interpretation": (
            "Per-token decode is more expensive than the analytical model predicts."
            if measured_ratio < analytical_ratio
            else "Per-token decode is cheaper than the analytical model predicts."
        ),
    }


if __name__ == "__main__":

    print("Loading gpt2...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    print("\n--- Prefill benchmark ---")
    pf = benchmark_prefill(model, tokenizer, "gpt2", batch_size=1, seq_len=128,
                            n_warmup=3, n_runs=10)
    print(f"  Median latency: {pf.median_latency_s*1000:.2f} ms")
    print(f"  Predicted:      {pf.predicted_latency_s*1000:.2f} ms")
    print(f"  Error:          {pf.latency_error_pct:+.1f}%")
    print(f"  FLOPs:          {pf.predicted_flops/1e9:.2f} G")
    print(f"  Class:          {pf.predicted_class}")

    print("\n--- Decode benchmark ---")
    dc = benchmark_decode(model, tokenizer, "gpt2", prompt_len=128, n_generate=8,
                          n_warmup=2, n_runs=5)
    print(f"  Per-token median: {dc.median_latency_s*1000:.2f} ms")
    print(f"  Predicted:        {dc.predicted_latency_s*1000:.2f} ms")
    print(f"  FLOPs/token:      {dc.predicted_flops/1e6:.2f} M")

    print("\n--- Ratio analysis ---")
    ratio = prefill_decode_ratio(pf, dc)
    for k, v in ratio.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
