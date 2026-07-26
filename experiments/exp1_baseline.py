"""
experiments/exp1_baseline.py
----------------------------
EXPERIMENT 1 — Baseline component profile

RESEARCH QUESTION:
  Where does GPT-2 actually spend its time on CPU? How does that
  distribution change with sequence length?

HYPOTHESIS:
  At short sequences (S=32), MLP will dominate (~60-65% of time) because
  attention's S² term is small and MLP is O(S·d²). At longer sequences
  (S=512), attention's share should grow toward the S=3072 crossover.
  LayerNorm and embeddings will register as overhead-sensitive: measured
  time much higher than the tiny FLOP count predicts.

METHOD:
  1. Load GPT-2 small (later medium if time permits).
  2. For S ∈ {32, 128, 512}:
     a. Warmup 5 runs
     b. Measure whole-model latency (median of 30)
     c. Attach component hooks and measure per-category time (median of 30)
  3. Compute predicted per-component time from cost model.
  4. Save one JSON per (model, seq_len) config.
  5. Print component share table.

OUTPUTS:
  - experiments/results/exp1/{model}_b{B}_s{S}.json  (structured data)
  - Console table showing measured vs predicted shares

Run:
  uv run python experiments/exp1_baseline.py
"""

import json
from pathlib import Path
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from miniroofline.benchmark.timing import (
    set_perf_cores, time_forward, get_hardware_metadata,
)
from miniroofline.benchmark.results import BenchmarkResult, save_result
from miniroofline.profiler.hooks import ComponentTimer
from miniroofline.cost_model.flops import (
    flops_prefill_for_model
)
from miniroofline.cost_model.hardware import DEFAULT_HW


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MODELS = ["gpt2"]                          # add "gpt2-medium" once gpt2 works
SEQ_LENS = [32, 128, 512]
BATCH_SIZE = 1
N_WARMUP = 5
N_RUNS = 30
OUT_DIR = Path("experiments/results/exp1")


# ─────────────────────────────────────────────────────────────────────────────
# Component-level cost model prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_component_times(model_name: str, B: int, S: int) -> dict[str, float]:
    """
    Predict how much time each component *should* take, if every operation
    ran at peak throughput. This is a strict lower bound — reality will
    be higher (that gap is the finding).

    We use a simple mapping: predicted_time = FLOPs / peak_flops.
    A more sophisticated version would use per-component arithmetic
    intensity to decide compute-bound vs memory-bound — Experiment 2 does that.
    """
    flops = flops_prefill_for_model(model_name, B, S)
    peak = DEFAULT_HW.peak_flops_fp32

    return {
        "attention": flops["attention_total_flops"] / peak,
        "mlp":       flops["mlp_flops"] / peak,
        "layernorm": flops["layernorm_flops"] / peak,
        "lm_head":   flops["lm_head_flops"] / peak,
    }


def predict_component_shares(model_name: str, B: int, S: int) -> dict[str, float]:
    """
    Predicted fraction of total FLOPs per component.
    """
    flops = flops_prefill_for_model(model_name, B, S)
    total = flops["total_flops"]

    return {
        "attention": flops["attention_total_flops"] / total,
        "mlp":       flops["mlp_flops"] / total,
        "layernorm": flops["layernorm_flops"] / total,
        "lm_head":   flops["lm_head_flops"] / total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# One measurement
# ─────────────────────────────────────────────────────────────────────────────

def measure_one_config(model, tokenizer, model_name: str, S: int, B: int) -> BenchmarkResult:
    """
    Measure one (model, seq_len) configuration end-to-end.

    Returns a fully populated BenchmarkResult ready to serialise.
    """
    vocab = tokenizer.vocab_size if tokenizer else 50000
    input_ids = torch.randint(0, vocab, (B, S))

    # 1. Whole-model latency
    timing = time_forward(
        model, input_ids,
        label=f"{model_name}-prefill-b{B}-s{S}",
        n_warmup=N_WARMUP,
        n_runs=N_RUNS,
    )

    # 2. Component-level timing via hooks
    #    (separate run because hooks add small per-op overhead)
    #    Warmup first
    with torch.inference_mode():
        for _ in range(N_WARMUP):
            _ = model(input_ids)

    with ComponentTimer(model) as ct:
        with torch.no_grad():
            for _ in range(N_RUNS):
                _ = model(input_ids)
    component_times_raw = ct.results()

    # Normalise: total_s across all forward passes / N_RUNS = per-pass time
    component_times_per_pass = {
        cat: stats["total_s"] / N_RUNS
        for cat, stats in component_times_raw.items()
    }
    component_shares_measured = {
        cat: stats["fraction"]
        for cat, stats in component_times_raw.items()
    }

    # 3. Predictions
    flops = flops_prefill_for_model(model_name, B, S)
    predicted_per_component = predict_component_times(model_name, B, S)
    predicted_shares = predict_component_shares(model_name, B, S)
    predicted_total = sum(predicted_per_component.values())

    # 4. Assemble result
    result = BenchmarkResult(
        model=model_name,
        batch_size=B,
        seq_len=S,
        mode="prefill",
        prefill=timing.to_dict(),
        predicted_flops=flops["total_flops"],
        predicted_latency_s=predicted_total,
        predicted_class="compute_bound",   # coarse — refined in Exp 2
        component_times_s=component_times_per_pass,
    )
    # Attach analysis extras via notes (json-serialisable)
    result.notes = json.dumps({
        "component_shares_measured": component_shares_measured,
        "component_shares_predicted": predicted_shares,
        "component_times_predicted_s": predicted_per_component,
        "raw_hook_stats": {k: dict(v) for k, v in component_times_raw.items()},
    })
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_row(model_name: str, S: int, result: BenchmarkResult) -> None:
    notes = json.loads(result.notes)
    m_share = notes["component_shares_measured"]
    p_share = notes["component_shares_predicted"]

    def pct(v): return f"{v*100:5.1f}%"

    latency_ms = result.median_latency_s * 1000
    predicted_ms = result.predicted_latency_s * 1000
    overhead_ratio = latency_ms / predicted_ms if predicted_ms else float("inf")

    print(f"\n── {model_name}, S={S}, B={result.batch_size} ─────────────────────────────")
    print(f"  Whole-model latency: {latency_ms:6.2f} ms measured  |  "
          f"{predicted_ms:6.2f} ms predicted  |  {overhead_ratio:.2f}× gap")

    # Component share comparison
    print(f"  {'Component':<12} {'Measured':>10} {'Predicted':>10} {'Δ':>8}")
    cats = ["attention", "mlp", "layernorm", "embedding", "lm_head"]
    for cat in cats:
        m = m_share.get(cat, 0.0)
        p = p_share.get(cat, 0.0)
        if m == 0 and p == 0:
            continue
        delta = (m - p) * 100
        print(f"  {cat:<12} {pct(m):>10} {pct(p):>10} {delta:>+7.1f}pp")

    # Sum of hook times vs whole-model — gap is dispatch overhead
    hook_total = sum(notes["raw_hook_stats"][c]["total_s"] / N_RUNS
                     for c in notes["raw_hook_stats"])
    dispatch_overhead_ms = (result.median_latency_s - hook_total) * 1000
    if dispatch_overhead_ms > 0:
        print(f"  Hook-sum: {hook_total*1000:6.2f} ms  |  "
              f"unaccounted (dispatch, softmax, residuals): "
              f"{dispatch_overhead_ms:6.2f} ms "
              f"({dispatch_overhead_ms / latency_ms * 100:.1f}% of total)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    set_perf_cores()

    print("=" * 70)
    print("EXPERIMENT 1 — Baseline component profile")
    print("=" * 70)

    hw = get_hardware_metadata()
    print(f"\nHardware:")
    for k, v in hw.items():
        print(f"  {k}: {v}")
    print(f"  ridge_point: {DEFAULT_HW.ridge_point:.1f} FLOP/byte")
    print(f"  peak_flops:  {DEFAULT_HW.peak_flops_fp32/1e12:.2f} TFLOP/s")
    print(f"  memory_bw:   {DEFAULT_HW.memory_bw/1e9:.0f} GB/s")

    all_results = []
    for model_name in MODELS:
        print(f"\nLoading {model_name}...")
        model = GPT2LMHeadModel.from_pretrained(model_name).eval()
        model.config.use_cache = False    # avoid dynamic cache warnings
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        for S in SEQ_LENS:
            print(f"  Measuring S={S}...")
            result = measure_one_config(model, tokenizer, model_name, S, BATCH_SIZE)
            save_result(result, OUT_DIR)
            print_summary_row(model_name, S, result)
            all_results.append(result)

        del model  # free memory before next model

    # ── Cross-configuration summary ──
    print("\n" + "=" * 70)
    print("SUMMARY — component shares across sequence lengths")
    print("=" * 70)
    print(f"\n{'Config':<20} {'attention':>10} {'mlp':>10} {'lm_head':>10} {'other':>10} {'measured/pred':>15}")
    for r in all_results:
        notes = json.loads(r.notes)
        m = notes["component_shares_measured"]
        gap = r.median_latency_s / r.predicted_latency_s if r.predicted_latency_s else 0
        other = 1.0 - sum(m.get(c, 0) for c in ["attention", "mlp", "lm_head"])
        print(
            f"{r.model} B={r.batch_size} S={r.seq_len:<7} "
            f"{m.get('attention', 0)*100:>9.1f}% "
            f"{m.get('mlp', 0)*100:>9.1f}% "
            f"{m.get('lm_head', 0)*100:>9.1f}% "
            f"{other*100:>9.1f}% "
            f"{gap:>14.2f}×"
        )

    print(f"\nResults saved to {OUT_DIR}/")
    print("Ready to plot in notebooks/02_results.ipynb")


if __name__ == "__main__":
    main()