"""
experiments/exp2_validation.py
------------------------------
EXPERIMENT 2 — Per-component roofline validation

RESEARCH QUESTION:
  Experiment 1 showed a systematic gap between whole-model latency and
  the peak-throughput prediction (7.9× at S=32, 1.7× at S=512).
  How much of that gap does the roofline model close when we predict
  per-component instead of assuming everything runs at peak?

  Concretely: for each component (attention, MLP, LayerNorm, LM head),
  compute its arithmetic intensity, classify it, and predict its latency
  using the appropriate roofline bound (compute-bound → FLOPs/peak;
  memory-bound → traffic/BW). Compare to Experiment 1's measured times.

HYPOTHESIS:
  1. LM head is compute-bound (large matmul, high AI) → roofline prediction
     will match measured to within ~30%.
  2. MLP is at the ridge (moderate AI) → roofline prediction improves over
     peak-only prediction but still under by 1.5-2×.
  3. Attention is compute-bound aggregated, but its internal ops mix regimes
     — expect similar accuracy to MLP.
  4. LayerNorm is memory-bound (AI ~0.4 FLOP/byte on M4 Pro) → memory-bound
     prediction will be closer to measured, but still under by 2-5× because
     of PyTorch dispatch overhead (LayerNorm gets called 2*L=24 times per pass).
  5. Aggregate roofline prediction (sum of per-component roofline bounds)
     will be within 2× of measured at S=512, better than the peak-only
     prediction of 1.7×.

METHOD:
  Reuse the measured component times from Exp 1's JSON output (no re-run needed).
  For each config, compute per-component:
    - arithmetic intensity I = FLOPs / bytes_moved
    - classification (compute / memory / overhead)
    - roofline predicted latency = FLOPs / min(peak_flops, BW * I)
    - error vs measured
  Produce a table per config and a summary across S.

OUTPUTS:
  - experiments/results/exp2/roofline_analysis_{model}_s{S}.json
  - Console table: component | AI | class | measured ms | predicted ms | error

Run:
  uv run python experiments/exp2_validation.py
"""

import json
from pathlib import Path
from miniroofline.benchmark.results import load_all_results
from miniroofline.cost_model.flops import (
    flops_prefill, flops_prefill_for_model, GPT2_CONFIGS,
)
from miniroofline.cost_model.memory import (
    traffic_attention_layer, traffic_mlp_layer, bytes_model_weights,
    DTYPE_BYTES,
)
from miniroofline.cost_model.hardware import DEFAULT_HW


EXP1_DIR = Path("experiments/results/exp1")
OUT_DIR = Path("experiments/results/exp2")
OVERHEAD_THRESHOLD = 3.0    # measured/predicted > 3× → overhead-sensitive


# ─────────────────────────────────────────────────────────────────────────────
# Per-component traffic (bytes read + written for one forward pass)
# ─────────────────────────────────────────────────────────────────────────────

def component_traffic_bytes(model_name: str, B: int, S: int) -> dict[str, int]:
    """
    Compute total memory traffic (bytes read + written) per component
    across the whole forward pass. Used to compute arithmetic intensity.

    Simplifications:
      - fp32 throughout
      - ignores L2 cache reuse (worst-case DRAM traffic)
      - ignores residual / dropout / gelu traffic (small)
    """
    cfg = GPT2_CONFIGS[model_name]
    L, d, H, d_ff = cfg["L"], cfg["d"], cfg["H"], cfg["d_ff"]
    V = cfg["vocab"]
    nb = 4   # float32

    attn_1layer = traffic_attention_layer(B, S, d, H)
    mlp_1layer  = traffic_mlp_layer(B, S, d, d_ff)

    # LayerNorm traffic: read input, read gamma+beta, write output
    #   ≈ 3 * B * S * d * nb per LayerNorm call, 2 per layer
    ln_traffic = 3 * B * S * d * nb * 2 * L

    # LM head traffic: read [B,S,d] activations + read [d,V] weight + write [B,S,V]
    lm_head_traffic = (B * S * d + d * V + B * S * V) * nb

    # Embedding: read token_emb slice (B*S*d) + pos_emb (S*d) + write [B,S,d]
    emb_traffic = (B * S * d + S * d + B * S * d) * nb

    return {
        "attention": attn_1layer["total_traffic_bytes"] * L,
        "mlp":       mlp_1layer["total_traffic_bytes"] * L,
        "layernorm": ln_traffic,
        "lm_head":   lm_head_traffic,
        "embedding": emb_traffic,
    }


def component_flops(model_name: str, B: int, S: int) -> dict[str, int]:
    """Per-component FLOPs, matching the categories the hooks report."""
    flops = flops_prefill_for_model(model_name, B, S)
    return {
        "attention": flops["attention_total_flops"],
        "mlp":       flops["mlp_flops"],
        "layernorm": flops["layernorm_flops"],
        "lm_head":   flops["lm_head_flops"],
        # embedding is close to zero FLOPs — mostly a lookup + one add
        "embedding": 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Roofline classification for one component
# ─────────────────────────────────────────────────────────────────────────────

def classify_component(
    flops: int,
    traffic_bytes: int,
    measured_s: float,
) -> dict:
    """
    Compute arithmetic intensity, roofline prediction, and classification
    for a single component.

    Roofline bound:
        perf(I) = min(P_peak, BW * I)
        predicted_time = FLOPs / perf(I)

    Overhead-sensitive rule:
        if measured_s > OVERHEAD_THRESHOLD * predicted_s
        AND the component has near-zero FLOPs (embedding)
        THEN classify as overhead-sensitive.
    """
    peak = DEFAULT_HW.peak_flops_fp32
    bw   = DEFAULT_HW.memory_bw
    ridge = DEFAULT_HW.ridge_point

    if traffic_bytes == 0:
        # Purely FLOP-free — treat as overhead
        intensity = 0.0
        predicted_s = 0.0
        base_class = "overhead_sensitive"
    else:
        intensity = flops / traffic_bytes if flops > 0 else 0.0
        # Bytes-limited time (memory-bound floor)
        time_from_bw = traffic_bytes / bw
        # Compute-limited time
        time_from_compute = flops / peak if flops > 0 else 0.0
        predicted_s = max(time_from_compute, time_from_bw)
        base_class = "compute_bound" if intensity >= ridge else "memory_bound"

    # Overhead correction
    ratio = measured_s / predicted_s if predicted_s > 0 else float("inf")
    if predicted_s == 0 or ratio > OVERHEAD_THRESHOLD:
        classification = "overhead_sensitive"
    else:
        classification = base_class

    error_pct = ((measured_s - predicted_s) / predicted_s * 100) if predicted_s > 0 else None

    return {
        "flops": flops,
        "traffic_bytes": traffic_bytes,
        "arithmetic_intensity": intensity,
        "predicted_time_s": predicted_s,
        "measured_time_s": measured_s,
        "classification": classification,
        "measured_over_predicted": ratio,
        "prediction_error_pct": error_pct,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Analyse one config
# ─────────────────────────────────────────────────────────────────────────────

def analyse_config(result) -> dict:
    """
    Take a BenchmarkResult from Exp 1, produce per-component roofline analysis.
    """
    model = result.model
    B = result.batch_size
    S = result.seq_len

    flops_by_cat = component_flops(model, B, S)
    traffic_by_cat = component_traffic_bytes(model, B, S)
    measured_by_cat = result.component_times_s

    per_component = {}
    for cat in ["attention", "mlp", "layernorm", "embedding", "lm_head"]:
        m = measured_by_cat.get(cat, 0.0)
        if m == 0:
            continue
        per_component[cat] = classify_component(
            flops=flops_by_cat.get(cat, 0),
            traffic_bytes=traffic_by_cat.get(cat, 0),
            measured_s=m,
        )

    # Aggregate: sum of per-component predictions
    sum_predicted = sum(c["predicted_time_s"] for c in per_component.values())
    sum_measured  = sum(c["measured_time_s"] for c in per_component.values())
    whole_measured = result.median_latency_s
    naive_predicted = result.predicted_latency_s   # from Exp 1's simple divide

    return {
        "model": model,
        "batch_size": B,
        "seq_len": S,
        "per_component": per_component,
        "sum_predicted_s": sum_predicted,
        "sum_measured_s": sum_measured,
        "whole_measured_s": whole_measured,
        "naive_predicted_s": naive_predicted,
        # Two ratios of interest:
        #   how much better is roofline than naive?
        "roofline_improvement": naive_predicted / sum_predicted if sum_predicted else 1.0,
        #   how close is roofline to reality?
        "roofline_error": whole_measured / sum_predicted if sum_predicted else float("inf"),
        "naive_error": whole_measured / naive_predicted if naive_predicted else float("inf"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

CLASS_SYMBOL = {
    "compute_bound": "⚙",
    "memory_bound": "▤",
    "overhead_sensitive": "◈",
}


def print_config_table(analysis: dict) -> None:
    print(f"\n── {analysis['model']}, S={analysis['seq_len']}, B={analysis['batch_size']} ─────────")
    print(f"  Ridge point: {DEFAULT_HW.ridge_point:.1f} FLOP/byte")
    print()
    print(f"  {'Component':<12} {'AI':>7} {'Class':<18}  "
          f"{'Meas.(ms)':>10} {'Pred.(ms)':>10} {'m/p':>6}")
    print(f"  {'-'*12} {'-'*7} {'-'*18}  {'-'*10} {'-'*10} {'-'*6}")

    for cat, c in analysis["per_component"].items():
        sym = CLASS_SYMBOL.get(c["classification"], " ")
        ai = c["arithmetic_intensity"]
        ai_str = f"{ai:7.1f}" if ai > 0 else "     — "
        print(
            f"  {cat:<12} {ai_str} {sym} {c['classification']:<16}  "
            f"{c['measured_time_s']*1000:>10.2f} "
            f"{c['predicted_time_s']*1000:>10.2f} "
            f"{c['measured_over_predicted']:>5.2f}×"
        )

    print()
    print(f"  Whole-model measured        : {analysis['whole_measured_s']*1000:6.2f} ms")
    print(f"  Sum of component roofline   : {analysis['sum_predicted_s']*1000:6.2f} ms")
    print(f"  Naive (Exp 1) peak-only     : {analysis['naive_predicted_s']*1000:6.2f} ms")
    print(f"  Roofline error factor       : {analysis['roofline_error']:.2f}×")
    print(f"  Naive prediction error      : {analysis['naive_error']:.2f}×")
    print(f"  Roofline improvement        : {analysis['roofline_improvement']:.2f}×")


def print_cross_summary(analyses: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("SUMMARY — roofline prediction vs naive prediction")
    print("=" * 70)
    print(f"\n{'Config':<24} {'Naive gap':>10} {'Roofline gap':>13} {'Improvement':>12}")
    print(f"{'-'*24} {'-'*10} {'-'*13} {'-'*12}")
    for a in analyses:
        cfg = f"{a['model']} B={a['batch_size']} S={a['seq_len']}"
        print(
            f"{cfg:<24} "
            f"{a['naive_error']:>9.2f}× "
            f"{a['roofline_error']:>12.2f}× "
            f"{a['roofline_improvement']:>11.2f}×"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not EXP1_DIR.exists():
        raise SystemExit(
            f"No Experiment 1 results found in {EXP1_DIR}. "
            "Run experiments/exp1_baseline.py first."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT 2 — Per-component roofline validation")
    print("=" * 70)
    print(f"\nHardware constants (from hardware.py):")
    print(f"  peak_flops = {DEFAULT_HW.peak_flops_fp32/1e12:.2f} TFLOP/s")
    print(f"  memory_bw  = {DEFAULT_HW.memory_bw/1e9:.0f} GB/s")
    print(f"  ridge_pt   = {DEFAULT_HW.ridge_point:.1f} FLOP/byte")
    print(f"\nLoading Exp 1 results from {EXP1_DIR}")

    exp1_results = load_all_results(EXP1_DIR)
    if not exp1_results:
        raise SystemExit(f"No JSON files in {EXP1_DIR}")

    analyses = []
    for r in exp1_results:
        a = analyse_config(r)
        analyses.append(a)
        print_config_table(a)

        # Save analysis
        out_path = OUT_DIR / f"roofline_analysis_{r.model}_b{r.batch_size}_s{r.seq_len}.json"
        with open(out_path, "w") as f:
            # per_component contains nested dicts — dump as-is (already json-safe)
            json.dump(a, f, indent=2)

    print_cross_summary(analyses)

    print(f"\nResults saved to {OUT_DIR}/")

    # ── Key findings preview ──────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("KEY FINDINGS TO LOG")
    print("─" * 70)

    # 1. Which components are most overhead-sensitive?
    overhead_cats = set()
    for a in analyses:
        for cat, c in a["per_component"].items():
            if c["classification"] == "overhead_sensitive":
                overhead_cats.add(cat)
    if overhead_cats:
        print(f"\n  Components classified overhead-sensitive: {', '.join(sorted(overhead_cats))}")

    # 2. How much did roofline improve over naive?
    improvements = [a["roofline_improvement"] for a in analyses]
    print(f"  Roofline improvement over naive prediction: "
          f"{min(improvements):.2f}× to {max(improvements):.2f}× across configs")

    # 3. Remaining gap
    final_gaps = [a["roofline_error"] for a in analyses]
    print(f"  Remaining gap (roofline → measured): "
          f"{min(final_gaps):.2f}× to {max(final_gaps):.2f}×")
    print("  The remaining gap is the paper's central open question.")


if __name__ == "__main__":
    main()