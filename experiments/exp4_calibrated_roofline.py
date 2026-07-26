"""
experiments/exp4_calibrated_roofline.py
---------------------------------------
EXPERIMENT 4 — Calibrated roofline validation

RESEARCH QUESTION:
  Experiment 2 showed the roofline model, using a single peak_flops constant
  (2.91 TFLOP/s from a 4096² matmul), over-predicts GPT-2 throughput by
  1.4-7×. Experiment 3 showed the reason: GPT-2's matmuls don't saturate
  AMX because their M dimension (= sequence length) is often below the
  M ≈ 512 saturation knee.

  Question: if we replace the global peak with per-shape peaks measured
  in Experiment 3, does the roofline model become accurate?

HYPOTHESIS:
  Substituting per-shape peaks should collapse the roofline error factor
  from 1.4-7× down to ~1.1-1.3× — the remaining gap being non-matmul
  operations (LayerNorm, softmax, dispatch overhead) which Exp 1 measured
  at 5-15% of total time.

METHOD:
  No new measurements — this is pure re-analysis.

  1. Load Exp 3 results (per-shape achieved throughput for each GPT-2
     matmul shape at each S).
  2. Load Exp 1 results (measured whole-model and per-component times).
  3. For each component in each Exp 1 config, look up the achieved
     throughput of the actual matmul shape(s) that make up that component
     from Exp 3, and compute a shape-calibrated latency prediction.
  4. Compare naive prediction, standard-roofline prediction (Exp 2),
     and shape-calibrated prediction against measured latency.

OUTPUTS:
  - experiments/results/exp4/calibrated_analysis_{model}_s{S}.json
  - Console table: three predictions side by side, per component and total

Run:
  uv run python experiments/exp4_calibrated_roofline.py
"""

from __future__ import annotations
import json
from pathlib import Path
from miniroofline.benchmark.results import load_all_results
from miniroofline.cost_model.flops import (
    flops_prefill_for_model, GPT2_CONFIGS,
)
from miniroofline.cost_model.hardware import DEFAULT_HW


EXP1_DIR = Path("experiments/results/exp1")
EXP3_FILE = Path("experiments/results/exp3/shape_sensitivity.json")
OUT_DIR = Path("experiments/results/exp4")


# ─────────────────────────────────────────────────────────────────────────────
# Load per-shape peaks from Experiment 3
# ─────────────────────────────────────────────────────────────────────────────

def load_shape_peaks() -> dict[str, float]:
    """
    Return a mapping from GPT-2 shape label → achieved GFLOP/s.

    Labels have form "QKV_S32", "FC1_S128", "LMH_S512", etc.
    Values are converted to FLOP/s (not GFLOP/s) to match the rest of the code.
    """
    if not EXP3_FILE.exists():
        raise SystemExit(
            f"Experiment 3 results not found at {EXP3_FILE}. "
            "Run exp3_shape_sensitivity.py first."
        )

    with open(EXP3_FILE) as f:
        measurements = json.load(f)

    peaks = {}
    for m in measurements:
        if m["category"] == "gpt2":
            peaks[m["label"]] = m["achieved_gflops"] * 1e9
    return peaks


# ─────────────────────────────────────────────────────────────────────────────
# Per-component prediction using per-shape peaks
# ─────────────────────────────────────────────────────────────────────────────

def component_flops_breakdown(model_name: str, B: int, S: int) -> dict[str, dict]:
    """
    Break each hook category into individual matmul operations, with
    the FLOPs and Exp-3 shape label for each. This is the bridge between
    the coarse "attention" bucket and the fine per-matmul measurements.

    Returns:
        {
          "attention": {
            "QKV":  { "flops": ..., "shape_label": "QKV_S128", ... },
            "Out":  { "flops": ..., "shape_label": "Out_S128", ... },
            "scores_and_weighted": { "flops": ..., "shape_label": None, ... },
          },
          "mlp": { "FC1": ..., "FC2": ...},
          "lm_head": { "LMH": ... },
          "layernorm": { "layernorm": ...},
          "embedding": { "embedding": ...},
        }
    """
    cfg = GPT2_CONFIGS[model_name]
    L, d, H, d_ff = cfg["L"], cfg["d"], cfg["H"], cfg["d_ff"]
    V = cfg["vocab"]

    # Individual matmul FLOPs, summed across all L layers
    qkv_flops = L * 3 * 2 * B * S * d * d          # 3 projections, one matmul
    out_flops = L * 2 * B * S * d * d
    fc1_flops = L * 2 * B * S * d * d_ff
    fc2_flops = L * 2 * B * S * d_ff * d
    lmh_flops = 2 * B * S * d * V   # LM head — only once, not per layer

    # Attention scores + weighted sum = attention-internal matmuls
    # These have irregular shape [B*H, S, Dh] × [B*H, Dh, S] etc.
    # No Exp-3 measurement covers them, so we treat them as compute-bound
    # at the global peak (best-case).
    scores_flops = L * 2 * B * H * S * S * (d // H)
    weighted_flops = L * 2 * B * H * S * S * (d // H)
    attn_internal_flops = scores_flops + weighted_flops

    return {
        "attention": {
            "QKV":  {"flops": qkv_flops, "shape_label": f"QKV_S{S}"},
            "Out":  {"flops": out_flops, "shape_label": f"Out_S{S}"},
            "scores_and_weighted": {
                "flops": attn_internal_flops, "shape_label": None,
            },
        },
        "mlp": {
            "FC1": {"flops": fc1_flops, "shape_label": f"FC1_S{S}"},
            "FC2": {"flops": fc2_flops, "shape_label": f"FC2_S{S}"},
        },
        "lm_head": {
            "LMH": {"flops": lmh_flops, "shape_label": f"LMH_S{S}"},
        },
        # LayerNorm and embedding — no matmul, we skip the shape-calibration
        # and rely on Exp 2's overhead-sensitive treatment for these.
        "layernorm": {},
        "embedding": {},
    }


def calibrated_latency_for_component(
    breakdown: dict, peaks: dict[str, float], fallback_peak: float,
) -> tuple[float, list[dict]]:
    """
    Given a component's matmul breakdown and per-shape peaks, compute
    total latency by summing individual matmul latencies.

    Returns (total_latency_s, list_of_per_matmul_details).
    """
    total_s = 0.0
    details = []
    for matmul_name, info in breakdown.items():
        shape_label = info["shape_label"]
        flops = info["flops"]

        if shape_label and shape_label in peaks:
            peak = peaks[shape_label]
            source = f"Exp3:{shape_label}"
        else:
            # No Exp-3 measurement for this matmul (e.g. attention scores).
            # Use global peak — this is the best-case assumption.
            peak = fallback_peak
            source = "global_peak"

        latency = flops / peak if peak > 0 else 0.0
        total_s += latency
        details.append({
            "matmul": matmul_name,
            "flops": flops,
            "peak_flops": peak,
            "peak_source": source,
            "latency_s": latency,
        })
    return total_s, details


# ─────────────────────────────────────────────────────────────────────────────
# Analyse one Exp 1 config
# ─────────────────────────────────────────────────────────────────────────────

def analyse_config(exp1_result, peaks: dict[str, float]) -> dict:
    model = exp1_result.model
    B = exp1_result.batch_size
    S = exp1_result.seq_len

    breakdown_all = component_flops_breakdown(model, B, S)
    measured = exp1_result.component_times_s

    per_component = {}
    total_calibrated_s = 0.0

    for cat, matmul_breakdown in breakdown_all.items():
        m = measured.get(cat, 0.0)
        if not matmul_breakdown:
            # LayerNorm / embedding — no calibration, just record measurement
            per_component[cat] = {
                "measured_s": m,
                "calibrated_predicted_s": None,
                "matmul_details": [],
                "note": "no matmul; overhead-sensitive category",
            }
            continue

        cal_s, details = calibrated_latency_for_component(
            matmul_breakdown, peaks, DEFAULT_HW.peak_flops_fp32,
        )
        total_calibrated_s += cal_s

        per_component[cat] = {
            "measured_s": m,
            "calibrated_predicted_s": cal_s,
            "error_ratio": (m / cal_s) if cal_s > 0 else None,
            "matmul_details": details,
        }

    # Whole-model comparison
    whole_measured = exp1_result.median_latency_s
    naive_predicted = exp1_result.predicted_latency_s   # global peak

    return {
        "model": model,
        "batch_size": B,
        "seq_len": S,
        "per_component": per_component,
        "whole_measured_s": whole_measured,
        "naive_predicted_s": naive_predicted,
        "calibrated_predicted_s": total_calibrated_s,
        "naive_error_factor": whole_measured / naive_predicted if naive_predicted else float("inf"),
        "calibrated_error_factor": whole_measured / total_calibrated_s if total_calibrated_s else float("inf"),
        "improvement_over_naive": naive_predicted / total_calibrated_s if total_calibrated_s else 1.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_config_analysis(a: dict) -> None:
    print(f"\n── {a['model']}, S={a['seq_len']}, B={a['batch_size']} ─────────────────")

    print(f"\n  Component breakdown (matmul-only, calibrated peaks from Exp 3):")
    print(f"  {'Component':<12} {'Measured(ms)':>13} {'Calibrated(ms)':>15} {'m/p':>6}")
    print(f"  {'-'*12} {'-'*13} {'-'*15} {'-'*6}")
    for cat, c in a["per_component"].items():
        m_ms = c["measured_s"] * 1000
        if c["calibrated_predicted_s"] is None:
            print(f"  {cat:<12} {m_ms:>13.2f} {'(overhead)':>15} {'—':>6}")
            continue
        cal_ms = c["calibrated_predicted_s"] * 1000
        ratio = c["error_ratio"]
        ratio_str = f"{ratio:.2f}×" if ratio is not None else "—"
        print(f"  {cat:<12} {m_ms:>13.2f} {cal_ms:>15.2f} {ratio_str:>6}")

    # Per-matmul detail (attention/mlp/lm_head only)
    print(f"\n  Per-matmul detail:")
    for cat in ["attention", "mlp", "lm_head"]:
        details = a["per_component"][cat].get("matmul_details", [])
        for d in details:
            print(
                f"    {cat:<10} {d['matmul']:<20} "
                f"peak={d['peak_flops']/1e9:>6.0f} GFLOP/s "
                f"({d['peak_source']:<18}) "
                f"→ {d['latency_s']*1000:>6.2f} ms"
            )

    # Whole-model summary
    print(f"\n  Whole-model comparison:")
    print(f"    Measured                     : {a['whole_measured_s']*1000:6.2f} ms")
    print(f"    Naive (global peak)          : {a['naive_predicted_s']*1000:6.2f} ms  "
          f"→ error factor {a['naive_error_factor']:.2f}×")
    print(f"    Calibrated (per-shape peaks) : {a['calibrated_predicted_s']*1000:6.2f} ms  "
          f"→ error factor {a['calibrated_error_factor']:.2f}×")

    if a['naive_error_factor'] > 0 and a['calibrated_error_factor'] > 0:
        gap_closed = 1 - (a['calibrated_error_factor'] - 1) / (a['naive_error_factor'] - 1)
        print(f"    Fraction of naive gap explained: {gap_closed*100:.1f}%")


def print_cross_summary(analyses: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("SUMMARY — calibrated roofline vs standard roofline")
    print("=" * 70)
    print(f"\n{'Config':<24} {'Naive err':>10} {'Calib err':>11} {'Gap closed':>12}")
    print(f"{'-'*24} {'-'*10} {'-'*11} {'-'*12}")
    for a in analyses:
        cfg = f"{a['model']} B={a['batch_size']} S={a['seq_len']}"
        n_err = a["naive_error_factor"]
        c_err = a["calibrated_error_factor"]
        if n_err > 1 and c_err > 0:
            gap_closed = 1 - (c_err - 1) / (n_err - 1)
            gap_str = f"{gap_closed*100:5.1f}%"
        else:
            gap_str = "—"
        print(
            f"{cfg:<24} "
            f"{n_err:>9.2f}× "
            f"{c_err:>10.2f}× "
            f"{gap_str:>12}"
        )


def print_key_findings(analyses: list[dict]) -> None:
    print("\n" + "─" * 70)
    print("KEY FINDINGS TO LOG")
    print("─" * 70)

    naive_range = (
        min(a["naive_error_factor"] for a in analyses),
        max(a["naive_error_factor"] for a in analyses),
    )
    cal_range = (
        min(a["calibrated_error_factor"] for a in analyses),
        max(a["calibrated_error_factor"] for a in analyses),
    )
    print(f"\n  Naive roofline error   : {naive_range[0]:.2f}× to {naive_range[1]:.2f}×")
    print(f"  Calibrated roofline err: {cal_range[0]:.2f}× to {cal_range[1]:.2f}×")

    remaining_gaps_ms = []
    for a in analyses:
        gap_ms = (a["whole_measured_s"] - a["calibrated_predicted_s"]) * 1000
        remaining_gaps_ms.append((a["seq_len"], gap_ms))
    print(f"\n  Remaining gap (measured − calibrated, in ms):")
    for s, g in sorted(remaining_gaps_ms):
        print(f"    S={s:<4}: {g:>6.2f} ms")
    print(f"  This residual is attributable to non-matmul operations")
    print(f"  (LayerNorm, softmax, dispatch overhead, residual adds).")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("EXPERIMENT 4 — Calibrated roofline validation")
    print("=" * 70)

    if not EXP1_DIR.exists():
        raise SystemExit(f"Exp 1 results missing at {EXP1_DIR}.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    peaks = load_shape_peaks()
    print(f"\nLoaded {len(peaks)} per-shape peaks from Exp 3:")
    for label, peak in sorted(peaks.items()):
        print(f"  {label:<12} → {peak/1e9:>6.0f} GFLOP/s")

    exp1_results = load_all_results(EXP1_DIR)
    print(f"\nLoaded {len(exp1_results)} Exp 1 configs")

    analyses = []
    for r in exp1_results:
        a = analyse_config(r, peaks)
        analyses.append(a)
        print_config_analysis(a)

        out_path = OUT_DIR / f"calibrated_{r.model}_b{r.batch_size}_s{r.seq_len}.json"
        with open(out_path, "w") as f:
            json.dump(a, f, indent=2)

    print_cross_summary(analyses)
    print_key_findings(analyses)

    print(f"\nResults saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()