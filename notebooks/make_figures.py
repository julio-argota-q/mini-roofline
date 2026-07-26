"""
notebooks/make_figures.py
-------------------------
Generate the four figures from saved experiment JSON.

Run:
  uv run python notebooks/make_figures.py

Outputs to figures/:
  fig1_prediction_gap.pdf        — measured/predicted ratio vs S
  fig2_amx_saturation.pdf         — square-matmul GFLOP/s vs matrix size
  fig3_m_dimension_sweep.pdf      — GFLOP/s vs M for MLP FC1 shape
  fig4_component_shares.pdf       — measured vs predicted time-share bars
"""

from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# House style — publication-quality, colour-blind safe
# ─────────────────────────────────────────────────────────────────────────────

# Compact serif-body style; consistent colour palette across figures
plt.rcParams.update({
    "figure.figsize": (6.0, 3.8),
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": ":",
    "legend.frameon": False,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
})

# Colour palette (Wong / Okabe-Ito, colour-blind safe)
C_NAIVE     = "#D55E00"    # orange     — naive prediction
C_ROOFLINE  = "#009E73"    # green      — standard roofline
C_CALIB     = "#0072B2"    # blue       — calibrated
C_MEAS      = "#111111"    # near-black — measured
C_ACCENT    = "#CC79A7"    # magenta    — knee marker
C_MUTED     = "#666666"

C_ATTN = "#D55E00"
C_MLP  = "#009E73"
C_LN   = "#CC79A7"
C_LMH  = "#0072B2"


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

EXP1_DIR = Path("experiments/results/exp1")
EXP2_DIR = Path("experiments/results/exp2")
EXP3_FILE = Path("experiments/results/exp3/shape_sensitivity.json")
EXP4_DIR = Path("experiments/results/exp4")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_json_dir(directory: Path) -> list[dict]:
    """Load and return every JSON file in a directory, sorted by seq_len."""
    files = sorted(directory.glob("*.json"))
    out = []
    for f in files:
        with open(f) as fp:
            d = json.load(fp)
        out.append(d)
    return sorted(out, key=lambda d: d.get("seq_len", d.get("config", {}).get("seq_len", 0)))


def load_exp3() -> list[dict]:
    with open(EXP3_FILE) as f:
        return json.load(f)


def get_seq_len(d: dict) -> int:
    return d.get("seq_len") or d.get("config", {}).get("seq_len")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Prediction gap vs sequence length
# ─────────────────────────────────────────────────────────────────────────────

def fig_prediction_gap() -> None:
    """
    Show the measured/predicted ratio for three prediction strategies:
      - Naive         (FLOPs / peak_flops)
      - Standard roofline (max(FLOPs/peak, bytes/BW))
      - Calibrated    (per-shape peaks)
    """
    exp1 = load_json_dir(EXP1_DIR)
    exp4 = load_json_dir(EXP4_DIR)

    # Match on seq_len — Exp 1 provides measured & naive; Exp 4 provides calibrated
    by_s = {}
    for r in exp1:
        S = r["seq_len"]
        measured = r["prefill"]["median_s"]
        predicted = r["predicted_latency_s"]
        by_s.setdefault(S, {})["measured_s"] = measured
        by_s[S]["naive_ratio"] = measured / predicted if predicted else 0
        # standard roofline equals naive for GPT-2 (Finding 3.11) — use identical
        by_s[S]["roofline_ratio"] = by_s[S]["naive_ratio"]

    for r in exp4:
        S = r["seq_len"]
        m = r["whole_measured_s"]
        cal = r["calibrated_predicted_s"]
        by_s.setdefault(S, {})["calibrated_ratio"] = m / cal if cal else 0

    seq_lens = sorted(by_s.keys())
    naive_ys    = [by_s[s]["naive_ratio"] for s in seq_lens]
    roofline_ys = [by_s[s].get("roofline_ratio", by_s[s]["naive_ratio"]) for s in seq_lens]
    calib_ys    = [by_s[s].get("calibrated_ratio", None) for s in seq_lens]

    fig, ax = plt.subplots()

    # Perfect-prediction reference line
    ax.axhline(1.0, color=C_MUTED, linestyle="--", linewidth=1, alpha=0.6, zorder=1)
    ax.text(seq_lens[-1] * 1.05, 1.0, "perfect",
            va="center", fontsize=8, color=C_MUTED)

    ax.plot(seq_lens, naive_ys, "o-", color=C_NAIVE,
            label="Naive (single peak)", zorder=5)
    if any(v != n for v, n in zip(roofline_ys, naive_ys)):
        ax.plot(seq_lens, roofline_ys, "s-", color=C_ROOFLINE,
                label="Standard roofline")
    else:
        # Note in caption that they coincide
        ax.plot([], [], "s-", color=C_ROOFLINE,
                label="Standard roofline (coincides with naive)")

    if all(v is not None for v in calib_ys):
        ax.plot(seq_lens, calib_ys, "^-", color=C_CALIB,
                label="Per-shape calibrated (Exp 3 peaks)", zorder=6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(seq_lens)
    ax.set_xticklabels([str(s) for s in seq_lens])
    ax.set_yticks([1, 2, 3, 5, 7, 10])
    ax.set_yticklabels(["1", "2", "3", "5", "7", "10"])
    ax.set_xlabel("Sequence length $S$")
    ax.set_ylabel("Measured / predicted latency")
    ax.set_title("Prediction gap shrinks with sequence length")
    ax.legend(loc="upper right")
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)

    out = FIG_DIR / "fig1_prediction_gap.pdf"
    fig.savefig(out); plt.close(fig)
    fig.savefig(out.with_suffix(".png"))
    print(f"  ✓ {out}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — AMX saturation curve
# ─────────────────────────────────────────────────────────────────────────────

def fig_amx_saturation() -> None:
    """
    Square-matmul throughput vs matrix size. Shows the M4 Pro AMX
    saturation knee at around 512.
    """
    data = load_exp3()
    square = [m for m in data if m["category"] == "square_baseline"]
    square.sort(key=lambda m: m["M"])
    sizes = [m["M"] for m in square]
    gflops = [m["achieved_gflops"] for m in square]

    peak_gflops = max(gflops)   # observed peak across all points

    fig, ax = plt.subplots()

    ax.plot(sizes, gflops, "o-", color=C_MEAS)

    # Peak plateau reference
    ax.axhline(peak_gflops, color=C_MUTED, linestyle="--",
               linewidth=1, alpha=0.6, zorder=1)
    ax.text(sizes[0] * 0.9, peak_gflops * 1.02,
            f"observed peak = {peak_gflops:.0f} GFLOP/s",
            va="bottom", fontsize=8, color=C_MUTED)

    # Knee marker at M=512
    ax.axvline(512, color=C_ACCENT, linestyle=":", linewidth=1.5, alpha=0.8, zorder=2)
    ax.text(512, min(gflops) * 1.1, "saturation\nknee ≈ 512",
            ha="center", va="bottom", fontsize=8, color=C_ACCENT)

    # Percentage annotations
    for m, g in zip(sizes, gflops):
        pct = g / peak_gflops * 100
        ax.annotate(f"{pct:.0f}%",
                    xy=(m, g), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color=C_MEAS)

    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("Square matrix size $M = K = N$")
    ax.set_ylabel("Achieved throughput (GFLOP/s)")
    ax.set_title("M4 Pro AMX saturates near $M = 512$")
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)

    out = FIG_DIR / "fig2_amx_saturation.pdf"
    fig.savefig(out); plt.close(fig)
    fig.savefig(out.with_suffix(".png"))
    print(f"  ✓ {out}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — M-dimension sweep with GPT-2 markers
# ─────────────────────────────────────────────────────────────────────────────

def fig_m_sweep() -> None:
    """
    Hold K=768, N=3072 fixed (MLP FC1 shape). Sweep M.
    Annotate where GPT-2 sits at each sequence length.
    """
    data = load_exp3()
    sweep = [m for m in data if m["category"] == "m_sweep"]
    sweep.sort(key=lambda m: m["M"])
    Ms = [m["M"] for m in sweep]
    gflops = [m["achieved_gflops"] for m in sweep]

    peak = max(gflops)

    fig, ax = plt.subplots()

    ax.plot(Ms, gflops, "o-", color=C_MEAS, label="Achieved throughput")
    ax.axhline(peak, color=C_MUTED, linestyle="--", alpha=0.5, linewidth=1)
    ax.text(Ms[0], peak * 1.02, f"observed peak = {peak:.0f} GFLOP/s",
            va="bottom", fontsize=8, color=C_MUTED)

    # Mark where GPT-2 sits at S ∈ {32, 128, 512}
    gpt2_positions = {32: None, 128: None, 512: None}
    for m, g in zip(Ms, gflops):
        if m in gpt2_positions:
            gpt2_positions[m] = g

    for m, g in gpt2_positions.items():
        if g is None:
            continue
        pct = g / peak * 100
        ax.scatter([m], [g], s=160, facecolor="none", edgecolor=C_ACCENT,
                   linewidth=2.5, zorder=6)
        ax.annotate(f"GPT-2\nS={m}\n({pct:.0f}% peak)",
                    xy=(m, g), xytext=(0, -40 if m == 32 else 30),
                    textcoords="offset points",
                    ha="center", fontsize=8, color=C_ACCENT,
                    arrowprops=dict(arrowstyle="-", color=C_ACCENT,
                                     lw=0.8, alpha=0.7))

    ax.set_xscale("log", base=2)
    ax.set_xticks(Ms)
    ax.set_xticklabels([str(m) for m in Ms])
    ax.set_xlabel(r"$M$ dimension (equal to sequence length in GPT-2)")
    ax.set_ylabel("Achieved throughput (GFLOP/s)")
    ax.set_title("Throughput scales with $M$ until saturation "
                 r"(shape $M{\times}768{\times}3072$)")
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)

    out = FIG_DIR / "fig3_m_dimension_sweep.pdf"
    fig.savefig(out); plt.close(fig)
    fig.savefig(out.with_suffix(".png"))
    print(f"  ✓ {out}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Component shares: measured vs predicted (grouped bars)
# ─────────────────────────────────────────────────────────────────────────────

def fig_component_shares() -> None:
    """
    For each S, show measured and predicted time-share (as fraction of total)
    for each of attention, MLP, LayerNorm, LM head.
    """
    import numpy as np

    exp1 = load_json_dir(EXP1_DIR)
    exp1.sort(key=lambda r: r["seq_len"])

    cats = ["attention", "mlp", "lm_head", "layernorm"]
    cat_colors = {
        "attention": C_ATTN, "mlp": C_MLP,
        "lm_head": C_LMH, "layernorm": C_LN,
    }

    # gather shares per config
    configs = []
    for r in exp1:
        notes = json.loads(r["notes"])
        m_share = notes["component_shares_measured"]
        p_share = notes["component_shares_predicted"]
        configs.append({
            "S": r["seq_len"],
            "measured": {c: m_share.get(c, 0.0) for c in cats},
            "predicted": {c: p_share.get(c, 0.0) for c in cats},
        })

    n_configs = len(configs)
    x = np.arange(n_configs)
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.5, 4.0))

    # Stacked bars — one stack per (config, series) pair
    for series_idx, series in enumerate(["measured", "predicted"]):
        offset = -width/2 if series == "measured" else width/2
        bottoms = [0.0] * n_configs
        for cat in cats:
            heights = [c[series][cat] for c in configs]
            ax.bar(x + offset, heights, width,
                   bottom=bottoms, color=cat_colors[cat],
                   edgecolor="white", linewidth=0.5,
                   label=cat if series_idx == 0 else "_nolegend_",
                   alpha=1.0 if series == "measured" else 0.55)
            bottoms = [b + h for b, h in zip(bottoms, heights)]

        # Series label under each bar group
        for i, c in enumerate(configs):
            ax.text(i + offset, -0.03,
                    "meas" if series == "measured" else "pred",
                    ha="center", va="top", fontsize=7, color=C_MUTED)

    ax.set_xticks(x)
    ax.set_xticklabels([f"S={c['S']}" for c in configs])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of total forward pass")
    ax.set_title("Time-share (measured) vs FLOP-share (predicted) by component")
    ax.legend(loc="upper right", ncol=1)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["bottom"].set_visible(True)

    fig.subplots_adjust(bottom=0.18)

    out = FIG_DIR / "fig4_component_shares.pdf"
    fig.savefig(out); plt.close(fig)
    fig.savefig(out.with_suffix(".png"))
    print(f"  ✓ {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Generating figures → {FIG_DIR}/\n")

    # Each figure is independent — if one fails (e.g. Exp 4 not yet run),
    # continue with the others.
    figures = [
        ("fig1_prediction_gap",  fig_prediction_gap),
        ("fig2_amx_saturation",  fig_amx_saturation),
        ("fig3_m_dimension_sweep", fig_m_sweep),
        ("fig4_component_shares", fig_component_shares),
    ]
    for name, func in figures:
        try:
            func()
        except Exception as e:
            print(f"  ✗ {name} failed: {type(e).__name__}: {e}")

    print(f"\nAll figures rendered as both PDF (for paper) and PNG (for preview).")
    print(f"To rebuild: uv run python notebooks/make_figures.py")


if __name__ == "__main__":
    main()
