"""
hooks.py
--------
PyTorch forward hooks for component-level timing.

The goal: break a model's forward pass into named buckets
(attention, MLP, layernorm, embedding, lm_head) and time each
separately. This is what feeds the roofline scatter plot — each
component becomes a point on the chart.

How it works:
  - We register pre-hooks (start timer) and forward hooks (stop timer)
    on every module of interest.
  - For each forward pass, hooks accumulate elapsed time per module.
  - Multiple forward passes? Times accumulate; reset() between runs.

Limitations to know about:
  - Hook overhead is non-zero (~5-50 microseconds per hook). For very
    small modules (LayerNorm at S=32), this can be 10% of the measured
    time. Hooks expose framework overhead — useful for the
    "overhead-sensitive" classification, distorting for pure compute timing.
  - time.perf_counter() resolution is ~1 microsecond on macOS, fine for
    this purpose.
  - On CPU, all work is synchronous so timer stop-start gives true
    wall-clock time. On GPU we would need cuda.synchronize().

For GPT-2 specifically:
  HuggingFace's GPT2Model exposes named submodules we can target:
    transformer.wte                  → token embedding
    transformer.wpe                  → position embedding
    transformer.h[i].attn            → attention block
    transformer.h[i].mlp             → MLP block
    transformer.h[i].ln_1, ln_2      → layer norms
    lm_head                          → LM head

Reference: PyTorch hooks documentation
  https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
"""

from __future__ import annotations
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Component classifier — map a module to a logical category
# ---------------------------------------------------------------------------

def classify_gpt2_module(name: str, module: nn.Module) -> str | None:
    """
    Map a HuggingFace GPT-2 submodule name to a category.
    Return None to skip (we don't time every Linear separately).

    Categories:
      embedding   : token + position embeddings
      attention   : self-attention block (incl. QKV, output proj)
      mlp         : feed-forward block
      layernorm   : pre-attention and pre-MLP layer norms
      lm_head     : final language-model head
    """
    # Skip empty or container modules at the top level
    if name == "":
        return None
    if name in {"transformer", "transformer.h"}:
        return None

    # Embeddings
    if name in {"transformer.wte", "transformer.wpe"}:
        return "embedding"

    # Per-layer blocks — only register on the block itself, not its sublayers,
    # to avoid double-counting
    parts = name.split(".")
    if len(parts) == 3 and parts[0] == "transformer" and parts[1] == "h":
        # e.g. "transformer.h.0" — the whole block, skip
        return None
    if len(parts) == 4 and parts[0] == "transformer" and parts[1] == "h":
        # e.g. "transformer.h.0.attn", "transformer.h.0.mlp", "transformer.h.0.ln_1"
        leaf = parts[3]
        if leaf == "attn":
            return "attention"
        if leaf == "mlp":
            return "mlp"
        if leaf in {"ln_1", "ln_2"}:
            return "layernorm"

    # Final layer norm + LM head
    if name == "transformer.ln_f":
        return "layernorm"
    if name == "lm_head":
        return "lm_head"

    return None  # skip everything else


# ---------------------------------------------------------------------------
# Component timer — accumulates per-category times across forward passes
# ---------------------------------------------------------------------------

@dataclass
class ComponentTimer:
    """
    Accumulates timing data per category across forward passes.

    Usage:
        timer = ComponentTimer(model, classify_gpt2_module)
        timer.attach()
        # ... run forward pass(es) ...
        timer.detach()
        result = timer.results()
    """
    model: nn.Module
    classifier: Callable[[str, nn.Module], str | None] = classify_gpt2_module
    # accumulated times per category (across all forward calls + all modules)
    times_s: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    # per-module start times (transient; populated by pre-hook)
    _start_times: dict[int, float] = field(default_factory=dict)
    _handles: list = field(default_factory=list)
    _modules: dict[int, str] = field(default_factory=dict)  # id -> category

    def attach(self) -> None:
        """Register hooks on every classifiable module."""
        for name, module in self.model.named_modules():
            category = self.classifier(name, module)
            if category is None:
                continue
            self._modules[id(module)] = category

            pre_handle = module.register_forward_pre_hook(self._pre_hook)
            post_handle = module.register_forward_hook(self._post_hook)
            self._handles.append(pre_handle)
            self._handles.append(post_handle)

    def detach(self) -> None:
        """Remove all hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def reset(self) -> None:
        """Clear accumulated times (call before each new timing run)."""
        self.times_s.clear()
        self._start_times.clear()

    def _pre_hook(self, module, inputs):
        self._start_times[id(module)] = time.perf_counter()

    def _post_hook(self, module, inputs, output):
        end = time.perf_counter()
        start = self._start_times.pop(id(module), None)
        if start is None:
            return
        category = self._modules.get(id(module))
        if category:
            self.times_s[category].append(end - start)

    def results(self) -> dict[str, dict]:
        """
        Aggregate per-category statistics.
        Returns: {category: {total_s, mean_s, n_calls, fraction}}
        """
        total_all = sum(sum(v) for v in self.times_s.values()) or 1e-12
        results = {}
        for cat, times in self.times_s.items():
            total = sum(times)
            results[cat] = {
                "total_s": total,
                "mean_s": total / len(times) if times else 0,
                "n_calls": len(times),
                "fraction": total / total_all,
            }
        return results

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.detach()


def print_component_table(results: dict[str, dict]) -> None:
    """Print a formatted component timing table."""
    print(f"{'Component':<14} {'Total (ms)':>11} {'Mean (ms)':>11} {'Calls':>7} {'Share':>7}")
    print("-" * 55)
    # sort by total time descending
    items = sorted(results.items(), key=lambda kv: -kv[1]["total_s"])
    for cat, stats in items:
        print(
            f"{cat:<14} "
            f"{stats['total_s']*1000:>11.2f} "
            f"{stats['mean_s']*1000:>11.3f} "
            f"{stats['n_calls']:>7d} "
            f"{stats['fraction']*100:>6.1f}%"
        )


if __name__ == "__main__":
    # End-to-end test on GPT-2 (requires transformers installed)
    try:
        from transformers import GPT2LMHeadModel
    except ImportError:
        print("transformers not installed — skipping test")
        raise SystemExit(0)

    from miniroofline.benchmark.timing import set_perf_cores
    set_perf_cores()

    print("Loading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    input_ids = torch.randint(0, 50000, (1, 128))

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_ids)

    # Measured
    with ComponentTimer(model) as timer:
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_ids)

    print("\nComponent timing for GPT-2 (B=1, S=128, 10 forward passes):")
    print_component_table(timer.results())

    print("\nExpected pattern:")
    print("  MLP > attention > layernorm/embedding/lm_head")
    print("  At S=128, MLP should be ~50-65% of total time")
