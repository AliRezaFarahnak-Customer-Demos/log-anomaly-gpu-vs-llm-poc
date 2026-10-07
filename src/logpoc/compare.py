"""Write runs/RUNS.md: one row per evaluated run, costs recomputed from one pricing file."""

from __future__ import annotations

import json

from logpoc.common.metrics import fmt
from logpoc.common.pricing import (
    gpu_cost_per_1000,
    gpu_training_cost,
    llm_cost_per_1000,
    load_pricing,
)
from logpoc.common.runs import ml_root

TYPES = ["visible_error", "silent_skip", "wrong_order", "truncated", "retry_storm"]


def _cost(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def costs(d: dict, pricing: dict) -> tuple[float | None, float | None]:
    """(cost per 1,000 traces, one-off training cost) from the usage stored in metrics.json."""
    n = d["metrics"]["n_traces"]
    if d["method"] == "poc2-llm":
        tokens_in, tokens_out = d.get("input_tokens", 0), d.get("output_tokens", 0)
        return llm_cost_per_1000(tokens_in, tokens_out, n, pricing), 0.0
    if d["method"] == "poc1-gpu":
        per_1000 = gpu_cost_per_1000(d.get("scoring_seconds") or 0, n, pricing)
        return per_1000, gpu_training_cost(d.get("training_seconds") or 0, pricing)
    return d.get("cost_per_1000"), 0.0


def run(pricing_path: str | None = None) -> int:
    root = ml_root() / "runs"
    pricing = load_pricing(pricing_path)
    rows = [
        "| run | method | variant | traces | precision | recall | F1 | FPR "
        "| cost per 1,000 traces | training cost |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    per_type = ["| run | " + " | ".join(TYPES) + " |", "|---|" + "---|" * len(TYPES)]
    for path in sorted(root.glob("*/eval/metrics.json")):
        d = json.loads(path.read_text())
        m = d["metrics"]
        per_1000, training = costs(d, pricing)
        rows.append(
            f"| {d['run_id']} | {d['method']} | {d['variant']} | {m['n_traces']} "
            f"| {fmt(m['precision'])} | {fmt(m['recall'])} | {fmt(m['f1'])} "
            f"| {fmt(m['false_positive_rate'])} | {_cost(per_1000)} | {_cost(training)} |"
        )
        rec = m["recall_per_type"]
        per_type.append(
            f"| {d['run_id']} | "
            + " | ".join(fmt(rec[t]["recall"]) if t in rec else "-" for t in TYPES)
            + " |"
        )
    text = "\n".join(["# Runs", "", *rows, "", "## Recall per anomaly type", "", *per_type, ""])
    root.mkdir(parents=True, exist_ok=True)
    (root / "RUNS.md").write_text(text)
    print(text)
    return 0
