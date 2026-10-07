"""Cost per 1,000 traces from prices in configs/pricing.yaml. Prices are never invented."""

from __future__ import annotations

from pathlib import Path

import yaml

NA = "n/a (set prices in configs/pricing.yaml)"
PRICE_KEYS = ["llm_input_per_1m_tokens", "llm_output_per_1m_tokens", "gpu_per_hour"]


def load_pricing(path: str | Path | None = None) -> dict:
    p = Path(path) if path else Path("configs/pricing.yaml")
    if not p.exists():
        return dict.fromkeys(PRICE_KEYS)
    data = yaml.safe_load(p.read_text()) or {}
    return {k: data.get(k) for k in PRICE_KEYS}


def llm_cost_per_1000(
    input_tokens: int, output_tokens: int, n_traces: int, pricing: dict
) -> float | None:
    pin, pout = pricing.get("llm_input_per_1m_tokens"), pricing.get("llm_output_per_1m_tokens")
    if pin is None or pout is None or n_traces <= 0:
        return None
    total = input_tokens / 1e6 * pin + output_tokens / 1e6 * pout
    return total / n_traces * 1000


def gpu_cost_per_1000(scoring_seconds: float, n_traces: int, pricing: dict) -> float | None:
    rate = pricing.get("gpu_per_hour")
    if rate is None or n_traces <= 0:
        return None
    return scoring_seconds / 3600.0 * rate / n_traces * 1000


def gpu_training_cost(training_seconds: float, pricing: dict) -> float | None:
    rate = pricing.get("gpu_per_hour")
    return None if rate is None else training_seconds / 3600.0 * rate


def fmt_cost(x: float | None) -> str:
    return NA if x is None else f"{x:.4f}"
