from __future__ import annotations

import pytest

from logpoc.common import pricing
from logpoc.common.metrics import (
    auroc,
    compute_metrics,
    localisation_hit,
    percentile_threshold,
)


def rec(label, flagged, atype="none", score=None, pred=None, truth=None):
    return {
        "label": label,
        "flagged": flagged,
        "anomaly_type": atype,
        "score": score,
        "pred_index": pred,
        "first_deviation_index": truth,
    }


def test_basic_counts_and_rates():
    records = [
        rec(1, True, "a", pred=3, truth=3),
        rec(1, True, "a", pred=5, truth=3),
        rec(1, False, "b"),
        rec(1, False, "b"),
        rec(0, True),
        rec(0, False),
        rec(0, False),
        rec(0, False),
    ]
    m = compute_metrics(records)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 2, 3)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))
    assert m["false_positive_rate"] == pytest.approx(0.25)
    assert m["recall_per_type"]["a"]["recall"] == 1.0
    assert m["recall_per_type"]["b"]["recall"] == 0.0
    assert m["auroc"] is None
    assert m["localisation_hit_rate"] == pytest.approx(0.5)


def test_localisation_tolerance():
    assert localisation_hit(4, 3)
    assert localisation_hit(2, 3)
    assert not localisation_hit(5, 3)
    assert not localisation_hit(None, 3)
    assert not localisation_hit(3, None)


def test_localisation_only_counts_correctly_flagged():
    records = [rec(1, False, "a", pred=3, truth=3), rec(1, True, "a", pred=3, truth=3)]
    m = compute_metrics(records)
    assert m["localisation_n"] == 1
    assert m["localisation_hit_rate"] == 1.0


def test_empty_classes_give_none():
    m = compute_metrics([rec(0, False), rec(0, False)])
    assert m["precision"] is None and m["recall"] is None and m["f1"] is None


def test_auroc_perfect_inverse_and_ties():
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert auroc([0, 1], [0.5, 0.5]) == 0.5
    assert auroc([1, 1], [0.1, 0.2]) is None


def test_auroc_in_compute_metrics_when_scores_exist():
    records = [rec(0, False, score=0.1), rec(1, True, "a", score=0.9, pred=1, truth=1)]
    assert compute_metrics(records)["auroc"] == 1.0


def test_percentile_threshold():
    assert percentile_threshold(list(range(101)), 99) == pytest.approx(99.0)


def test_pricing_null_gives_none():
    p = pricing.load_pricing("configs/pricing.yaml")
    assert all(v is None for v in p.values())
    assert pricing.llm_cost_per_1000(1000, 100, 10, p) is None
    assert pricing.gpu_cost_per_1000(10, 10, p) is None
    assert pricing.fmt_cost(None).startswith("n/a (set prices")


def test_pricing_math():
    p = {"llm_input_per_1m_tokens": 2.0, "llm_output_per_1m_tokens": 8.0, "gpu_per_hour": 3.6}
    assert pricing.llm_cost_per_1000(1_000_000, 500_000, 1000, p) == pytest.approx(6.0)
    assert pricing.gpu_cost_per_1000(360, 100, p) == pytest.approx(3.6 / 3600 * 360 / 100 * 1000)
    assert pricing.gpu_training_cost(3600, p) == pytest.approx(3.6)
