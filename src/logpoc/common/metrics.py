"""Shared metrics for every method, so numbers are comparable.

A record is a dict with:
  label: 0 normal, 1 anomalous
  anomaly_type: "none" or one of the five types
  flagged: bool, the method says this trace is anomalous
  score: float or None, a continuous anomaly score when the method has one
  pred_index: int or None, the line the method points at (0-based over template lines,
              the TRACE END marker has index = number of lines)
  first_deviation_index: int or None, from the labels file
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

LOCALISATION_TOLERANCE = 1


def _div(a: float, b: float) -> float | None:
    return a / b if b else None


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Rank based AUROC with average ranks for ties. None if one class is missing."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def localisation_hit(pred: int | None, truth: int | None) -> bool:
    if pred is None or truth is None:
        return False
    return abs(pred - truth) <= LOCALISATION_TOLERANCE


def compute_metrics(records: Sequence[dict]) -> dict:
    tp = sum(1 for r in records if r["label"] == 1 and r["flagged"])
    fn = sum(1 for r in records if r["label"] == 1 and not r["flagged"])
    fp = sum(1 for r in records if r["label"] == 0 and r["flagged"])
    tn = sum(1 for r in records if r["label"] == 0 and not r["flagged"])
    precision = _div(tp, tp + fp)
    recall = _div(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    elif precision is not None and recall is not None:
        f1 = 0.0

    per_type: dict[str, dict] = {}
    for t in sorted({r["anomaly_type"] for r in records if r["label"] == 1}):
        rows = [r for r in records if r["anomaly_type"] == t]
        hit = sum(1 for r in rows if r["flagged"])
        per_type[t] = {"n": len(rows), "flagged": hit, "recall": hit / len(rows)}

    caught = [r for r in records if r["label"] == 1 and r["flagged"]]
    loc_hits = sum(
        1 for r in caught if localisation_hit(r.get("pred_index"), r.get("first_deviation_index"))
    )

    scored = [r for r in records if r.get("score") is not None]
    auc = None
    if len(scored) == len(records) and records:
        auc = auroc([r["label"] for r in records], [r["score"] for r in records])

    return {
        "n_traces": len(records),
        "n_normal": fp + tn,
        "n_anomalous": tp + fn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": _div(fp, fp + tn),
        "recall_per_type": per_type,
        "auroc": auc,
        "localisation_hit_rate": _div(loc_hits, len(caught)),
        "localisation_hits": loc_hits,
        "localisation_n": len(caught),
    }


def percentile_threshold(scores: Sequence[float], pct: float = 99.0) -> float:
    return float(np.percentile(np.asarray(scores, dtype=float), pct))


def fmt(x: float | None, digits: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"
