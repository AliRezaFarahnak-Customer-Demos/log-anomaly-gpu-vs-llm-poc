"""Write eval/metrics.json, eval/scores.csv and eval/report.md for any method."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

from logpoc.common.metrics import compute_metrics, fmt
from logpoc.common.pricing import fmt_cost
from logpoc.common.runs import git_info, read_meta, utc_now, write_json

SCORE_COLUMNS = [
    "trace_id",
    "label",
    "anomaly_type",
    "score",
    "flagged",
    "worst_line_index",
    "worst_line_text",
]


def eval_dir_name(split: str) -> str:
    """The test split goes to eval/, other splits to eval-<split>/."""
    return "eval" if split == "test" else f"eval-{split}"


def _histogram(records: Sequence[dict], path: Path, threshold: float | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    normal = [r["score"] for r in records if r["label"] == 0 and r["score"] is not None]
    anomalous = [r["score"] for r in records if r["label"] == 1 and r["score"] is not None]
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = 40
    ax.hist(normal, bins=bins, alpha=0.6, label="normal")
    ax.hist(anomalous, bins=bins, alpha=0.6, label="anomalous")
    if threshold is not None:
        ax.axvline(threshold, color="k", linestyle="--", label="threshold")
    ax.set_xlabel("trace score")
    ax.set_ylabel("traces")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_eval(
    rdir: Path,
    split: str,
    method: str,
    variant: str,
    records: list[dict],
    summary: dict | None = None,
    threshold: float | None = None,
    title: str | None = None,
) -> dict:
    """records: dicts for compute_metrics plus trace_id and worst_line_text."""
    rdir = Path(rdir)
    out = rdir / eval_dir_name(split)
    out.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(records)
    sha, dirty = git_info()
    meta = read_meta(rdir)
    payload = {
        "method": method,
        "run_id": rdir.name,
        "variant": variant,
        "split": split,
        "git_sha": meta.get("git_sha", sha),
        "git_dirty": meta.get("git_dirty", dirty),
        "created": utc_now(),
        "threshold": threshold,
        "metrics": metrics,
    }
    payload.update(summary or {})
    write_json(out / "metrics.json", payload)

    with (out / "scores.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SCORE_COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in records:
            w.writerow(
                {
                    "trace_id": r["trace_id"],
                    "label": r["label"],
                    "anomaly_type": r["anomaly_type"],
                    "score": "" if r.get("score") is None else f"{r['score']:.6f}",
                    "flagged": int(bool(r["flagged"])),
                    "worst_line_index": "" if r.get("pred_index") is None else r["pred_index"],
                    "worst_line_text": r.get("worst_line_text", ""),
                }
            )

    has_scores = any(r.get("score") is not None for r in records)
    if has_scores:
        _histogram(records, out / "scores.png", threshold)
    (out / "report.md").write_text(
        render_report(payload, records, title or f"{method} on {split}", has_scores)
    )
    return payload


def render_report(payload: dict, records: list[dict], title: str, has_plot: bool) -> str:
    m = payload["metrics"]
    lines = [
        f"# {title}",
        "",
        f"Run `{payload['run_id']}`, variant `{payload['variant']}`, split `{payload['split']}`, "
        f"git `{payload['git_sha']}`{' (dirty)' if payload.get('git_dirty') else ''}.",
        "",
        "This is a pipeline check on tiny synthetic data, not an accuracy study.",
        "",
        "| metric | value |",
        "|---|---|",
        f"| traces | {m['n_traces']} ({m['n_normal']} normal, {m['n_anomalous']} anomalous) |",
        f"| precision | {fmt(m['precision'])} |",
        f"| recall | {fmt(m['recall'])} |",
        f"| F1 | {fmt(m['f1'])} |",
        f"| false positive rate | {fmt(m['false_positive_rate'])} |",
        f"| AUROC | {fmt(m['auroc'])} |",
        f"| localisation hit rate | {fmt(m['localisation_hit_rate'])} "
        f"({m['localisation_hits']} of {m['localisation_n']}) |",
    ]
    if payload.get("threshold") is not None:
        lines.append(f"| threshold | {payload['threshold']:.4f} |")
    for key, label in [
        ("scoring_seconds", "scoring seconds"),
        ("wall_seconds", "wall seconds"),
    ]:
        if payload.get(key) is not None:
            lines.append(f"| {label} | {payload[key]:.2f} |")
    if "cost_per_1000" in payload:
        lines.append(f"| cost per 1,000 traces | {fmt_cost(payload['cost_per_1000'])} |")
    lines += [
        "",
        "## Recall per anomaly type",
        "",
        "| type | flagged | traces | recall |",
        "|---|---|---|---|",
    ]
    for t, d in m["recall_per_type"].items():
        lines.append(f"| {t} | {d['flagged']} | {d['n']} | {d['recall']:.3f} |")

    flagged = [r for r in records if r["flagged"] and r["label"] == 1][:5]
    lines += ["", "## Example flagged anomalous traces", ""]
    if not flagged:
        lines.append("None.")
    for r in flagged:
        score = "" if r.get("score") is None else f", score {r['score']:.3f}"
        lines.append(
            f"- `{r['trace_id']}` ({r['anomaly_type']}{score}): line {r.get('pred_index')}, "
            f"`{r.get('worst_line_text', '')}`"
        )
    if has_plot:
        lines += ["", "![score histogram](scores.png)"]
    return "\n".join(lines) + "\n"
