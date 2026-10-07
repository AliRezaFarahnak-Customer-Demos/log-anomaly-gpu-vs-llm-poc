"""Keyword baseline: flag a trace if any line matches an error keyword."""

from __future__ import annotations

import re
import time
from pathlib import Path

from logpoc.common.report import write_eval
from logpoc.common.runs import (
    create_fresh_run,
    git_info,
    library_versions,
    mark_done,
    resolve_run_id,
    update_meta,
    utc_now,
)
from logpoc.data.prepare import data_manifest_hash, load_labels, load_sequences

KEYWORDS = re.compile(r"ERROR|FATAL|exception|timeout", re.IGNORECASE)


def first_match(lines: list[str]) -> int | None:
    for i, line in enumerate(lines):
        if KEYWORDS.search(line):
            return i
    return None


def classify(sequences: list[dict], labels: dict[str, dict]) -> list[dict]:
    records = []
    for seq in sequences:
        lab = labels[seq["trace_id"]]
        idx = first_match(seq["lines"])
        records.append(
            {
                "trace_id": seq["trace_id"],
                "label": lab["label"],
                "anomaly_type": lab["anomaly_type"],
                "first_deviation_index": lab["first_deviation_index"],
                "flagged": idx is not None,
                "score": None,
                "pred_index": idx,
                "worst_line_text": "" if idx is None else seq["lines"][idx],
            }
        )
    return records


def run(data_dir: Path, split: str, run_id: str | None = None) -> dict:
    t0 = time.time()
    rid = resolve_run_id(run_id, prefix="grep-")
    rdir = create_fresh_run(rid)
    sha, dirty = git_info()
    update_meta(
        rdir,
        run_id=rid,
        method="grep",
        git_sha=sha,
        git_dirty=dirty,
        split=split,
        data_manifest_sha256=data_manifest_hash(data_dir),
        versions=library_versions(),
        start_time=utc_now(),
    )
    records = classify(load_sequences(data_dir, split), load_labels(data_dir, split))
    wall = time.time() - t0
    payload = write_eval(
        rdir,
        split,
        method="grep",
        variant="keywords",
        records=records,
        summary={"wall_seconds": wall, "scoring_seconds": wall, "cost_per_1000": 0.0},
        title=f"Keyword baseline on {split}",
    )
    update_meta(rdir, end_time=utc_now())
    mark_done(rdir)
    return payload
