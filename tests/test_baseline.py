from __future__ import annotations

from logpoc.baselines.grep_rules import classify, run
from logpoc.common.runs import is_done, run_dir
from logpoc.data.prepare import load_labels, load_sequences


def test_grep_recall_per_type(data_dir):
    records = classify(load_sequences(data_dir, "test"), load_labels(data_dir, "test"))
    from logpoc.common.metrics import compute_metrics

    m = compute_metrics(records)
    per = m["recall_per_type"]
    assert per["visible_error"]["recall"] == 1.0
    for t in ["silent_skip", "wrong_order", "truncated", "retry_storm"]:
        assert per[t]["recall"] == 0.0
    assert m["false_positive_rate"] == 0.0
    assert m["localisation_hit_rate"] == 1.0


def test_run_writes_folder(data_dir, ml_root):
    payload = run(data_dir, "test", "grep-test")
    rdir = run_dir("grep-test")
    assert is_done(rdir)
    assert (rdir / "eval" / "metrics.json").exists()
    assert (rdir / "eval" / "scores.csv").exists()
    assert (rdir / "eval" / "report.md").exists()
    assert payload["metrics"]["recall"] == 0.2
