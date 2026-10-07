from __future__ import annotations

import json
from pathlib import Path

import pytest

from logpoc.common import runs
from logpoc.data.prepare import build_text, load_sequences
from logpoc.poc1_gpu import evaluate, train

ROOT = Path(__file__).resolve().parents[1]


def test_line_starts():
    lines = ["aa", "bbb"]
    text = build_text(lines)
    starts = evaluate.line_starts(lines)
    assert text[starts[0] :].startswith("aa")
    assert text[starts[1] :].startswith("bbb")
    assert text[starts[2] :] == "TRACE END"


def test_aggregate_tokens_maps_to_lines():
    lines = ["aa", "bbb"]
    text = build_text(lines)  # "TRACE START\naa\nbbb\nTRACE END"
    starts = evaluate.line_starts(lines)
    # tokens: TRACE, START, \n, aa, \n, bbb, \n, TRACE, END
    spans = ["TRACE", " START", "\n", "aa", "\n", "bbb", "\n", "TRACE", " END"]
    offsets, pos = [], 0
    for s in spans:
        offsets.append((pos, pos + len(s)))
        pos += len(s)
    assert text == "".join(spans)
    nll = [0.5, 9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # for tokens 1..8
    agg = evaluate.aggregate_tokens(text, offsets, nll, starts)
    scores = agg["line_scores"]
    # " START" is header (ignored), "\n" after header belongs to header, so ignored too
    assert -1 not in scores
    # nll[i] belongs to token i+1. Line 0: "aa" (1.0) and its closing newline (2.0)
    assert scores[0] == pytest.approx(1.5)
    # line 1: "bbb" (3.0) and its closing newline (4.0)
    assert scores[1] == pytest.approx(3.5)
    # TRACE END marker index = number of lines: "TRACE" (5.0) and " END" (6.0)
    assert scores[2] == pytest.approx(5.5)
    res = evaluate.trace_result(lines, agg)
    assert res["worst_index"] == 2 and res["worst_text"] == "TRACE END"
    assert res["score"] == pytest.approx(5.5)


def test_newline_merged_with_next_token_belongs_to_next_line():
    lines = ["aa", "bb"]
    text = build_text(lines)
    starts = evaluate.line_starts(lines)
    # a token "\nbb" starts with a newline but its first real character is in line 1
    i = text.index("\nbb")
    offsets = [(0, 5), (i, i + 3)]
    agg = evaluate.aggregate_tokens(text, offsets, [3.0], starts)
    assert agg["line_scores"] == {1: 3.0}


def test_trace_result_no_scores():
    res = evaluate.trace_result(["a"], {"line_scores": {}, "mean_nll": 0.0})
    assert res["worst_index"] is None


def fake_scorer(seqs):
    """Normal traces score low. Anything with an unusual line count or an ERROR scores high."""
    out = []
    for s in seqs:
        n = len(s["lines"])
        odd = any("ERROR" in x for x in s["lines"]) or n < 4 or n > 8
        out.append(
            {
                "score": 5.0 if odd else 1.0 + (n % 3) * 0.01,
                "mean_nll": 2.0 if odd else 0.5,
                "worst_index": 3 if odd else 0,
                "worst_text": s["lines"][min(3, n - 1)],
            }
        )
    return out


def test_evaluate_run_with_fake_scorer(ml_root, monkeypatch, data_dir):
    monkeypatch.setenv("RUN_ID", "run-e")
    train.run(
        ROOT / "configs" / "tiny.yaml",
        data_dir,
        fit=lambda ctx: {"final_loss": 0.1, "steps": 1},
    )
    payload = evaluate.run("run-e", data_dir, "test", scorer=fake_scorer)
    rdir = runs.run_dir("run-e")
    for name in ["metrics.json", "scores.csv", "report.md", "scores.png"]:
        assert (rdir / "eval" / name).exists(), name
    m = json.loads((rdir / "eval" / "metrics.json").read_text())
    assert m["method"] == "poc1-gpu"
    assert m["metrics"]["recall_per_type"]["visible_error"]["recall"] == 1.0
    assert m["metrics"]["auroc"] is not None
    assert m["cost_per_1000"] is None
    assert "n/a (set prices" in (rdir / "eval" / "report.md").read_text()
    assert payload["threshold"] > 0
    # dev goes to its own folder and reuses the cached val scores
    evaluate.run("run-e", data_dir, "dev", scorer=fake_scorer)
    assert (rdir / "eval-dev" / "metrics.json").exists()
    assert (rdir / "val_scores.json").exists()


def test_evaluate_refuses_unfinished_run(ml_root, data_dir):
    with pytest.raises(SystemExit):
        evaluate.run("nope", data_dir, "test", scorer=fake_scorer)


def test_sequences_fixture_sane(data_dir):
    assert len(load_sequences(data_dir, "val")) == 200
