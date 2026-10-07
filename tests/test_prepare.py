from __future__ import annotations

import json

from logpoc.data.prepare import (
    TRACE_END,
    TRACE_START,
    build_text,
    load_labels,
    load_sequences,
    parse_raw,
)


def test_templates_learned_on_train(prepared_dir):
    templates = json.loads((prepared_dir / "templates.json").read_text())["templates"]
    train_seen = [t for t in templates if t["seen_in_train"]]
    assert len(train_seen) == 11
    unseen = [t for t in templates if not t["seen_in_train"]]
    assert len(unseen) == 1
    assert "ERROR" in unseen[0]["text"]
    assert all("<*>" in t["text"] or "attempt=1" in t["text"] for t in templates)


def test_templates_keep_service_and_level_first(prepared_dir):
    templates = json.loads((prepared_dir / "templates.json").read_text())["templates"]
    for t in templates:
        service, level = t["text"].split(" ")[:2]
        assert service.endswith(("-gateway", "-service"))
        assert level in {"INFO", "WARN", "ERROR"}


def test_grouping_of_interleaved_traces(data_dir, prepared_dir):
    raw = parse_raw(data_dir / "raw_test.log")
    ids_in_raw_order = [tid for tid, _ in raw]
    assert any(a != b for a, b in zip(ids_in_raw_order, ids_in_raw_order[1:], strict=False))
    seqs = load_sequences(data_dir, "test")
    assert len(seqs) == 300
    expected: dict[str, int] = {}
    for tid, _ in raw:
        expected[tid] = expected.get(tid, 0) + 1
    for s in seqs:
        assert len(s["lines"]) == expected[s["trace_id"]]
        assert s["text"] == build_text(s["lines"])


def test_text_markers():
    text = build_text(["a", "b"])
    assert text == f"{TRACE_START}\na\nb\n{TRACE_END}"


def test_normal_trace_order(data_dir, prepared_dir):
    labels = load_labels(data_dir, "test")
    for s in load_sequences(data_dir, "test"):
        lab = labels[s["trace_id"]]
        if lab["anomaly_type"] == "silent_skip":
            assert not any("payment authorised" in x for x in s["lines"])
        if lab["anomaly_type"] == "visible_error":
            assert any(x.startswith("payment-service ERROR") for x in s["lines"])


def test_unseen_template_only_in_visible_error(data_dir, prepared_dir):
    templates = json.loads((prepared_dir / "templates.json").read_text())["templates"]
    unseen = {t["id"] for t in templates if not t["seen_in_train"]}
    labels = load_labels(data_dir, "test")
    for s in load_sequences(data_dir, "test"):
        has_unseen = bool(unseen & set(s["template_ids"]))
        assert has_unseen == (labels[s["trace_id"]]["anomaly_type"] == "visible_error")
