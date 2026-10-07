from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

from logpoc.data.generate import generate_all

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "data_v1.yaml"
LINE_RE = re.compile(r"^(\S+) (\w+) (\S+) trace=(\S+) (.*)$")


def read_raw(path: Path) -> dict[str, list[tuple[str, str, str]]]:
    traces: dict[str, list[tuple[str, str, str]]] = {}
    for line in path.read_text().splitlines():
        m = LINE_RE.match(line)
        assert m, line
        _, level, service, tid, msg = m.groups()
        traces.setdefault(tid, []).append((level, service, msg))
    return traces


def read_labels(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("v1")
    generate_all(CONFIG, out)
    return out


def test_deterministic(data_dir, tmp_path):
    generate_all(CONFIG, tmp_path)
    import json

    a = json.loads((data_dir / "manifest.json").read_text())
    b = json.loads((tmp_path / "manifest.json").read_text())
    assert a["files"] == b["files"]
    assert len(a["files"]) == 5 + 3


def test_committed_data_matches_generator(data_dir):
    committed = ROOT / "data" / "synthetic" / "v1" / "manifest.json"
    if not committed.exists():
        pytest.skip("committed data not generated yet")
    import json

    assert (
        json.loads(committed.read_text())["files"]
        == json.loads((data_dir / "manifest.json").read_text())["files"]
    )


def test_split_sizes_and_disjoint_ids(data_dir):
    expected = {"train": 2000, "val": 200, "dev": 40, "fewshot": 6, "test": 300}
    seen: dict[str, str] = {}
    for split, n in expected.items():
        traces = read_raw(data_dir / f"raw_{split}.log")
        assert len(traces) == n, split
        for tid in traces:
            assert tid not in seen, f"{tid} in {split} and {seen[tid]}"
            seen[tid] = split


def test_label_counts(data_dir):
    for split, normal, per_type in [("dev", 20, 4), ("fewshot", 1, 1), ("test", 200, 20)]:
        rows = read_labels(data_dir / f"labels_{split}.csv")
        assert sum(r["label"] == "0" for r in rows) == normal
        for t in ["visible_error", "silent_skip", "wrong_order", "truncated", "retry_storm"]:
            assert sum(r["anomaly_type"] == t for r in rows) == per_type


def test_train_and_val_are_all_normal_and_clean(data_dir):
    for split in ["train", "val"]:
        text = (data_dir / f"raw_{split}.log").read_text().lower()
        for bad in ["error", "fatal", "exception", "timeout"]:
            assert bad not in text


def test_traces_are_interleaved(data_dir):
    ids = [LINE_RE.match(x).group(4) for x in (data_dir / "raw_train.log").read_text().splitlines()]
    switches = sum(1 for a, b in zip(ids, ids[1:], strict=False) if a != b)
    assert switches > len(set(ids)) * 2


def test_timestamps_sorted(data_dir):
    stamps = [x.split(" ", 1)[0] for x in (data_dir / "raw_test.log").read_text().splitlines()]
    assert stamps == sorted(stamps)


def test_labels_match_data(data_dir):
    traces = read_raw(data_dir / "raw_test.log")
    rows = read_labels(data_dir / "labels_test.csv")
    for r in rows:
        lines = traces[r["trace_id"]]
        msgs = [m for _, _, m in lines]
        has_error = any(level == "ERROR" for level, _, _ in lines)
        has_payment = any(m.startswith("payment authorised") for m in msgs)
        t = r["anomaly_type"]
        assert has_error == (t == "visible_error"), r
        if t == "silent_skip":
            assert not has_payment
            assert not has_error
        if t == "none":
            assert r["first_deviation_index"] == ""


def test_first_deviation_index(data_dir):
    traces = read_raw(data_dir / "raw_test.log")
    for r in read_labels(data_dir / "labels_test.csv"):
        if r["label"] == "0":
            continue
        msgs = [m.split(" ")[0] + " " + m.split(" ")[1] for _, _, m in traces[r["trace_id"]]]
        n = len(msgs)
        idx = int(r["first_deviation_index"])
        t = r["anomaly_type"]
        # step 3 is the "order created" line. Benign extras shift it by 0 to 2.
        i3 = next(i for i, m in enumerate(msgs) if m == "order created")
        if t == "truncated":
            assert idx == n == i3 + 1
        elif t == "wrong_order":
            # the two steps are swapped, so "order created" now sits one line later
            assert idx == i3 - 1 and msgs[idx] == "payment authorised"
        elif t == "retry_storm":
            assert idx == i3 + 1 and msgs[idx] == "order created"
        elif t == "silent_skip":
            assert idx == i3 + 1 and msgs[idx] == "order confirmed"
        elif t == "visible_error":
            assert idx == i3 + 1 and msgs[idx] == "payment gateway"
        else:
            raise AssertionError(t)


def test_retry_storm_repeats(data_dir):
    traces = read_raw(data_dir / "raw_test.log")
    for r in read_labels(data_dir / "labels_test.csv"):
        if r["anomaly_type"] == "retry_storm":
            n = sum(m.startswith("order created") for _, _, m in traces[r["trace_id"]])
            assert 4 <= n <= 6


def test_anomalies_only_in_order_flow(data_dir):
    traces = read_raw(data_dir / "raw_test.log")
    for r in read_labels(data_dir / "labels_test.csv"):
        if r["label"] == "1":
            assert not any("refund" in m for _, _, m in traces[r["trace_id"]])


def test_only_known_services(data_dir):
    allowed = {"api-gateway", "auth-service", "order-service", "payment-service"}
    for tid_lines in read_raw(data_dir / "raw_test.log").values():
        assert {s for _, s, _ in tid_lines} <= allowed
