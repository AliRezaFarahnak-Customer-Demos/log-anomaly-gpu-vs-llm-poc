import csv
import json
import re
from pathlib import Path

import pytest

from logpoc.data.generate_prod_like import FLOWS, generate_all, parse_line
from logpoc.data.prepare import load_sequences

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "data_prod_like.yaml"


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    d = tmp_path_factory.mktemp("prod_like")
    generate_all(CONFIG, d)
    return d


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_deterministic(out, tmp_path):
    again = generate_all(CONFIG, tmp_path)
    assert again["files"] == json.loads((out / "manifest.json").read_text())["files"]


def test_ten_days_every_line_parses(out):
    logs = sorted((out / "logs").glob("prod_*.log"))
    assert len(logs) == 10
    for path in logs:
        for line in path.read_text(encoding="utf-8").splitlines():
            parse_line(line)


def test_two_incidents_with_anomalies_inside_their_windows(out):
    assert [r["id"] for r in _rows(out / "incidents.csv")] == ["INC-1", "INC-2"]
    truth = _rows(out / "truth" / "trace_truth.csv")
    for inc in _rows(out / "incidents.csv"):
        hits = [r for r in truth if r["incident_id"] == inc["id"] and r["phase"] == "incident"]
        assert sum(int(r["label"]) for r in hits) >= 10
        assert all(inc["start"][:10] == r["start"][:10] for r in hits)


def test_labelled_set_shape(out):
    rows = [json.loads(x) for x in (out / "traces_labelled.jsonl").read_text().splitlines()]
    assert len(rows) == 80
    assert sum(r["expected_verdict"] == "anomalous" for r in rows) == 30
    assert {r["flow"] for r in rows} == set(FLOWS)
    templates = {r["template_id"] for r in _rows(out / "drain_templates.csv")}
    assert all(t in templates for r in rows for t in r["template_ids"])
    assert all(r["explanation"] and r["raw_lines"] for r in rows)


def test_labelled_traces_not_leaked_as_anomalies_by_name(out):
    # Raw lines must not carry the label: the verdict has to come from the sequence itself.
    rows = [json.loads(x) for x in (out / "traces_labelled.jsonl").read_text().splitlines()]
    assert not any("anomal" in line.lower() for r in rows for line in r["raw_lines"])


def test_split_files_for_the_pipeline(out):
    labelled = {
        json.loads(x)["trace_id"] for x in (out / "traces_labelled.jsonl").read_text().splitlines()
    }

    def ids(split):
        return set(re.findall(r" trace=(\S+) ", (out / f"raw_{split}.log").read_text()))

    assert ids("test") == labelled
    assert not (ids("train") | ids("val")) & labelled
    assert len(ids("train")) > 3000 and len(ids("val")) > 300
    assert len(_rows(out / "labels_test.csv")) == 80


def test_prepare_masks_prod_like_values(out):
    lines = [ln for s in load_sequences(out, "test") for ln in s["lines"]]
    assert lines
    assert not any(re.search(r"APP-2026-|=C\d{8}|=P\d{7}|=LN\d{9}", ln) for ln in lines)
