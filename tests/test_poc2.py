import io
import json

from logpoc import compare
from logpoc.cli import main
from logpoc.common import runs
from logpoc.poc2_llm import classify


def test_poc2_dry_run_then_compare(ml_root, data_dir):
    assert classify.run(data_dir, split="test", dry_run=True, run_id="p2") == 0
    m = json.loads((runs.run_dir("p2") / "eval" / "metrics.json").read_text())
    assert m["method"] == "poc2-llm"
    assert m["metrics"]["recall_per_type"]["visible_error"]["recall"] == 1.0
    assert m["metrics"]["false_positive_rate"] == 0.0
    assert (runs.run_dir("p2") / "answers.jsonl").exists()

    assert compare.run() == 0
    assert "| p2 | poc2-llm |" in (ml_root / "runs" / "RUNS.md").read_text()


def test_expected_flow_goes_into_the_system_prompt():
    system, user = classify.build_messages(["a", "b"], "flow doc")
    assert "flow doc" in system["content"]
    assert user["content"] == "0: a\n1: b"


def test_every_schema_property_is_described_and_required():
    props = classify.SCHEMA["properties"]
    assert all(p.get("description") for p in props.values())
    assert classify.SCHEMA["required"] == list(props)


def test_save_result_from_job_log_line(ml_root, monkeypatch):
    files = {"meta.json": {"run_id": "gpu1"}, "eval/metrics.json": {"method": "poc1-gpu"}}
    monkeypatch.setattr("sys.stdin", io.StringIO("noise\nRESULT " + json.dumps(files) + "\n"))
    assert main(["save-result"]) == 0
    assert json.loads((runs.run_dir("gpu1") / "eval" / "metrics.json").read_text()) == {
        "method": "poc1-gpu"
    }
