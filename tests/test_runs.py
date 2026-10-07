from __future__ import annotations

import pytest

from logpoc.common import runs


def test_run_id_explicit_env_default(ml_root, monkeypatch):
    assert runs.resolve_run_id("abc") == "abc"
    monkeypatch.setenv("RUN_ID", "from-env")
    assert runs.resolve_run_id() == "from-env"
    monkeypatch.delenv("RUN_ID")
    monkeypatch.setenv("GIT_SHA", "abc1234")
    rid = runs.resolve_run_id()
    assert rid.startswith("abc1234-") and len(rid.split("-")) == 3


def test_git_info_from_env(monkeypatch):
    monkeypatch.setenv("GIT_SHA", "deadbee")
    monkeypatch.setenv("GIT_DIRTY", "true")
    assert runs.git_info() == ("deadbee", True)


def test_never_overwrite(ml_root):
    runs.create_fresh_run("r1")
    with pytest.raises(runs.RunAlreadyExists):
        runs.create_fresh_run("r1")


def _ckpt(root, step, complete=True):
    d = root / f"checkpoint-{step}"
    d.mkdir(parents=True)
    if complete:
        (d / "trainer_state.json").write_text("{}")
    return d


def test_resume_picks_newest_complete(tmp_path):
    _ckpt(tmp_path, 20)
    _ckpt(tmp_path, 40)
    _ckpt(tmp_path, 100, complete=False)
    assert runs.find_resume_checkpoint(tmp_path).name == "checkpoint-40"


def test_resume_numeric_order_not_lexical(tmp_path):
    _ckpt(tmp_path, 9)
    _ckpt(tmp_path, 100)
    assert runs.find_resume_checkpoint(tmp_path).name == "checkpoint-100"


def test_resume_none_when_nothing_valid(tmp_path):
    assert runs.find_resume_checkpoint(tmp_path / "missing") is None
    _ckpt(tmp_path, 20, complete=False)
    assert runs.find_resume_checkpoint(tmp_path) is None


def test_fail_after_steps_fires_once(tmp_path):
    runs.maybe_inject_failure(tmp_path, 5, fail_after=10)
    with pytest.raises(runs.InjectedFailure):
        runs.maybe_inject_failure(tmp_path, 10, fail_after=10)
    runs.maybe_inject_failure(tmp_path, 11, fail_after=10)
    runs.maybe_inject_failure(tmp_path, 10, fail_after=10)


def test_fail_after_steps_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv("FAIL_AFTER_STEPS", raising=False)
    runs.maybe_inject_failure(tmp_path, 1000)
    monkeypatch.setenv("FAIL_AFTER_STEPS", "0")
    runs.maybe_inject_failure(tmp_path, 1000)


def test_record_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTAINER_APP_JOB_EXECUTION_NAME", "job-train-abc")
    runs.update_meta(tmp_path, run_id="x")
    meta = runs.record_resume(tmp_path, "checkpoint-40", 40, "test")
    assert meta["resumes"][0]["checkpoint"] == "checkpoint-40"
    assert meta["resumes"][0]["job_execution"] == "job-train-abc"
    assert runs.read_meta(tmp_path)["run_id"] == "x"
