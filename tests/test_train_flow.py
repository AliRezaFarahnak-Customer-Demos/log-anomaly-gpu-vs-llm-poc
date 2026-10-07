from __future__ import annotations

from pathlib import Path

import pytest

from logpoc.common import runs
from logpoc.poc1_gpu import train

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "tiny.yaml"
TOTAL_STEPS = 100


def fake_fit(ctx: train.FitContext) -> dict:
    """Pretends to train: writes a checkpoint every 20 steps and honours FAIL_AFTER_STEPS."""
    start = runs.checkpoint_step(ctx.resume_from) if ctx.resume_from else 0
    ckpt_dir = ctx.rdir / "checkpoints"
    for step in range(start + 1, TOTAL_STEPS + 1):
        runs.maybe_inject_failure(ctx.rdir, step)
        if step % 20 == 0:
            d = ckpt_dir / f"checkpoint-{step}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "trainer_state.json").write_text("{}")
    (ctx.rdir / "adapter").mkdir(exist_ok=True)
    return {"final_loss": 0.1, "steps": TOTAL_STEPS, "peak_gpu_memory_bytes": None}


@pytest.fixture
def env(ml_root, monkeypatch, data_dir):
    monkeypatch.setenv("RUN_ID", "run-a")
    monkeypatch.setenv("GIT_SHA", "abc1234")
    return data_dir


def test_full_run_writes_everything(env):
    assert train.run(CONFIG, env, fit=fake_fit) == 0
    rdir = runs.run_dir("run-a")
    for name in ["config.yaml", "meta.json", "train_metrics.json", "DONE", "adapter"]:
        assert (rdir / name).exists(), name
    meta = runs.read_meta(rdir)
    assert meta["git_sha"] == "abc1234"
    assert meta["data_manifest_sha256"]
    assert meta["base_model"]["repo_id"] == "Qwen/Qwen3-0.6B"
    assert meta["resumes"] == []
    assert "python" in meta["versions"]


def test_done_run_is_a_noop(env):
    train.run(CONFIG, env, fit=fake_fit)
    called = []
    train.run(CONFIG, env, fit=lambda ctx: called.append(1) or {})
    assert not called


def test_injected_failure_then_resume(env, monkeypatch):
    monkeypatch.setenv("FAIL_AFTER_STEPS", "50")
    with pytest.raises(runs.InjectedFailure):
        train.run(CONFIG, env, fit=fake_fit)
    rdir = runs.run_dir("run-a")
    assert not runs.is_done(rdir)
    assert (rdir / "fail_injected").exists()

    assert train.run(CONFIG, env, fit=fake_fit) == 0
    meta = runs.read_meta(rdir)
    assert runs.is_done(rdir)
    assert len(meta["resumes"]) == 1
    assert meta["resumes"][0]["checkpoint"] == "checkpoint-40"
    assert meta["resumes"][0]["step"] == 40
    assert len(meta["attempt_seconds"]) == 2
    assert runs.read_json(rdir / "train_metrics.json")["attempts"] == 2


def test_partial_checkpoint_is_skipped_on_resume(env, monkeypatch):
    monkeypatch.setenv("FAIL_AFTER_STEPS", "50")
    with pytest.raises(runs.InjectedFailure):
        train.run(CONFIG, env, fit=fake_fit)
    rdir = runs.run_dir("run-a")
    (rdir / "checkpoints" / "checkpoint-40" / "trainer_state.json").unlink()
    train.run(CONFIG, env, fit=fake_fit)
    assert runs.read_meta(rdir)["resumes"][0]["checkpoint"] == "checkpoint-20"


def test_changed_config_is_refused(env, tmp_path):
    with pytest.raises(runs.InjectedFailure):
        # leave the run unfinished
        def boom(ctx):
            raise runs.InjectedFailure("stop")

        train.run(CONFIG, env, fit=boom)
    other = tmp_path / "other.yaml"
    other.write_text((ROOT / "configs" / "tiny_r8.yaml").read_text())
    with pytest.raises(SystemExit):
        train.run(other, env, fit=fake_fit)
