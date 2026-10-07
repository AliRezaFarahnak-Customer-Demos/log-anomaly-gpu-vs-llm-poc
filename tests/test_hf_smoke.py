"""Slow smoke test: 5 training steps on Qwen/Qwen3-0.6B. Needs internet, set RUN_HF_TESTS=1."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from logpoc.common import runs

pytestmark = pytest.mark.skipif(os.environ.get("RUN_HF_TESTS") != "1", reason="set RUN_HF_TESTS=1")

ROOT = Path(__file__).resolve().parents[1]


def test_five_steps_then_evaluate(ml_root, monkeypatch, data_dir, tmp_path):
    import yaml

    from logpoc.poc1_gpu import download_model, evaluate, train

    cfg = yaml.safe_load((ROOT / "configs" / "tiny.yaml").read_text())
    cfg["train"]["max_steps"] = 5
    cfg["train"]["save_steps"] = 2
    cfg["train"]["per_device_batch_size"] = 4
    cfg_path = tmp_path / "smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    monkeypatch.setenv("HF_HOME", str(ml_root / "hf-cache"))
    download_model.download(cfg["base_model"]["repo_id"], ml_root / cfg["base_model"]["local_dir"])
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("RUN_ID", "smoke")
    assert train.run(cfg_path, data_dir) == 0
    rdir = runs.run_dir("smoke")
    assert (rdir / "adapter" / "adapter_config.json").exists()
    assert runs.read_json(rdir / "train_metrics.json")["steps"] == 5
    evaluate.run("smoke", data_dir, "dev", device="cpu")
    assert (rdir / "eval-dev" / "metrics.json").exists()
