"""Run folders, meta.json, never-overwrite and resume helpers."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_ML_ROOT = "/mnt/ml"


class InjectedFailure(RuntimeError):
    """Raised once per run when FAIL_AFTER_STEPS is set."""


class RunAlreadyExists(RuntimeError):
    pass


def ml_root() -> Path:
    return Path(os.environ.get("ML_ROOT") or DEFAULT_ML_ROOT)


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_info() -> tuple[str, bool]:
    """Git sha and dirty flag. Baked in at image build time, read from git otherwise."""
    sha = os.environ.get("GIT_SHA")
    dirty = os.environ.get("GIT_DIRTY")
    if sha:
        return sha, str(dirty).lower() == "true"
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return sha, bool(status)
    except (OSError, subprocess.CalledProcessError):
        return "nogit", False


def default_run_id(prefix: str = "") -> str:
    sha, _ = git_info()
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M")
    return f"{prefix}{sha}-{stamp}"


def resolve_run_id(explicit: str | None = None, prefix: str = "") -> str:
    """Explicit value, then the RUN_ID env var, then a local-only default.

    In jobs RUN_ID must be set by the caller. A default computed at process start would
    change on a platform retry and break resume.
    """
    return explicit or os.environ.get("RUN_ID") or default_run_id(prefix)


def run_dir(run_id: str, root: Path | None = None) -> Path:
    return (root or ml_root()) / "runs" / run_id


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def write_json(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=False, default=str) + "\n")
    os.replace(tmp, path)


def is_done(rdir: Path) -> bool:
    return (Path(rdir) / "DONE").exists()


def mark_done(rdir: Path) -> None:
    (Path(rdir) / "DONE").write_text(utc_now() + "\n")


def create_fresh_run(run_id: str, root: Path | None = None) -> Path:
    """For one-shot runs (baseline, PoC 2). Refuses to touch an existing run."""
    rdir = run_dir(run_id, root)
    if rdir.exists():
        raise RunAlreadyExists(f"run folder already exists, pick a new RUN_ID: {rdir}")
    rdir.mkdir(parents=True)
    return rdir


def read_meta(rdir: Path) -> dict:
    p = Path(rdir) / "meta.json"
    return read_json(p) if p.exists() else {}


def update_meta(rdir: Path, **fields) -> dict:
    meta = read_meta(rdir)
    meta.update(fields)
    write_json(Path(rdir) / "meta.json", meta)
    return meta


def record_resume(rdir: Path, checkpoint: str | None, step: int | None, reason: str = "") -> dict:
    meta = read_meta(rdir)
    meta.setdefault("resumes", []).append(
        {
            "at": utc_now(),
            "checkpoint": checkpoint,
            "step": step,
            "job_execution": os.environ.get("CONTAINER_APP_JOB_EXECUTION_NAME"),
            "reason": reason,
        }
    )
    write_json(Path(rdir) / "meta.json", meta)
    return meta


def checkpoint_step(path: Path) -> int:
    return int(Path(path).name.rsplit("-", 1)[1])


def is_complete_checkpoint(path: Path) -> bool:
    return (Path(path) / "trainer_state.json").is_file()


def find_resume_checkpoint(ckpt_dir: Path) -> Path | None:
    """Newest checkpoint that has trainer_state.json. A partial newest one is skipped."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return None
    candidates = []
    for p in ckpt_dir.iterdir():
        if p.is_dir() and p.name.startswith("checkpoint-"):
            try:
                candidates.append((checkpoint_step(p), p))
            except ValueError:
                continue
    for _, p in sorted(candidates, reverse=True):
        if is_complete_checkpoint(p):
            return p
    return None


def maybe_inject_failure(rdir: Path, step: int, fail_after: int | None = None) -> None:
    """Raise InjectedFailure once per run when step >= FAIL_AFTER_STEPS. A marker file stops
    it firing again after the platform retries the replica."""
    if fail_after is None:
        raw = os.environ.get("FAIL_AFTER_STEPS", "").strip()
        fail_after = int(raw) if raw else 0
    if fail_after <= 0 or step < fail_after:
        return
    marker = Path(rdir) / "fail_injected"
    if marker.exists():
        return
    marker.write_text(f"injected at step {step} at {utc_now()}\n")
    raise InjectedFailure(f"FAIL_AFTER_STEPS={fail_after} reached at step {step}")


def library_versions() -> dict:
    """Versions of the libraries that matter, without importing heavy ones when absent."""
    from importlib import metadata

    out = {"python": platform.python_version()}
    for name in ["torch", "transformers", "peft", "drain3", "openai", "numpy"]:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out
