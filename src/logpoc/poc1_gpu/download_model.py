"""Download base model files once. This is the only step that needs internet."""

from __future__ import annotations

import os
import time
from pathlib import Path

from logpoc.common.runs import utc_now, write_json

ALLOW = [
    "*.safetensors",
    "*.safetensors.index.json",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
]


def download(repo_id: str, out: Path) -> dict:
    # must be set before huggingface_hub is imported, the image default is offline
    os.environ["HF_HUB_OFFLINE"] = "0"
    from huggingface_hub import HfApi, snapshot_download

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or None
    t0 = time.time()
    sha = HfApi().model_info(repo_id, token=token).sha
    snapshot_download(
        repo_id,
        revision=sha,
        local_dir=str(out),
        allow_patterns=ALLOW,
        token=token,
    )
    files = {
        str(p.relative_to(out)): p.stat().st_size
        for p in sorted(out.rglob("*"))
        if p.is_file() and ".cache" not in p.parts and p.name != "model_manifest.json"
    }
    manifest = {
        "repo_id": repo_id,
        "commit": sha,
        "files": files,
        "total_bytes": sum(files.values()),
        "download_seconds": round(time.time() - t0, 1),
        "downloaded_at": utc_now(),
    }
    write_json(out / "model_manifest.json", manifest)
    print(f"downloaded {repo_id} at {sha} to {out} ({manifest['total_bytes']} bytes)")
    return manifest
