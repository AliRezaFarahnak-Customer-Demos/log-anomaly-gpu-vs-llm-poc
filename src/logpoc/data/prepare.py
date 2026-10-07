"""Parse raw logs into template sequences with drain3."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

from drain3 import TemplateMiner
from drain3.masking import MaskingInstruction
from drain3.persistence_handler import PersistenceHandler
from drain3.template_miner_config import TemplateMinerConfig

from logpoc.data.generate import SPLIT_ORDER

LINE_RE = re.compile(r"^(\S+) (\w+) (\S+) trace=(\S+) (.*)$")
MASKED_KEYS = ["order_id", "refund_id", "user", "client", "items", "amount", "currency", "key"]
TRACE_START = "TRACE START"
TRACE_END = "TRACE END"


class MemoryPersistence(PersistenceHandler):
    """Keeps the miner state in memory so a copy can be parsed without touching the original."""

    def __init__(self, state: bytes | None = None):
        self.state = state

    def save_state(self, state):
        self.state = state

    def load_state(self):
        return self.state


def make_config() -> TemplateMinerConfig:
    cfg = TemplateMinerConfig()
    cfg.drain_sim_th = 0.9
    cfg.drain_depth = 4
    cfg.profiling_enabled = False
    cfg.snapshot_compress_state = False
    cfg.masking_instructions = [
        MaskingInstruction(r"(?<=\b" + k + r"=)\S+", "*") for k in MASKED_KEYS
    ]
    return cfg


def parse_raw(path: Path) -> list[tuple[str, str]]:
    """Return (trace_id, content) pairs in timestamp order. Content is 'service LEVEL message'."""
    rows = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        m = LINE_RE.match(line)
        if not m:
            raise ValueError(f"{path}:{n}: unparseable line: {line!r}")
        ts, level, service, tid, msg = m.groups()
        rows.append((ts, n, tid, f"{service} {level} {msg}"))
    rows.sort(key=lambda r: (r[0], r[1]))
    return [(tid, content) for _, _, tid, content in rows]


def build_text(lines: list[str]) -> str:
    return TRACE_START + "\n" + "\n".join(lines) + "\n" + TRACE_END


def _template_text(miner: TemplateMiner, cluster_id: int) -> str:
    return miner.drain.id_to_cluster[cluster_id].get_template()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(data_dir: Path, out_dir: Path | None = None) -> Path:
    data_dir = Path(data_dir)
    out_dir = Path(out_dir) if out_dir else data_dir / "prepared"
    tmp = out_dir.with_name(out_dir.name + f".tmp-{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    cfg = make_config()

    parsed = {s: parse_raw(data_dir / f"raw_{s}.log") for s in SPLIT_ORDER}

    miners: dict[str, TemplateMiner] = {}
    grouped: dict[str, dict[str, list[int]]] = {}
    train_miner = TemplateMiner(MemoryPersistence(), cfg)
    for split in SPLIT_ORDER:
        # drain3 0.9.1 cannot reload its own saved state with current jsonpickle,
        # so later splits start from an in-memory copy of the learned train miner
        miner = train_miner if split == "train" else copy.deepcopy(train_miner)
        by_trace: dict[str, list[int]] = {}
        for tid, content in parsed[split]:
            by_trace.setdefault(tid, []).append(miner.add_log_message(content)["cluster_id"])
        if split == "train":
            miner.save_state("train-final")
            (tmp / "miner_state.bin").write_bytes(miner.persistence_handler.state)
        miners[split] = miner
        grouped[split] = by_trace

    # global template ids: train templates first, then templates first seen in later splits
    text_ids: dict[str, int] = {}
    for cid in sorted(train_miner.drain.id_to_cluster):
        text_ids.setdefault(_template_text(train_miner, cid), len(text_ids) + 1)

    counts: Counter = Counter()
    train_counts: Counter = Counter()
    for split in SPLIT_ORDER:
        miner = miners[split]
        records = []
        for tid, cids in grouped[split].items():
            lines = [_template_text(miner, c) for c in cids]
            ids = [text_ids.setdefault(t, len(text_ids) + 1) for t in lines]
            counts.update(ids)
            if split == "train":
                train_counts.update(ids)
            records.append(
                {"trace_id": tid, "lines": lines, "template_ids": ids, "text": build_text(lines)}
            )
        with (tmp / f"sequences_{split}.jsonl").open("w", newline="\n") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    id_to_text = {i: t for t, i in text_ids.items()}
    templates = [
        {
            "id": i,
            "text": id_to_text[i],
            "count": counts[i],
            "train_count": train_counts[i],
            "seen_in_train": train_counts[i] > 0,
        }
        for i in sorted(id_to_text)
    ]
    (tmp / "templates.json").write_text(json.dumps({"templates": templates}, indent=2) + "\n")
    manifest = data_dir / "manifest.json"
    (tmp / "prepare_meta.json").write_text(
        json.dumps(
            {
                "data_manifest_sha256": sha256_file(manifest) if manifest.exists() else None,
                "templates_in_train": sum(1 for t in templates if t["seen_in_train"]),
                "templates_total": len(templates),
            },
            indent=2,
        )
        + "\n"
    )
    if out_dir.exists():
        shutil.rmtree(out_dir)
    os.replace(tmp, out_dir)
    return out_dir


def ensure_prepared(data_dir: Path) -> Path:
    data_dir = Path(data_dir)
    out = data_dir / "prepared"
    meta = out / "prepare_meta.json"
    manifest = data_dir / "manifest.json"
    want = sha256_file(manifest) if manifest.exists() else None
    if meta.exists() and json.loads(meta.read_text()).get("data_manifest_sha256") == want:
        return out
    return prepare(data_dir)


def data_manifest_hash(data_dir: Path) -> str | None:
    manifest = Path(data_dir) / "manifest.json"
    return sha256_file(manifest) if manifest.exists() else None


def load_sequences(data_dir: Path, split: str) -> list[dict]:
    path = ensure_prepared(data_dir) / f"sequences_{split}.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def load_labels(data_dir: Path, split: str) -> dict[str, dict]:
    path = Path(data_dir) / f"labels_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"no labels for split {split!r}: {path}")
    with path.open() as f:
        out = {}
        for r in csv.DictReader(f):
            dev = r["first_deviation_index"]
            out[r["trace_id"]] = {
                "label": int(r["label"]),
                "anomaly_type": r["anomaly_type"],
                "first_deviation_index": int(dev) if dev != "" else None,
            }
        return out
