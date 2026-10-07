"""Deterministic synthetic log generator for four fictional services."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

SPLIT_ORDER = ["train", "val", "dev", "fewshot", "test"]


@dataclass(frozen=True)
class Line:
    kind: str
    service: str
    level: str
    message: str


def _hex(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("0123456789abcdef") for _ in range(n))


def _ids(rng: random.Random) -> dict[str, str]:
    return {
        "ip": ".".join(str(rng.randint(1, 254)) for _ in range(4)),
        "user": f"u{rng.randint(10000, 99999)}",
        "order": f"ord-{rng.randint(100000, 999999)}",
        "refund": f"ref-{rng.randint(100000, 999999)}",
        "items": str(rng.randint(1, 9)),
        "amount": f"{rng.uniform(5, 900):.2f}",
        "ccy": rng.choice(["DKK", "EUR", "USD"]),
        "key": f"k{_hex(rng, 6)}",
    }


def order_steps(v: dict[str, str]) -> dict[str, Line]:
    return {
        "s1": Line(
            "s1",
            "api-gateway",
            "INFO",
            f"request received method=POST path=/orders client={v['ip']}",
        ),
        "s2": Line("s2", "auth-service", "INFO", f"token validated user={v['user']}"),
        "s3": Line(
            "s3",
            "order-service",
            "INFO",
            f"order created order_id={v['order']} items={v['items']}",
        ),
        "s4": Line(
            "s4",
            "payment-service",
            "INFO",
            f"payment authorised order_id={v['order']} amount={v['amount']} currency={v['ccy']}",
        ),
        "s5": Line("s5", "order-service", "INFO", f"order confirmed order_id={v['order']}"),
        "cache": Line("cache", "api-gateway", "INFO", f"cache hit key={v['key']}"),
        "retry": Line("retry", "auth-service", "WARN", "auth retry attempt=1"),
        "err4": Line(
            "err4",
            "payment-service",
            "ERROR",
            f"payment gateway timeout order_id={v['order']}",
        ),
    }


def refund_lines(v: dict[str, str]) -> list[Line]:
    return [
        Line(
            "r1",
            "api-gateway",
            "INFO",
            f"request received method=POST path=/refunds client={v['ip']}",
        ),
        Line("r2", "auth-service", "INFO", f"token validated user={v['user']}"),
        Line(
            "r3",
            "order-service",
            "INFO",
            f"refund created refund_id={v['refund']} order_id={v['order']}",
        ),
        Line(
            "r4",
            "payment-service",
            "INFO",
            f"refund processed refund_id={v['refund']} amount={v['amount']} currency={v['ccy']}",
        ),
        Line("r5", "order-service", "INFO", f"refund confirmed refund_id={v['refund']}"),
    ]


def normal_order(rng: random.Random, cfg: dict, v: dict[str, str]) -> list[Line]:
    s = order_steps(v)
    lines: list[Line] = [s["s1"]]
    if rng.random() < cfg["flows"]["auth_retry_prob"]:
        lines.append(s["retry"])
    lines.append(s["s2"])
    if rng.random() < cfg["flows"]["cache_hit_prob"]:
        lines.append(s["cache"])
    lines += [s["s3"], s["s4"], s["s5"]]
    return lines


def inject(anomaly: str, normal: list[Line], v: dict[str, str], rng: random.Random) -> list[Line]:
    s = order_steps(v)
    kinds = [ln.kind for ln in normal]
    i3, i4 = kinds.index("s3"), kinds.index("s4")
    out = list(normal)
    if anomaly == "visible_error":
        out[i4] = s["err4"]
    elif anomaly == "silent_skip":
        del out[i4]
    elif anomaly == "wrong_order":
        out[i3], out[i4] = out[i4], out[i3]
    elif anomaly == "truncated":
        out = out[: i3 + 1]
    elif anomaly == "retry_storm":
        copies = rng.randint(4, 6)
        out = out[:i3] + [s["s3"]] * copies + out[i4:]
    else:
        raise ValueError(f"unknown anomaly type: {anomaly}")
    return out


def first_deviation(normal: list[Line], actual: list[Line]) -> int:
    """0-based index of the first line that differs from the normal flow."""
    for i, (a, b) in enumerate(zip(normal, actual, strict=False)):
        if a.kind != b.kind:
            return i
    return min(len(normal), len(actual))


def _fmt_ts(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"


def _make_trace(rng, cfg, used_ids, anomaly):
    while True:
        tid = "t" + _hex(rng, 12)
        if tid not in used_ids:
            used_ids.add(tid)
            break
    v = _ids(rng)
    if anomaly is None and rng.random() >= cfg["flows"]["order_share"]:
        return tid, refund_lines(v), None, None
    normal = normal_order(rng, cfg, v)
    if anomaly is None:
        return tid, normal, None, None
    actual = inject(anomaly, normal, v, rng)
    return tid, actual, anomaly, first_deviation(normal, actual)


def generate_split(name: str, spec: dict, cfg: dict, used_ids: set[str]):
    rng = random.Random(f"{cfg['seed']}:{name}")
    plan: list[str | None] = [None] * spec["normal"]
    for t in cfg["anomaly_types"]:
        plan += [t] * spec["per_type"]
    rng.shuffle(plan)

    clock = datetime.fromisoformat(cfg["start_time"].replace("Z", "+00:00")).astimezone(UTC)
    clock += timedelta(hours=SPLIT_ORDER.index(name) * 24)
    events = []
    labels = []
    for seq, anomaly in enumerate(plan):
        clock += timedelta(seconds=rng.expovariate(1.0 / cfg["mean_trace_gap_s"]))
        tid, lines, atype, dev = _make_trace(rng, cfg, used_ids, anomaly)
        t = clock
        for ln in lines:
            t = t + timedelta(milliseconds=rng.uniform(5, 140))
            head = f"{_fmt_ts(t)} {ln.level} {ln.service} trace={tid}"
            events.append((t, seq, f"{head} {ln.message}"))
        labels.append(
            {
                "trace_id": tid,
                "label": 0 if atype is None else 1,
                "anomaly_type": atype or "none",
                "first_deviation_index": "" if dev is None else dev,
            }
        )
    events.sort(key=lambda e: (e[0], e[1]))
    return [e[2] for e in events], labels


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_all(config_path: Path, out_dir: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    used_ids: set[str] = set()
    files: dict[str, str] = {}
    counts: dict[str, dict] = {}
    for name in SPLIT_ORDER:
        spec = cfg["splits"][name]
        raw, labels = generate_split(name, spec, cfg, used_ids)
        raw_path = out_dir / f"raw_{name}.log"
        raw_path.write_text("\n".join(raw) + "\n", newline="\n")
        files[raw_path.name] = sha256_file(raw_path)
        if spec["labels"]:
            lab_path = out_dir / f"labels_{name}.csv"
            with lab_path.open("w", newline="") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=["trace_id", "label", "anomaly_type", "first_deviation_index"],
                    lineterminator="\n",
                )
                w.writeheader()
                w.writerows(labels)
            files[lab_path.name] = sha256_file(lab_path)
        counts[name] = {
            "traces": len(labels),
            "normal": sum(1 for x in labels if x["label"] == 0),
            "anomalous": sum(1 for x in labels if x["label"] == 1),
            "lines": len(raw),
        }
    manifest = {
        "generator_version": cfg["generator_version"],
        "seed": cfg["seed"],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "counts": counts,
        "files": files,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
