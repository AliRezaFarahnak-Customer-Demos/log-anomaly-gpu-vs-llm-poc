"""Deterministic generator for a synthetic, production-shaped log extract of a mortgage bank.

It mimics what a Splunk export of application logs would look like: ten days, four business
flows, background noise, two major incidents with a precursor phase, and a small labelled set.
Everything here is invented. No customer data is used.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml
from drain3 import TemplateMiner
from drain3.masking import MaskingInstruction
from drain3.template_miner_config import TemplateMinerConfig

ANOMALY_TYPES = ["visible_error", "silent_skip", "wrong_order", "truncated", "retry_storm"]
LINE_RE = re.compile(r"^(\S+) (\w+) (\S+) host=(\S+) corrId=(\S+) (.*)$")
NO_CORR = "-"


@dataclass(frozen=True)
class Line:
    kind: str
    service: str
    level: str
    message: str


@dataclass(frozen=True)
class FlowSpec:
    label: str
    steps: tuple[tuple[str, str, str, str, str], ...]  # kind, service, level, label, message
    err_kind: str
    err: tuple[str, str]  # service, message
    skip_kind: str
    swap: tuple[str, str]
    trunc_kind: str
    storm_kind: str


FLOWS: dict[str, FlowSpec] = {
    "mortgage_application": FlowSpec(
        label="mortgage application",
        steps=(
            (
                "a1",
                "portal-bff",
                "INFO",
                "application submitted",
                "loan application submitted application_id={app} customer_segment={seg}",
            ),
            (
                "a2",
                "identity-service",
                "INFO",
                "MitID validation",
                "MitID session validated customer_id={cust}",
            ),
            (
                "a3",
                "customer-service",
                "INFO",
                "KYC profile load",
                "KYC profile loaded customer_id={cust} risk_class={risk}",
            ),
            (
                "a4",
                "valuation-service",
                "INFO",
                "property valuation request",
                "property valuation requested property_id={prop} provider={provider}",
            ),
            (
                "a5",
                "valuation-service",
                "INFO",
                "property valuation",
                "property valuation received property_id={prop} value_dkk={value} latency_ms={lat}",
            ),
            (
                "a6",
                "credit-service",
                "INFO",
                "credit assessment",
                "credit assessment completed application_id={app} score={score} decision=APPROVED",
            ),
            (
                "a7",
                "offer-service",
                "INFO",
                "loan offer generation",
                "loan offer generated application_id={app} loan_type={loan_type}"
                " principal_dkk={principal}",
            ),
            (
                "a8",
                "document-service",
                "INFO",
                "offer document storage",
                "offer document stored application_id={app} doc_id={doc}",
            ),
            (
                "a9",
                "portal-bff",
                "INFO",
                "application completion",
                "application completed application_id={app} status=OFFER_ISSUED",
            ),
        ),
        err_kind="a5",
        err=(
            "valuation-service",
            "property valuation failed property_id={prop} provider={provider}"
            " reason=PROVIDER_TIMEOUT after_ms=10000",
        ),
        skip_kind="a6",
        swap=("a6", "a7"),
        trunc_kind="a5",
        storm_kind="a4",
    ),
    "instalment_collection": FlowSpec(
        label="instalment collection",
        steps=(
            (
                "b1",
                "payment-scheduler",
                "INFO",
                "collection start",
                "instalment collection started loan_id={loan} due_date={due}",
            ),
            (
                "b2",
                "payment-service",
                "INFO",
                "direct debit mandate check",
                "direct debit mandate verified loan_id={loan} mandate_id={mandate}",
            ),
            (
                "b3",
                "payment-service",
                "INFO",
                "payment instruction",
                "payment instruction sent loan_id={loan} amount_dkk={amount}",
            ),
            (
                "b4",
                "ledger-service",
                "INFO",
                "ledger posting",
                "ledger posting accepted loan_id={loan} amount_dkk={amount} posting_id={post}",
            ),
            (
                "b5",
                "payment-service",
                "INFO",
                "payment settlement",
                "payment settled loan_id={loan} amount_dkk={amount}",
            ),
            (
                "b6",
                "notification-service",
                "INFO",
                "payment receipt",
                "payment receipt sent loan_id={loan} channel={channel}",
            ),
        ),
        err_kind="b4",
        err=("ledger-service", "ledger posting rejected loan_id={loan} reason=QUEUE_TIMEOUT"),
        skip_kind="b4",
        swap=("b5", "b6"),
        trunc_kind="b3",
        storm_kind="b3",
    ),
    "loan_conversion_quote": FlowSpec(
        label="loan conversion quote",
        steps=(
            (
                "c1",
                "portal-bff",
                "INFO",
                "quote request",
                "conversion quote requested loan_id={loan} target_product={product}",
            ),
            (
                "c2",
                "identity-service",
                "INFO",
                "MitID validation",
                "MitID session validated customer_id={cust}",
            ),
            (
                "c3",
                "pricing-service",
                "INFO",
                "bond price fetch",
                "bond price fetched series={series} price={price}",
            ),
            (
                "c4",
                "pricing-service",
                "INFO",
                "conversion cost calculation",
                "conversion cost calculated loan_id={loan} cost_dkk={cost}",
            ),
            (
                "c5",
                "quote-service",
                "INFO",
                "quote issue",
                "conversion quote issued quote_id={quote} loan_id={loan}",
            ),
        ),
        err_kind="c3",
        err=("pricing-service", "bond price feed unavailable series={series}"),
        skip_kind="c4",
        swap=("c3", "c4"),
        trunc_kind="c3",
        storm_kind="c3",
    ),
    "disbursement": FlowSpec(
        label="disbursement",
        steps=(
            (
                "d1",
                "offer-service",
                "INFO",
                "offer acceptance",
                "offer accepted application_id={app}",
            ),
            (
                "d2",
                "ledger-service",
                "INFO",
                "funds reservation",
                "funds reserved application_id={app} amount_dkk={principal}",
            ),
            (
                "d3",
                "payment-service",
                "INFO",
                "payout instruction",
                "payout instruction sent application_id={app} amount_dkk={principal}",
            ),
            (
                "d4",
                "ledger-service",
                "INFO",
                "payout booking",
                "payout booked application_id={app} posting_id={post}",
            ),
            (
                "d5",
                "notification-service",
                "INFO",
                "payout confirmation",
                "payout confirmation sent application_id={app}",
            ),
        ),
        err_kind="d2",
        err=(
            "ledger-service",
            "funds reservation failed application_id={app} reason=INSUFFICIENT_LIMIT",
        ),
        skip_kind="d2",
        swap=("d3", "d4"),
        trunc_kind="d3",
        storm_kind="d3",
    ),
}

LABELS = {s[0]: s[3] for spec in FLOWS.values() for s in spec.steps}
LABELS.update({"a4r": "valuation retry", "a45": "cached valuation"})

# Optional, harmless variations. Real flows have these, so a detector must not flag them.
OPTIONAL_LINES = {
    "a2r": ("identity-service", "WARN", "MitID token refresh attempt=1"),
    "a4r": (
        "valuation-service",
        "WARN",
        "valuation provider slow, retrying property_id={prop} attempt=1",
    ),
    "a45": (
        "valuation-service",
        "INFO",
        "property valuation served from cache property_id={prop} value_dkk={value}",
    ),
    "b3r": ("ledger-service", "WARN", "ledger posting retried loan_id={loan} attempt=1"),
}


OPTIONAL_BY_FLOW = {
    "mortgage_application": ["a2r", "a4r", "a45"],
    "instalment_collection": ["b3r"],
}


def _hex(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("0123456789abcdef") for _ in range(n))


def _digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("0123456789") for _ in range(n))


def _vals(rng: random.Random, degraded: bool) -> dict[str, str]:
    return {
        "app": f"APP-2026-{_digits(rng, 6)}",
        "cust": f"C{_digits(rng, 8)}",
        "prop": f"P{_digits(rng, 7)}",
        "loan": f"LN{_digits(rng, 9)}",
        "quote": f"Q{_digits(rng, 8)}",
        "doc": f"DOC-{_hex(rng, 8)}",
        "post": f"PST-{_hex(rng, 10)}",
        "mandate": f"MD{_digits(rng, 7)}",
        "seg": rng.choice(["RETAIL", "RETAIL", "PRIVATE", "SME"]),
        "risk": rng.choice(["LOW", "LOW", "MEDIUM", "HIGH"]),
        "provider": rng.choice(["valuta-ejd", "bolig-vurdering"]),
        "value": str(rng.randrange(600_000, 9_000_000, 10_000)),
        "principal": str(rng.randrange(400_000, 6_000_000, 5_000)),
        "score": str(rng.randint(420, 880)),
        "loan_type": rng.choice(["FIXED_30Y", "FLEX_1Y", "FLEX_5Y", "FIXED_20Y_IO"]),
        "lat": str(rng.randint(2500, 6500) if degraded else rng.randint(180, 950)),
        "amount": f"{rng.uniform(2500, 18000):.2f}",
        "due": f"2026-03-{rng.randint(1, 28):02d}",
        "channel": rng.choice(["DIGITAL_POST", "EMAIL", "DIGITAL_POST"]),
        "product": rng.choice(["FIXED_30Y", "FLEX_5Y", "FLEX_1Y"]),
        "series": f"BND{rng.choice(['2.5', '3.0', '4.0'])}-{rng.choice(['2053', '2056'])}",
        "price": f"{rng.uniform(88, 102):.3f}",
        "cost": f"{rng.uniform(4000, 90000):.2f}",
    }


def normal_lines(
    flow: str, rng: random.Random, cfg: dict, v: dict[str, str], degraded: bool, variants: bool
) -> list[Line]:
    lines = [Line(k, svc, lvl, msg.format(**v)) for k, svc, lvl, _, msg in FLOWS[flow].steps]

    def optional(kind: str) -> Line:
        svc, lvl, msg = OPTIONAL_LINES[kind]
        return Line(kind, svc, lvl, msg.format(**v))

    def insert_after(kind: str, new: Line) -> None:
        lines.insert([ln.kind for ln in lines].index(kind) + 1, new)

    f = cfg["flows"]
    if flow == "mortgage_application":
        if rng.random() < f["mitid_retry_prob"]:
            lines.insert(1, optional("a2r"))
        if degraded and rng.random() < f["degraded_slow_prob"]:
            insert_after("a4", optional("a4r"))
        if variants and rng.random() < f["valuation_cache_prob"]:
            i = [ln.kind for ln in lines].index("a4")
            lines[i : i + 2] = [optional("a45")]
    elif flow == "instalment_collection" and rng.random() < f["ledger_retry_prob"]:
        insert_after("b3", optional("b3r"))
    return lines


def inject(
    flow: str, anomaly: str, normal: list[Line], v: dict[str, str], rng: random.Random
) -> list[Line]:
    spec = FLOWS[flow]
    kinds = [ln.kind for ln in normal]
    out = list(normal)
    if anomaly == "visible_error":
        svc, msg = spec.err
        out[kinds.index(spec.err_kind)] = Line(spec.err_kind + "!", svc, "ERROR", msg.format(**v))
    elif anomaly == "silent_skip":
        del out[kinds.index(spec.skip_kind)]
    elif anomaly == "wrong_order":
        i, j = kinds.index(spec.swap[0]), kinds.index(spec.swap[1])
        out[i], out[j] = out[j], out[i]
    elif anomaly == "truncated":
        out = out[: kinds.index(spec.trunc_kind) + 1]
    elif anomaly == "retry_storm":
        i = kinds.index(spec.storm_kind)
        out = out[:i] + [normal[i]] * rng.randint(4, 6) + out[i + 1 :]
    else:
        raise ValueError(f"unknown anomaly type: {anomaly}")
    return out


def first_deviation(normal: list[Line], actual: list[Line]) -> int:
    for i, (a, b) in enumerate(zip(normal, actual, strict=False)):
        if a.kind != b.kind:
            return i
    return min(len(normal), len(actual))


def explain(flow: str, anomaly: str | None, kinds: list[str]) -> str:
    spec = FLOWS[flow]
    if anomaly is None:
        return f"Complete {spec.label} flow: every expected step occurred in order."
    if anomaly == "visible_error":
        return (
            f"The {LABELS[spec.err_kind]} step failed with an explicit ERROR from "
            f"{spec.err[0]}, so the {spec.label} did not run as expected."
        )
    if anomaly == "silent_skip":
        return (
            f"Silent flow break: the {LABELS[spec.skip_kind]} step never happened, yet the "
            f"{spec.label} carried on and no error was logged."
        )
    if anomaly == "wrong_order":
        return (
            f"Steps out of order: {LABELS[spec.swap[1]]} was logged before "
            f"{LABELS[spec.swap[0]]}, which breaks the expected {spec.label} sequence."
        )
    if anomaly == "truncated":
        return (
            f"The {spec.label} stopped after {LABELS[spec.trunc_kind]}; the remaining steps "
            "never occurred and no error was logged."
        )
    n = kinds.count(spec.storm_kind)
    return (
        f"Retry storm: {LABELS[spec.storm_kind]} was repeated {n} times without the "
        f"{spec.label} progressing."
    )


def _tz(cfg: dict) -> timezone:
    return timezone(timedelta(hours=cfg["utc_offset_hours"]))


def _fmt_ts(t: datetime) -> str:
    offset = t.strftime("%z")
    return (
        t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}{offset[:3]}:{offset[3:]}"
    )


def _incident_for(cfg: dict, flow: str, t: datetime) -> tuple[dict | None, str]:
    for inc in cfg["incidents"]:
        if inc["flow"] != flow:
            continue
        if datetime.fromisoformat(inc["start"]) <= t <= datetime.fromisoformat(inc["end"]):
            return inc, "incident"
        if (
            datetime.fromisoformat(inc["precursor_start"])
            <= t
            < datetime.fromisoformat(inc["start"])
        ):
            return inc, "precursor"
    return None, "background"


def _pick_anomaly(rng: random.Random, cfg: dict, flow: str, t: datetime):
    inc, phase = _incident_for(cfg, flow, t)
    if phase == "incident":
        if rng.random() < inc["affected_share"]:
            w = inc["anomaly_weights"]
            return rng.choices(list(w), weights=list(w.values()))[0], inc["id"], phase
        return None, inc["id"], phase
    inc_id = inc["id"] if inc else ""
    if rng.random() < cfg["background_anomaly_rate"]:
        return rng.choice(ANOMALY_TYPES), inc_id, phase
    return None, inc_id, phase


def _plan_day(rng: random.Random, cfg: dict, day: datetime) -> list[tuple[datetime, str]]:
    weekend = day.weekday() >= 5
    plan = []
    for flow, tr in cfg["traffic"].items():
        n = tr["weekend"] if weekend else tr["weekday"]
        n = max(0, round(rng.gauss(n, n * 0.08))) if n else 0
        lo, hi, mode = tr["window"]
        for inc in cfg["incidents"]:
            end = datetime.fromisoformat(inc["end"])
            if inc["flow"] == flow and inc.get("stretch_window") and end.date() == day.date():
                hi = (end - day).total_seconds() / 3600
        for _ in range(n):
            plan.append((day + timedelta(hours=rng.triangular(lo, hi, mode)), flow))
    plan.sort()
    return plan


POD_SETS = ["7c9d5b", "5b8f6c", "a41e92"]
POD_IDS = ["x2k4p", "p9qmz", "t6zvn"]


def _pod(rng: random.Random, service: str) -> str:
    return f"{service}-{rng.choice(POD_SETS)}-{rng.choice(POD_IDS)}"


def _noise(cfg: dict, day: datetime, rng: random.Random) -> list[tuple[datetime, str]]:
    out: list[tuple[datetime, str]] = []

    def add(t: datetime, service: str, level: str, msg: str) -> None:
        out.append(
            (t, f"{_fmt_ts(t)} {level} {service} host={_pod(rng, service)} corrId={NO_CORR} {msg}")
        )

    for k in range(0, 24 * 60, 5):
        t = day + timedelta(minutes=k, milliseconds=rng.randint(0, 900))
        add(t, "api-gateway", "INFO", f"health probe ok status=200 latency_ms={rng.randint(2, 40)}")
    for k in range(0, 24 * 60, 10):
        t = day + timedelta(minutes=k, milliseconds=rng.randint(0, 900))
        add(t, "payment-scheduler", "INFO", f"scheduler heartbeat lag_ms={rng.randint(0, 120)}")
    for k in range(24):
        t = day + timedelta(hours=k, seconds=rng.randint(0, 59))
        add(
            t, "config-service", "INFO", f"configuration refreshed version=v{rng.randint(100, 140)}"
        )
    for _ in range(20):
        t = day + timedelta(seconds=rng.randint(0, 86399))
        add(t, "valuation-service", "INFO", f"jvm gc pause pause_ms={rng.randint(8, 90)}")
    for inc in cfg["incidents"]:
        nz = inc["noise"]
        t = datetime.fromisoformat(inc["precursor_start"])
        end = datetime.fromisoformat(inc["end"])
        lo, hi = nz["value_range"]
        while t <= end:
            if t.date() == day.date():
                msg = nz["message"].format(p95=rng.randint(lo, hi))
                add(
                    t + timedelta(milliseconds=rng.randint(0, 900)), nz["service"], nz["level"], msg
                )
            t += timedelta(seconds=nz["every_s"])
    return out


def _make_trace(rng, cfg, flow, t0, anomaly, degraded, used):
    while True:
        cid = "c-" + _hex(rng, 16)
        if cid not in used:
            used.add(cid)
            break
    v = _vals(rng, degraded)
    normal = normal_lines(flow, rng, cfg, v, degraded, variants=anomaly is None)
    if anomaly is None:
        lines, dev = normal, None
    else:
        lines = inject(flow, anomaly, normal, v, rng)
        dev = first_deviation(normal, lines)
    events = []
    t = t0
    for ln in lines:
        t += timedelta(milliseconds=rng.uniform(5, 140))
        if ln.kind == "a5":
            t += timedelta(milliseconds=int(v["lat"]))
        host = _pod(rng, ln.service)
        events.append(
            (t, f"{_fmt_ts(t)} {ln.level} {ln.service} host={host} corrId={cid} {ln.message}")
        )
    return cid, [ln.kind for ln in lines], events, dev


def parse_line(line: str) -> tuple[str, str, str]:
    """Return (corr_id, drain content, timestamp). Content is 'service LEVEL message'."""
    m = LINE_RE.match(line)
    if not m:
        raise ValueError(f"unparseable line: {line!r}")
    ts, level, service, _host, cid, msg = m.groups()
    return cid, f"{service} {level} {msg}", ts


def _miner() -> TemplateMiner:
    cfg = TemplateMinerConfig()
    cfg.drain_sim_th = 0.9
    cfg.drain_depth = 4
    cfg.profiling_enabled = False
    cfg.masking_instructions = [MaskingInstruction(r"(?<==)\S+", "*")]
    return TemplateMiner(config=cfg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _select_labelled(records: list[dict], spec: dict, rng: random.Random) -> list[dict]:
    anomalous = [r for r in records if r["label"] == 1]
    normal = [r for r in records if r["label"] == 0]
    chosen: list[dict] = []
    for key, quota in spec["anomalous_quota"].items():
        pool = [
            r
            for r in anomalous
            if (r["incident_id"] if r["phase"] == "incident" else "background") == key
        ]
        rng.shuffle(pool)
        chosen += pool[:quota]
    rest = [r for r in anomalous if r not in chosen]
    rng.shuffle(rest)
    chosen += rest[: spec["anomalous"] - len(chosen)]

    hard = [r for r in normal if r["phase"] == "precursor" and "a4r" in r["kinds"]]
    rng.shuffle(hard)
    picked = hard[: spec["hard_negatives"]]
    by_flow: dict[str, list[dict]] = defaultdict(list)
    for r in normal:
        if r not in picked:
            by_flow[r["flow"]].append(r)
    for pool in by_flow.values():
        rng.shuffle(pool)
    flows = sorted(by_flow)
    i = 0
    while len(picked) < spec["normal"]:
        pool = by_flow[flows[i % len(flows)]]
        if pool:
            picked.append(pool.pop())
        i += 1
    return sorted(chosen + picked, key=lambda r: r["start"])


def write_splits(data_dir: Path, val_share: float = 0.1) -> dict[str, int]:
    """Write raw_<split>.log and labels_test.csv in the layout `logpoc prepare` reads.

    Uses only what a real extract has: logs/, traces_labelled.jsonl and incidents.csv.
    test is the labelled set. train and val are all other traces that started outside the
    incident windows: unlabelled and mostly normal, like real production data.
    """
    lines: dict[str, list[str]] = defaultdict(list)
    for path in sorted((data_dir / "logs").glob("*.log")):
        for line in path.read_text(encoding="utf-8").splitlines():
            ts, level, service, _host, cid, msg = LINE_RE.match(line).groups()
            if cid != NO_CORR:
                lines[cid].append(f"{ts} {level} {service} trace={cid} {msg}")
    text = (data_dir / "traces_labelled.jsonl").read_text(encoding="utf-8")
    labelled = [json.loads(x) for x in text.splitlines() if x]
    with (data_dir / "incidents.csv").open(encoding="utf-8") as f:
        windows = [
            (datetime.fromisoformat(r["precursor_start"]), datetime.fromisoformat(r["end"]))
            for r in csv.DictReader(f)
        ]

    def in_incident(cid: str) -> bool:
        t = datetime.fromisoformat(lines[cid][0].split(" ", 1)[0])
        return any(a <= t <= b for a, b in windows)

    test = [r["trace_id"] for r in labelled]
    pool = sorted(c for c in lines if c not in set(test) and not in_incident(c))
    random.Random(0).shuffle(pool)
    n_val = round(len(pool) * val_share)
    splits = {"train": pool[n_val:], "val": pool[:n_val], "dev": [], "fewshot": [], "test": test}
    for name, ids in splits.items():
        body = "".join(ln + "\n" for c in ids for ln in lines[c])
        (data_dir / f"raw_{name}.log").write_text(body, encoding="utf-8", newline="\n")
    _write_csv(
        data_dir / "labels_test.csv",
        ["trace_id", "label", "anomaly_type", "first_deviation_index"],
        [
            {
                "trace_id": r["trace_id"],
                "label": int(r["expected_verdict"] == "anomalous"),
                "anomaly_type": r["anomaly_type"],
                "first_deviation_index": r["first_deviation_index"],
            }
            for r in labelled
        ],
    )
    return {k: len(v) for k, v in splits.items()}


def generate_all(config_path: Path, out_dir: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text())
    tz = _tz(cfg)
    first = datetime.fromisoformat(cfg["start_date"]).replace(tzinfo=tz)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    used: set[str] = set()
    records: list[dict] = []
    files: dict[str, str] = {}
    n_lines = 0
    for d in range(cfg["days"]):
        day = first + timedelta(days=d)
        rng = random.Random(f"{cfg['seed']}:{day.date()}")
        events: list[tuple[datetime, str]] = []
        for t0, flow in _plan_day(rng, cfg, day):
            anomaly, inc_id, phase = _pick_anomaly(rng, cfg, flow, t0)
            cid, kinds, ev, dev = _make_trace(
                rng, cfg, flow, t0, anomaly, phase in ("precursor", "incident"), used
            )
            events += ev
            records.append(
                {
                    "trace_id": cid,
                    "flow": flow,
                    "start": _fmt_ts(ev[0][0]),
                    "label": 0 if anomaly is None else 1,
                    "anomaly_type": anomaly or "none",
                    "first_deviation_index": "" if dev is None else dev,
                    "phase": phase,
                    "incident_id": inc_id,
                    "kinds": kinds,
                }
            )
        events += _noise(cfg, day, random.Random(f"{cfg['seed']}:{day.date()}:noise"))
        events.sort(key=lambda e: e[0])
        path = log_dir / f"prod_{day.date()}.log"
        path.write_text("\n".join(e[1] for e in events) + "\n", newline="\n", encoding="utf-8")
        files[f"logs/{path.name}"] = sha256_file(path)
        n_lines += len(events)

    # Drain pass over the written logs, the same way a customer pipeline would run it.
    miner = _miner()
    seqs: dict[str, list[int]] = defaultdict(list)
    raw_by_trace: dict[str, list[str]] = defaultdict(list)
    for path in sorted(log_dir.glob("prod_*.log")):
        for line in path.read_text(encoding="utf-8").splitlines():
            cid, content, _ = parse_line(line)
            res = miner.add_log_message(content)
            if cid != NO_CORR:
                seqs[cid].append(res["cluster_id"])
                raw_by_trace[cid].append(line)
    clusters = sorted(miner.drain.clusters, key=lambda c: c.cluster_id)
    _write_csv(
        out_dir / "drain_templates.csv",
        ["template_id", "count", "template"],
        [
            {"template_id": f"T{c.cluster_id:03d}", "count": c.size, "template": c.get_template()}
            for c in clusters
        ],
    )

    (out_dir / "truth").mkdir(exist_ok=True)
    _write_csv(
        out_dir / "truth" / "trace_truth.csv",
        [
            "trace_id",
            "flow",
            "start",
            "label",
            "anomaly_type",
            "first_deviation_index",
            "phase",
            "incident_id",
        ],
        [{k: v for k, v in r.items() if k != "kinds"} for r in records],
    )

    labelled = _select_labelled(
        records, cfg["labelled_set"], random.Random(f"{cfg['seed']}:labels")
    )
    with (out_dir / "traces_labelled.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in labelled:
            row = {
                "trace_id": r["trace_id"],
                "flow": r["flow"],
                "start": r["start"],
                "template_ids": [f"T{i:03d}" for i in seqs[r["trace_id"]]],
                "expected_verdict": "anomalous" if r["label"] else "normal",
                "anomaly_type": r["anomaly_type"],
                "first_deviation_index": r["first_deviation_index"],
                "explanation": explain(
                    r["flow"], None if not r["label"] else r["anomaly_type"], r["kinds"]
                ),
                "raw_lines": raw_by_trace[r["trace_id"]],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    incidents = [
        {k: inc[k] for k in ("id", "title", "flow", "precursor_start", "start", "end")}
        for inc in cfg["incidents"]
    ]
    _write_csv(
        out_dir / "incidents.csv",
        ["id", "title", "flow", "precursor_start", "start", "end"],
        incidents,
    )
    flows_doc = {
        name: {
            "description": spec.label,
            "expected_steps": [
                {"step": s[0], "service": s[1], "level": s[2], "what": s[3], "message": s[4]}
                for s in spec.steps
            ],
            "optional_harmless": {
                k: {
                    "service": OPTIONAL_LINES[k][0],
                    "level": OPTIONAL_LINES[k][1],
                    "message": OPTIONAL_LINES[k][2],
                }
                for k in OPTIONAL_BY_FLOW.get(name, [])
            },
        }
        for name, spec in FLOWS.items()
    }
    (out_dir / "flows.yaml").write_text(yaml.safe_dump(flows_doc, sort_keys=False, width=100))
    splits = write_splits(out_dir)

    for name in (
        "drain_templates.csv",
        "traces_labelled.jsonl",
        "incidents.csv",
        "flows.yaml",
        "truth/trace_truth.csv",
        "labels_test.csv",
        *(f"raw_{s}.log" for s in splits),
    ):
        files[name] = sha256_file(out_dir / name)
    counts = Counter(r["flow"] for r in records)
    manifest = {
        "generator_version": cfg["generator_version"],
        "seed": cfg["seed"],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "days": cfg["days"],
        "lines": n_lines,
        "traces": len(records),
        "traces_by_flow": dict(sorted(counts.items())),
        "anomalous_traces": sum(r["label"] for r in records),
        "templates": len(clusters),
        "labelled": {
            "traces": len(labelled),
            "anomalous": sum(r["label"] for r in labelled),
        },
        "splits": splits,
        "files": files,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", newline="\n")
    return manifest
