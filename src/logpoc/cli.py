"""Command line entry point. Heavy imports stay inside each command."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_DATA = "data/synthetic/v1"


def _data_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--data",
        default=os.environ.get("DATA_DIR", DEFAULT_DATA),
        help="folder with raw logs, labels and manifest (env DATA_DIR)",
    )


def _cmd_generate(a: argparse.Namespace) -> int:
    from logpoc.data.generate import generate_all

    manifest = generate_all(Path(a.config), Path(a.out))
    for name, c in manifest["counts"].items():
        print(f"{name}: {c['traces']} traces ({c['normal']} normal, {c['anomalous']} anomalous)")
    print(f"wrote {a.out}")
    return 0


def _cmd_prepare(a: argparse.Namespace) -> int:
    from logpoc.data.prepare import prepare

    out = prepare(Path(a.data))
    print(f"wrote {out}")
    return 0


def _cmd_download_model(a: argparse.Namespace) -> int:
    from logpoc.poc1_gpu.download_model import download

    download(a.repo_id, Path(a.out))
    return 0


def _cmd_train(a: argparse.Namespace) -> int:
    from logpoc.poc1_gpu.train import run

    config = a.config or os.environ.get("CONFIG")
    if not config:
        print("error: pass --config or set CONFIG", file=sys.stderr)
        return 2
    return run(Path(config), Path(a.data))


def _cmd_evaluate(a: argparse.Namespace) -> int:
    from logpoc.poc1_gpu.evaluate import run

    run_id = a.run_id or os.environ.get("RUN_ID")
    if not run_id:
        print("error: pass --run-id or set RUN_ID", file=sys.stderr)
        return 2
    run(run_id, Path(a.data), a.split, a.device, a.percentile, a.batch_size)
    return 0


def _cmd_poc2(a: argparse.Namespace) -> int:
    from logpoc.poc2_llm.classify import run

    return run(
        data_dir=Path(a.data),
        split=a.split,
        context=a.context,
        limit=a.limit,
        only_flagged_by=a.only_flagged_by,
        run_id=a.run_id,
        dry_run=a.dry_run,
        concurrency=a.concurrency,
    )


def _cmd_baseline(a: argparse.Namespace) -> int:
    from logpoc.baselines.grep_rules import run

    payload = run(Path(a.data), a.split, a.run_id)
    m = payload["metrics"]
    print(f"grep {a.split}: precision={m['precision']} recall={m['recall']} f1={m['f1']}")
    return 0


def _cmd_compare(a: argparse.Namespace) -> int:
    from logpoc.compare import run

    return run(a.pricing)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="logpoc", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate synthetic logs")
    g.add_argument("--config", default="configs/data_v1.yaml")
    g.add_argument("--out", default=DEFAULT_DATA)
    g.set_defaults(func=_cmd_generate)

    pr = sub.add_parser("prepare", help="parse raw logs into template sequences")
    _data_arg(pr)
    pr.set_defaults(func=_cmd_prepare)

    d = sub.add_parser("download-model", help="download base model files (needs internet)")
    d.add_argument("--repo-id", required=True)
    d.add_argument("--out", required=True)
    d.set_defaults(func=_cmd_download_model)

    t = sub.add_parser("train", help="PoC 1: LoRA fine-tune on normal traces")
    t.add_argument("--config", default=None, help="yaml config (env CONFIG)")
    _data_arg(t)
    t.set_defaults(func=_cmd_train)

    e = sub.add_parser("evaluate", help="PoC 1: score a split with a trained adapter")
    e.add_argument("--run-id", default=None, help="training run id (env RUN_ID)")
    e.add_argument("--split", default="test", choices=["dev", "test"])
    e.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    e.add_argument("--percentile", type=float, default=None, help="override config percentile")
    e.add_argument("--batch-size", type=int, default=8)
    _data_arg(e)
    e.set_defaults(func=_cmd_evaluate)

    q = sub.add_parser("poc2-llm", help="PoC 2: structured LLM calls")
    q.add_argument("--split", default="dev", choices=["dev", "test"])
    q.add_argument("--context", default="none", choices=["none", "expected-flow", "few-shot"])
    q.add_argument("--limit", type=int, default=None)
    q.add_argument("--only-flagged-by", default=None, metavar="RUN_ID")
    q.add_argument("--run-id", default=None)
    q.add_argument("--dry-run", action="store_true", help="use the offline fake client")
    q.add_argument("--concurrency", type=int, default=4)
    _data_arg(q)
    q.set_defaults(func=_cmd_poc2)

    b = sub.add_parser("baseline-grep", help="keyword baseline")
    b.add_argument("--split", default="test", choices=["dev", "test"])
    b.add_argument("--run-id", default=None)
    _data_arg(b)
    b.set_defaults(func=_cmd_baseline)

    c = sub.add_parser("compare", help="write runs/RUNS.md from every eval metrics.json")
    c.add_argument("--pricing", default="configs/pricing.yaml")
    c.set_defaults(func=_cmd_compare)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
