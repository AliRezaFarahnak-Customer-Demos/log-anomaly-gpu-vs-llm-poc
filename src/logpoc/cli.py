"""Command line entry point. Heavy imports stay inside each command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_generate(a: argparse.Namespace) -> int:
    from logpoc.data.generate import generate_all

    manifest = generate_all(Path(a.config), Path(a.out))
    for name, c in manifest["counts"].items():
        print(f"{name}: {c['traces']} traces ({c['normal']} normal, {c['anomalous']} anomalous)")
    print(f"wrote {a.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="logpoc", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate synthetic logs")
    g.add_argument("--config", default="configs/data_v1.yaml")
    g.add_argument("--out", default="data/synthetic/v1")
    g.set_defaults(func=_cmd_generate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
