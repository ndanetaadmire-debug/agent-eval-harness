"""evalh run cases.json --agent package.module:factory [--baseline base.json] [--out report.json] [--md report.md]"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

from evalh.model import Case
from evalh.report import compare, summarize, to_markdown
from evalh.runner import run_suite


def load_factory(spec: str):  # type: ignore[no-untyped-def]
    """Import `module:attr`, resolving modules relative to the current directory (like uvicorn and pytest),
    since the installed `evalh` script's own directory is what Python puts on sys.path."""
    module, _, attr = spec.partition(":")
    if not module or not attr:
        raise SystemExit(f"--agent must look like package.module:factory, got {spec!r}")
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    try:
        return getattr(importlib.import_module(module), attr)
    except (ImportError, AttributeError) as exc:
        raise SystemExit(f"could not load agent factory {spec!r}: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="evalh")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("cases", type=Path)
    r.add_argument("--agent", required=True, help="module:factory returning an Agent")
    r.add_argument("--baseline", type=Path)
    r.add_argument("--out", type=Path)
    r.add_argument("--md", type=Path, help="write a Markdown summary (e.g. $GITHUB_STEP_SUMMARY)")
    r.add_argument("--tolerance", type=float, default=0.02)
    r.add_argument("--min-pass-rate", type=float, default=0.0)
    r.add_argument("--tags", nargs="*", help="only run cases with any of these tags")
    a = p.parse_args(argv)

    cases = [Case.from_dict(d) for d in json.loads(a.cases.read_text())]
    if a.tags:
        cases = [c for c in cases if set(c.tags) & set(a.tags)]
    summary = summarize(run_suite(load_factory(a.agent), cases))
    baseline = json.loads(a.baseline.read_text()) if a.baseline and a.baseline.exists() else None
    gate = compare(summary, baseline, tolerance=a.tolerance, min_pass_rate=a.min_pass_rate)
    md = to_markdown(summary, gate)
    print(md)
    if a.out:
        a.out.write_text(summary.to_json())
    if a.md:
        with a.md.open("a") as f:
            f.write(md)
    return 0 if gate.ok else 1


if __name__ == "__main__":
    sys.exit(main())
