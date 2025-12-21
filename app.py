"""Convenience entrypoint for the Animals project.

Usage:
  - Run web app:           python app.py
  - Prepare dataset:       python app.py prepare
  - Train model:           python app.py train
  - Train (skip prepare):  python app.py train --skip-prepare

This wraps the existing pipeline in `src/` and keeps the default flow simple.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))


def _prepare() -> None:
    from src.data_preparation.prepare import prepare_dataset

    prepare_dataset()


def _train(skip_prepare: bool) -> None:
    from src.training.train import train_model

    if not skip_prepare:
        _prepare()
    train_model()


def _run_app(host: str, port: int, debug: bool) -> int:
    from src.webapp.app import app as flask_app
    from src.webapp.app import initialize_app

    ok = initialize_app()
    if not ok:
        # initialize_app already prints a helpful message.
        return 1

    flask_app.run(host=host, port=port, debug=debug)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Animals project entrypoint",
        epilog="Examples: python app.py | python app.py --debug | python app.py train",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=["prepare", "train", "run"],
        help="Action to run (default: run)",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="(train only) Skip dataset preparation step",
    )
    parser.add_argument("--host", default="0.0.0.0", help="(run only) Host to bind")
    parser.add_argument("--port", type=int, default=5000, help="(run only) Port to bind")
    parser.add_argument("--debug", action="store_true", help="(run only) Enable Flask debug")

    args = parser.parse_args(argv)
    command = args.command

    if command == "prepare":
        _prepare()
        return 0

    if command == "train":
        _train(skip_prepare=bool(args.skip_prepare))
        return 0

    if command == "run":
        return _run_app(host=args.host, port=args.port, debug=bool(args.debug))

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
