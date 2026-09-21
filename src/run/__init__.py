from .cli import build_parser, parse_args
from .runner import run_benchmark


def main(argv=None) -> None:
    from .__main__ import main as _main

    _main(argv)


__all__ = ["build_parser", "main", "parse_args", "run_benchmark"]
