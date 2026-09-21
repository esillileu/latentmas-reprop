import sys

from .cli import parse_args
from .runner import run_benchmark


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for src.run module."""
    args = parse_args(argv)
    run_benchmark(args)


if __name__ == "__main__":
    main(sys.argv[1:])
