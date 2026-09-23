import sys

from .cli import parse_run_matrix
from .runner import run_benchmark


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for src.run module."""
    runs = parse_run_matrix(argv)
    for args in runs:
        run_benchmark(args)


if __name__ == "__main__":
    main(sys.argv[1:])
