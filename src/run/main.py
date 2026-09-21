import sys

from .cli import parse_args
from .runner import run_benchmark


def main(argv: list[str] | None = None) -> None:
    """Main CLI entrypoint for running benchmarks."""
    args = parse_args(argv)
    run_benchmark(args)


if __name__ == "__main__":
    main(sys.argv[1:])
