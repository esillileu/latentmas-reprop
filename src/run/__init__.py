def main(argv=None) -> None:
    """Run the benchmark CLI without eagerly importing model infrastructure."""
    from .__main__ import main as _main

    _main(argv)


__all__ = ["main"]
