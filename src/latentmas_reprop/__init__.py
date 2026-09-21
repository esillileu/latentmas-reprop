"""LatentMAS reproduction library package."""

__version__ = "0.1.0"


def main(argv=None) -> None:
    """CLI entrypoint lazy import to prevent circular import on module load."""
    try:
        from src.run.main import main as _main
    except ModuleNotFoundError:
        from run.main import main as _main
    _main(argv)


__all__ = ["__version__", "main"]
