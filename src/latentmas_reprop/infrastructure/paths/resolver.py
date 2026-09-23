import os
import subprocess
from pathlib import Path
from typing import Final


def get_git_commit_hash(repo_dir: Path | None = None) -> str:
    """Safely obtain current git commit hash, or return 'unknown'."""
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        cwd = str(repo_dir) if repo_dir else None
        res = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"


class PathResolver:
    """Explicit and centralized resolver for all non-import file paths.

    Provides canonical access to repository root, cache directories,
    data files, logs, and assets without hardcoding relative paths.
    """

    def __init__(self, root_dir: Path | str | None = None) -> None:
        if root_dir is not None:
            self._root = Path(root_dir).resolve()
        else:
            self._root = self._find_project_root()

    @staticmethod
    def _find_project_root(start_path: Path | None = None) -> Path:
        """Find the project root by looking for pyproject.toml or .git upwards."""
        env_root = os.environ.get("LATENTMAS_PROJECT_ROOT")
        if env_root:
            return Path(env_root).resolve()

        curr = (start_path or Path(__file__)).resolve()
        for parent in [curr, *list(curr.parents)]:
            if (parent / "pyproject.toml").is_file() or (parent / ".git").is_dir():
                return parent
        return Path.cwd().resolve()

    @property
    def root(self) -> Path:
        """Repository root path."""
        return self._root

    def get_git_commit_hash(self) -> str:
        """Git commit hash of repository."""
        return get_git_commit_hash(self._root)

    @property
    def cache_dir(self) -> Path:
        """Repository root .cache directory."""
        path = self._root / ".cache"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_cache_layer_dir(self, layer: str) -> Path:
        """Get or create a hierarchical cache subdirectory under .cache/."""
        path = self.cache_dir / layer
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def data_dir(self) -> Path:
        """Data directory at repo root."""
        return self._root / "data"

    @property
    def medqa_json_path(self) -> Path:
        """Canonical path to data/medqa.json."""
        return self.data_dir / "medqa.json"

    @property
    def logs_dir(self) -> Path:
        """Example logs directory."""
        path = self._root / "example_logs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def assets_dir(self) -> Path:
        """Assets directory."""
        return self._root / "assets"

    @property
    def configs_dir(self) -> Path:
        """YAML configuration directory."""
        path = self._root / "configs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def templates_dir(self) -> Path:
        """Deprecated alias for configs_dir."""
        return self.configs_dir

    def resolve_config_path(self, config_path: str | Path) -> Path:
        """Resolve a configuration path.

        Checks:
        1. Exact path or relative to repo root
        2. Inside configs/ directory
        3. Inside configs/ with .yaml extension
        4. Inside configs/ with .yml extension
        """
        p = Path(config_path)
        if p.is_file():
            return p.resolve()

        resolved = self.resolve(p)
        if resolved.is_file():
            return resolved

        candidate = self.configs_dir / p
        if candidate.is_file():
            return candidate.resolve()

        candidate_yaml = self.configs_dir / f"{p}.yaml"
        if candidate_yaml.is_file():
            return candidate_yaml.resolve()

        candidate_yml = self.configs_dir / f"{p}.yml"
        if candidate_yml.is_file():
            return candidate_yml.resolve()

        return resolved

    def resolve(self, path: str | Path) -> Path:
        """Resolve any relative path against repository root, or return absolute path."""
        p = Path(path)
        if p.is_absolute():
            return p
        return (self._root / p).resolve()


# Default singleton instance
DEFAULT_PATH_RESOLVER: Final[PathResolver] = PathResolver()


def get_path_resolver() -> PathResolver:
    """Return the global default PathResolver instance."""
    return DEFAULT_PATH_RESOLVER
