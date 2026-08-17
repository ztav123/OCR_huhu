"""Load YAML config and expose to all modules."""
from pathlib import Path
import yaml


_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
_CONFIG_CACHE: dict | None = None


def load_config(path: str | Path | None = None) -> dict:
    """Load and cache config.yaml."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and path is None:
        return _CONFIG_CACHE
    cfg_path = Path(path) if path else _CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if path is None:
        _CONFIG_CACHE = cfg
    return cfg


def get(key_path: str, default=None):
    """Get nested config by dot-path, e.g. 'preprocess.max_edge_px'."""
    cfg = load_config()
    keys = key_path.split(".")
    cur = cfg
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def reset_cache() -> None:
    """Reset config cache (for tests)."""
    global _CONFIG_CACHE
    _CONFIG_CACHE = None
