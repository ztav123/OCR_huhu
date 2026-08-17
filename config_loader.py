"""Project root config_loader shim. Re-exports from src.config_loader."""
from src.config_loader import load_config, get, reset_cache  # noqa: F401

__all__ = ["load_config", "get", "reset_cache"]
