from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

try:
    import yaml  # PyYAML
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False


def load_config(path: Path | None) -> Dict[str, Any]:
    """Load a YAML or JSON config file if provided.

    Returns an empty dict if path is None or invalid.
    """
    if not path:
        return {}

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    # Prefer YAML if available and extension matches
    if _HAS_YAML and path.suffix.lower() in {".yml", ".yaml"}:
        with open(path, "r", encoding="utf-8") as f:
            return dict(yaml.safe_load(f) or {})

    # Fallback to JSON if YAML unavailable or using .json file
    import json
    with open(path, "r", encoding="utf-8") as f:
        return dict(json.load(f))
