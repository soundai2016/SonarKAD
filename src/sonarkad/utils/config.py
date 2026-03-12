"""Configuration utilities.

The refactor keeps configuration management intentionally simple:

- Use a single YAML file (merged from the previous multi-file configs).
- Select an experiment block by name (e.g., ``experiments.swellex96_s5_vla`` or
  ``experiments.swellex96_s5_vla``).

No Hydra/OmegaConf dependency is introduced to keep reviewer friction low.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file into a nested dict."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping/dict. Got {type(data)} in {p}")
    return data


def save_yaml(data: Mapping[str, Any], path: str | Path) -> None:
    """Save a nested mapping to YAML."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(data), f, sort_keys=False)


def deep_update(dst: MutableMapping[str, Any], src: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Recursively merge ``src`` into ``dst`` (in-place) and return ``dst``."""
    for k, v in src.items():
        if isinstance(v, Mapping) and isinstance(dst.get(k), Mapping):
            deep_update(dst[k], v)  # type: ignore[index]
        else:
            dst[k] = v
    return dst
