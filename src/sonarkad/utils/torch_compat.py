"""Torch serialization compatibility helpers.

PyTorch 2.6 changed the default of ``torch.load(weights_only=...)`` to
``True``. This improves safety when loading untrusted checkpoints, but it can
break legacy artifacts that contain numpy arrays or other non-tensor objects.

This repository saves some experiment artifacts (e.g., ``components.pt``) as
simple Python dictionaries containing numpy arrays for convenience. Those
files are produced locally by the user.

The helper below tries the default safe load first, and falls back to
``weights_only=False`` when needed.

Security note
-------------
Setting ``weights_only=False`` can execute arbitrary code during unpickling.
Only use it to load files you trust (e.g., artifacts produced by your own
runs).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def torch_load_compat(path: str | Path, *, map_location: str | torch.device = "cpu") -> Any:
    """Load a torch checkpoint with PyTorch-2.6 compatibility."""

    path = Path(path)

    try:
        return torch.load(path, map_location=map_location)
    except Exception as e:
        msg = str(e)
        if "Weights only load failed" in msg or "weights_only" in msg:
            try:
                return torch.load(path, map_location=map_location, weights_only=False)
            except TypeError:
                # Older PyTorch versions don't support `weights_only`.
                return torch.load(path, map_location=map_location)
        raise
