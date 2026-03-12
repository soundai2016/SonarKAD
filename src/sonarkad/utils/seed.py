"""Randomness / determinism utilities."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Set NumPy/Python/PyTorch RNG seeds.

    Parameters
    ----------
    seed:
        Seed value.
    deterministic:
        If True, enable (slower) deterministic PyTorch algorithms when possible.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Note: deterministic algorithms can hurt performance. Use for exact reproducibility.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Not available in older PyTorch builds.
            pass
