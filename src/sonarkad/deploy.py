"""Deployment utilities for SonarKAD.

This module provides minimal helpers to:
  - load a trained SonarKAD model bundle saved with ``torch.save`` (``.pt``), and
  - run fast batched inference on (r, f) query points.

The bundle format is produced by ``train_swellex96``.

NOTE: These utilities intentionally avoid any SWellex-specific assumptions
beyond the (range, frequency) normalization metadata stored in the bundle.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

# Optional progress bars (inference)
try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x

from .models import (
    BSplineLayerConfig,
    AbsorptionTermConfig,
    SonarKAD,
    SonarKADConfig,
    bspline_layer_config_from_dict,
)

from .utils.torch_compat import torch_load_compat


def sonarkad_config_from_dict(d: Dict[str, Any]) -> SonarKADConfig:
    """Reconstruct a :class:`~sonarkad.models.SonarKADConfig` from a plain dict.

    The SonarKAD model bundle stores ``model_cfg`` using ``dataclasses.asdict``.
    This helper is tolerant to older bundles that may contain extra keys.
    """

    spline_dict = d.get("spline", {}) if isinstance(d.get("spline", {}), dict) else {}
    spline_cfg = bspline_layer_config_from_dict(spline_dict) if spline_dict else BSplineLayerConfig()

    abs_dict = d.get("absorption", d.get("absorption_term", {}))
    if isinstance(abs_dict, dict):
        abs_spline = bspline_layer_config_from_dict(abs_dict.get("spline", {}))
        absorption_cfg = AbsorptionTermConfig(
            enabled=bool(abs_dict.get("enabled", False)),
            mode=str(abs_dict.get("mode", "thorp_scale")),
            reference_fc=bool(abs_dict.get("reference_fc", True)),
            init_log_scale=float(abs_dict.get("init_log_scale", 0.0)),
            spline=abs_spline,
            alpha_floor_db_per_km=float(abs_dict.get("alpha_floor_db_per_km", 0.0)),
        )
    elif isinstance(abs_dict, bool):
        absorption_cfg = AbsorptionTermConfig(enabled=bool(abs_dict))
    else:
        absorption_cfg = AbsorptionTermConfig(enabled=False)

    return SonarKADConfig(
        spline=spline_cfg,
        physics_init_grid_n=int(d.get("physics_init_grid_n", 256)),
        fc_hz=float(d.get("fc_hz", 3000.0)),
        use_absorption=bool(d.get("use_absorption", False)),
        f_min_hz=float(d.get("f_min_hz", 0.0)),
        f_max_hz=float(d.get("f_max_hz", 0.0)),
        absorption=absorption_cfg,
        gauge_fix_each_epoch=bool(d.get("gauge_fix_each_epoch", True)),
        gauge_fix_grid_n=int(d.get("gauge_fix_grid_n", 200)),
        gauge_fix_interaction=bool(d.get("gauge_fix_interaction", True)),
        gauge_fix_normalize_factors=bool(d.get("gauge_fix_normalize_factors", d.get("gauge_fix_normalize_factor", True))),
        gauge_fix_factor_mode=str(d.get("gauge_fix_factor_mode", "std")),
        gauge_fix_factor_eps=float(d.get("gauge_fix_factor_eps", 1e-6)),
        gauge_fix_fix_sign=bool(d.get("gauge_fix_fix_sign", True)),
        SL_db=float(d.get("SL_db", 0.0)),
        interaction_rank=int(d.get("interaction_rank", 0)),
    )


# Backwards-compat alias for older scripts.
sonarkad_config_from_dict = sonarkad_config_from_dict


def resolve_device(device: str = "auto") -> torch.device:
    dev = str(device).strip().lower()
    if dev == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    try:
        out = torch.device(dev)
    except Exception:
        return torch.device("cpu")
    if out.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if out.type == "mps":
        if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
            return torch.device("cpu")
    return out


def load_sonarkad_model_bundle(
    bundle_path: Union[str, Path],
    *,
    device: str = "auto",
) -> Tuple[SonarKAD, Dict[str, Any]]:
    """Load a SonarKAD model bundle (.pt) and return (model, metadata)."""

    bundle_path = Path(bundle_path)
    bundle = torch_load_compat(bundle_path, map_location="cpu")
    if not isinstance(bundle, dict):
        raise TypeError(f"Expected a dict bundle at {bundle_path}, got {type(bundle)}")

    cfg_dict = bundle.get("model_cfg", {})
    if not isinstance(cfg_dict, dict):
        raise TypeError("Bundle field 'model_cfg' must be a dict")

    cfg = sonarkad_config_from_dict(cfg_dict)

    norm = bundle.get("normalization", {})
    if not isinstance(norm, dict):
        raise TypeError("Bundle field 'normalization' must be a dict")
    r_min_m = float(norm.get("r_min_m", 0.0))
    r_max_m = float(norm.get("r_max_m", 1.0))
    model = SonarKAD(r_min_m=r_min_m, r_max_m=r_max_m, cfg=cfg)

    state = bundle.get("state_dict", None)
    if state is None:
        raise KeyError("Bundle missing 'state_dict'")
    model.load_state_dict(state)

    dev = resolve_device(device)
    model.to(dev)
    model.eval()

    meta = {
        "format": bundle.get("format"),
        "exp_name": bundle.get("exp_name"),
        "event": bundle.get("event"),
        "array": bundle.get("array"),
        "tone_set": bundle.get("tone_set"),
        "normalization": norm,
        "training_meta": bundle.get("training_meta", {}),
        "wgi_regularization": bundle.get("wgi_regularization", {}),
        "model_cfg": asdict(cfg),
        "device": str(dev),
    }
    return model, meta


@torch.no_grad()
def predict_rl(
    model: SonarKAD,
    *,
    r_m: Union[np.ndarray, float],
    f_hz: Union[np.ndarray, float],
    normalization: Dict[str, float],
    batch_size: int = 131072,
    progress_bar: bool = True,
) -> np.ndarray:
    """Predict RL(dB) for query points.

    Args:
        model: loaded SonarKAD/SonarKAD model.
        r_m: range(s) in meters.
        f_hz: frequency(s) in Hz.
        normalization: dict with keys {r_min_m, r_max_m, f_min_hz, f_max_hz}.
        batch_size: inference batch size.

    Returns:
        Predictions with broadcasted shape of (r_m, f_hz).
    """

    r = np.asarray(r_m, dtype=np.float32)
    f = np.asarray(f_hz, dtype=np.float32)
    rr, ff = np.broadcast_arrays(r, f)

    r_flat = rr.ravel()
    f_flat = ff.ravel()

    r_min = float(normalization["r_min_m"])
    r_max = float(normalization["r_max_m"])
    f_min = float(normalization["f_min_hz"])
    f_max = float(normalization["f_max_hz"])

    denom_r = (r_max - r_min) if (r_max - r_min) != 0 else 1.0
    denom_f = (f_max - f_min) if (f_max - f_min) != 0 else 1.0

    x0 = (r_flat - r_min) / denom_r
    x1 = (f_flat - f_min) / denom_f
    x = np.stack([x0, x1], axis=1)
    x = np.clip(x, 0.0, 1.0).astype(np.float32)

    dev = next(model.parameters()).device

    preds = []
    it = range(0, x.shape[0], int(batch_size))
    it = tqdm(it, desc="inference", unit="batch", disable=(not progress_bar))
    for i in it:
        xb = torch.from_numpy(x[i : i + int(batch_size)]).to(dev)
        yb = model(xb).detach().cpu().numpy().astype(np.float32)
        preds.append(yb)

    y = np.concatenate(preds, axis=0).reshape(rr.shape)
    return y


def predict_from_bundle(
    bundle_path: Union[str, Path],
    *,
    r_m: Union[np.ndarray, float],
    f_hz: Union[np.ndarray, float],
    device: str = "auto",
    batch_size: int = 131072,
    progress_bar: bool = True,
) -> np.ndarray:
    """Convenience wrapper: load bundle then predict."""

    model, meta = load_sonarkad_model_bundle(bundle_path, device=device)
    norm = meta.get("normalization", {})
    if not isinstance(norm, dict):
        raise TypeError("bundle normalization must be a dict")
    return predict_rl(
        model,
        r_m=r_m,
        f_hz=f_hz,
        normalization=norm,
        batch_size=batch_size,
        progress_bar=progress_bar,
    )
