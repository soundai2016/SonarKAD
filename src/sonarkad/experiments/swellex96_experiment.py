"""SWellEx-96 real-data experiment (prepare + train + plot).

This addresses the key editor/reviewer feedback:
- move beyond the crude, separable toy sonar equation;
- demonstrate on a realistic shallow-water dataset with known tonal sources;
- include a model extension that can represent non-separable TL(r,f) structure
  (low-rank interaction ψ(r,f)).

The pipeline is intentionally modular:
1) prepare:   raw .sio + range table -> processed .npz (tonal RL vs r,f,t)
2) train:     processed .npz -> SonarKAD(+ψ) fit and component plots
"""

from __future__ import annotations

import json
import os
import math
import copy
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Optional progress bars (training + inference)
try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x

from ..data.sio import SIOReader
from ..data.swellex96 import (
    NOISE_FREQS,
    TonalExtractionConfig,
    extract_tonal_rl_db,
    get_tone_frequencies,
    load_range_table,
    load_vla_depths_m,
    save_processed_npz,
)
from ..data.ctd import load_ctd_casts, estimate_depth_averaged_c0, aggregate_sound_speed_profile
from ..models import (
    BSplineLayerConfig,
    AbsorptionTermConfig,
    SonarKAD,
    SonarKADConfig,
    SmallMLP,
    count_parameters,
    bspline_layer_config_from_dict,
)
from ..baselines import (
    explained_variance,
    fit_gam_spline,
    fit_parametric_tl,
    fit_waveguide_invariant_striation,
    fit_pekeris_modal_striation,
    fit_profile_modal_striation,
    estimate_beta_structure_tensor,
    GAMConfig,
    ParametricTLConfig,
    WaveguideInvariantConfig,
    PekerisModalStriationConfig,
    ProfileModalStriationConfig,
)
from ..utils.paths import ensure_dir
from ..utils.seed import set_global_seed
from ..utils.torch_compat import torch_load_compat



# ---------------------------------------------------------------------
# Training utilities: device selection, early stopping, WGI regularization
# ---------------------------------------------------------------------

def _device_from_cfg(train_cfg: dict) -> torch.device:
    """Resolve a torch.device from config.

    Supported:
      - device: auto|cpu|cuda|cuda:0|mps (if available)
    """
    dev = str(train_cfg.get("device", "auto")).strip().lower()
    if dev == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    try:
        device = torch.device(dev)
    except Exception:
        print(f"[WARN] Invalid device='{dev}', falling back to CPU.")
        return torch.device("cpu")

    if device.type == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available; using CPU.")
        return torch.device("cpu")
    if device.type == "mps":
        if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
            print("[WARN] MPS requested but not available; using CPU.")
            return torch.device("cpu")
    return device


class EarlyStopper:
    """Simple early stopping on a scalar validation metric (lower is better)."""

    def __init__(self, patience: int = 50, min_delta: float = 0.0):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best: Optional[float] = None
        self.best_state: Optional[dict] = None
        self.best_epoch: int = -1
        self._bad: int = 0

    def step(self, value: float, model: nn.Module, epoch: int) -> bool:
        """Update with new metric. Returns True if training should stop."""
        if self.best is None or value < (self.best - self.min_delta):
            self.best = float(value)
            self.best_state = copy.deepcopy(model.state_dict())
            self.best_epoch = int(epoch)
            self._bad = 0
            return False
        self._bad += 1
        return self._bad >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def wgi_invariance_penalty(
    model: nn.Module,
    x_norm: torch.Tensor,
    *,
    beta: float,
    r_min_m: float,
    r_max_m: float,
    f_min_hz: float,
    f_max_hz: float,
    step_logr: float = 0.02,
) -> torch.Tensor:
    """Waveguide-invariant regularizer for the interaction term ψ(r,f).

    Enforces approximate invariance of ψ along constant-ξ contours, where:
        ξ = log(f) + β log(r).

    We use a *finite-difference consistency* penalty to avoid expensive second derivatives:
        ψ(r·e^{±h}, f·e^{∓βh}) should be close to ψ(r,f).

    Args:
        model: SonarKAD model exposing forward_components().
        x_norm: (B,2) normalized inputs in [0,1]^2 (r_norm, f_norm).
        beta: waveguide-invariant parameter β.
        r_min_m, r_max_m: physical range bounds for normalization.
        f_min_hz, f_max_hz: physical frequency bounds for normalization.
        step_logr: step size h in log-range.

    Returns:
        Scalar tensor penalty.
    """
    comps = model.forward_components(x_norm)
    psi = comps.get("psi_rf", None)
    if psi is None:
        return x_norm.new_tensor(0.0)

    psi = psi.view(-1)

    # Physical coordinates
    r = r_min_m + x_norm[:, 0] * (r_max_m - r_min_m)
    f = f_min_hz + x_norm[:, 1] * (f_max_hz - f_min_hz)

    h = x_norm.new_tensor(float(step_logr))
    beta_t = x_norm.new_tensor(float(beta))

    # Shift along ξ-contours: d log f = -β d log r
    r_plus = r * torch.exp(h)
    f_plus = f * torch.exp(-beta_t * h)
    r_minus = r * torch.exp(-h)
    f_minus = f * torch.exp(beta_t * h)

    # Clip to domain
    r_plus = torch.clamp(r_plus, min=r_min_m, max=r_max_m)
    r_minus = torch.clamp(r_minus, min=r_min_m, max=r_max_m)
    f_plus = torch.clamp(f_plus, min=f_min_hz, max=f_max_hz)
    f_minus = torch.clamp(f_minus, min=f_min_hz, max=f_max_hz)

    # Back to normalized
    denom_r = (r_max_m - r_min_m) if (r_max_m - r_min_m) != 0 else 1.0
    denom_f = (f_max_hz - f_min_hz) if (f_max_hz - f_min_hz) != 0 else 1.0
    x_plus = torch.stack([(r_plus - r_min_m) / denom_r, (f_plus - f_min_hz) / denom_f], dim=1)
    x_minus = torch.stack([(r_minus - r_min_m) / denom_r, (f_minus - f_min_hz) / denom_f], dim=1)

    psi_plus = model.forward_components(x_plus)["psi_rf"].view(-1)
    psi_minus = model.forward_components(x_minus)["psi_rf"].view(-1)

    # Consistency with the center point
    return 0.5 * (torch.mean((psi - psi_plus) ** 2) + torch.mean((psi - psi_minus) ** 2))


# ---------------------------------------------------------------------
# Configuration helpers (backward compatible)
# ---------------------------------------------------------------------

def _infer_swellex96_sio_path(data_root: Path, event: str, array: str) -> Optional[Path]:
    """Infer a likely SWellEx-96 .sio path from (data_root, event, array).

    The public SWellEx-96 distribution commonly uses:
      - Event S5  (VLA): J1312315.vla.21els.sio
      - Event S59 (VLA): J1341145.vla.21els.sio

    If the (event,array) pair is unknown, we fall back to searching for a unique
    .sio file under ``data_root`` (optionally filtered by the array tag).
    """
    event_u = (event or "").strip().upper()
    array_u = (array or "").strip().upper()

    known = {
        ("S5", "VLA"): "J1312315.vla.21els.sio",
        ("S59", "VLA"): "J1341145.vla.21els.sio",
    }
    fname = known.get((event_u, array_u))
    if fname:
        p = (data_root / fname).resolve()
        if p.exists():
            return p

    # Fallback: search for a unique .sio file in data_root.
    cands = sorted(data_root.glob("*.sio"))
    if array_u:
        tag = f".{array_u.lower()}"
        cands_tag = [c for c in cands if tag in c.name.lower()]
        if len(cands_tag) == 1:
            return cands_tag[0].resolve()
        if len(cands_tag) >= 1:
            cands = cands_tag

    if len(cands) == 1:
        return cands[0].resolve()

    return None


def _resolve_swellex96_dataset_config(ds_in: dict) -> dict:
    """Normalize/resolve dataset config keys for SWellEx-96.

    This makes the pipeline robust to two config styles:

    Style A (explicit paths; preferred for reproducibility)
      dataset:
        sio_path: data/J1312315.vla.21els.sio
        range_table_path: data/RangeEventS5

    Style B (data_root + event + array; convenience)
      dataset:
        data_root: data
        event: S5
        array: VLA

    The function fills in missing explicit paths when possible and returns
    a shallow-copied dict.
    """
    ds = dict(ds_in or {})
    data_root = Path(ds.get("data_root", "data"))
    event = str(ds.get("event", ""))
    array = str(ds.get("array", ""))

    # -------------------
    # Resolve sio_path
    # -------------------
    sio_raw = ds.get("sio_path", None)
    if sio_raw is None or (isinstance(sio_raw, str) and not sio_raw.strip()):
        inferred = _infer_swellex96_sio_path(data_root, event, array)
        if inferred is None:
            raise KeyError(
                "Missing dataset.sio_path and could not infer it. "
                "Set dataset.sio_path explicitly, or provide dataset.data_root + dataset.event + dataset.array."
            )
        ds["sio_path"] = str(inferred)
    else:
        ds["sio_path"] = str(Path(sio_raw))

    # ------------------------
    # Resolve range_table_path
    # ------------------------
    rng_raw = ds.get("range_table_path", None)
    if rng_raw is None or (isinstance(rng_raw, str) and not rng_raw.strip()):
        ev_u = str(event).strip().upper()
        if ev_u:
            # Common extracted layout: data/RangeEventS5/, data/RangeEventS59/
            cand_dir = data_root / f"RangeEvent{ev_u}"
            if cand_dir.exists():
                ds["range_table_path"] = str(cand_dir)
            else:
                # Fallback: a file directly under data_root.
                cand_file = data_root / f"SproulTo{str(array).strip().upper()}.{ev_u}.txt"
                if cand_file.exists():
                    ds["range_table_path"] = str(cand_file)

        if "range_table_path" not in ds:
            raise KeyError(
                "Missing dataset.range_table_path and could not infer it. "
                "Set dataset.range_table_path explicitly (file or directory), "
                "or provide dataset.data_root + dataset.event."
            )
    else:
        ds["range_table_path"] = str(Path(rng_raw))

    # ------------------------
    # Optional convenience paths
    # ------------------------
    if (ds.get("positions_vla_path", None) is None) and data_root.exists():
        cand_pos = data_root / "positions_vla.txt"
        if cand_pos.exists():
            ds["positions_vla_path"] = str(cand_pos)

    return ds


def _resolve_preprocess_config(exp_cfg: dict, ds: dict) -> dict:
    """Resolve tonal-extraction config with backward-compatibility.

    Preferred:
      exp_cfg['preprocess'] : TonalExtractionConfig fields

    Backward compatibility:
      dataset.stft_nperseg, dataset.stft_noverlap, dataset.stft_window
      dataset.noise_freq_hz

    Returns a dict suitable for TonalExtractionConfig(**dict).
    """
    pre = exp_cfg.get("preprocess", None)
    if not isinstance(pre, dict):
        pre = {}

    # Backward-compat: allow preprocess embedded under dataset
    if not pre:
        pre2 = ds.get("preprocess", None)
        if isinstance(pre2, dict):
            pre = dict(pre2)

    # Backward-compat: map STFT-style keys into (win_sec, hop_sec, n_fft)
    if "stft_nperseg" in ds:
        try:
            fs = float(pre.get("fs_hz", ds.get("fs_hz", TonalExtractionConfig.fs_hz)))
            nper = int(ds.get("stft_nperseg"))
            nover = int(ds.get("stft_noverlap", nper // 2))
            hop = max(1, nper - nover)
            pre.setdefault("fs_hz", fs)
            pre.setdefault("win_sec", float(nper) / fs)
            pre.setdefault("hop_sec", float(hop) / fs)
            pre.setdefault("n_fft", int(ds.get("stft_nfft", nper)))
            pre.setdefault("window", str(ds.get("stft_window", pre.get("window", "hann"))))
        except Exception:
            # If anything goes wrong, fall back to TonalExtractionConfig defaults.
            pass

    # Backward-compat: noise bins
    if "noise_freq_hz" in ds and ("noise_freqs_hz" not in pre):
        try:
            pre.setdefault("subtract_noise", True)
            pre.setdefault("noise_freqs_hz", list(ds.get("noise_freq_hz")))
        except Exception:
            pass

    return pre


def prepare_swellex96(exp_cfg: dict, out_dir: str | Path, *, force: bool = False) -> Path:
    """Prepare processed tonal dataset (.npz) from raw SWellEx-96 files.

    Required config keys
    --------------------
    exp_cfg['dataset']['sio_path']         : path to uncompressed .sio
    exp_cfg['dataset']['range_table_path'] : extracted/converted (time,range) file
    exp_cfg['dataset']['channels']         : list of channel indices (1-based default)
    exp_cfg['dataset']['tone_set']         : e.g. 'T49_13_high' or 'T49_13'
    exp_cfg['preprocess']                  : TonalExtractionConfig fields
    """
    out_dir = ensure_dir(out_dir)

    ds = _resolve_swellex96_dataset_config(exp_cfg.get("dataset", {}))
    exp_cfg["dataset"] = ds
    sio_path = Path(ds["sio_path"])
    range_path = Path(ds["range_table_path"])
    channels = ds.get("channels", list(range(1, 22)))
    channels_one_indexed = bool(ds.get("channels_are_one_indexed", True))
    freq_mode = str(ds.get("frequency_mode", "tones")).strip().lower()
    if freq_mode in ("band", "grid", "spectrum"):
        f_min = float(ds.get("band_f_min_hz", ds.get("f_min_hz", 50.0)))
        f_max = float(ds.get("band_f_max_hz", ds.get("f_max_hz", 400.0)))
        df = float(ds.get("band_df_hz", ds.get("df_hz", 1.0)))
        if df <= 0:
            raise ValueError("band_df_hz must be > 0")
        tones_hz = np.arange(f_min, f_max + 0.5 * df, df, dtype=np.float64).tolist()
        # Use a descriptive tag for filenames/metadata
        tone_set = ds.get("tone_set", f"band_{int(round(f_min))}_{int(round(f_max))}_df{df:g}")
    else:
        tone_set = ds.get("tone_set", "T49_13_high")
        tones_hz = get_tone_frequencies(tone_set)

    # Optional trimming
    t_start = float(ds.get("t_start_sec", 0.0))
    t_end = ds.get("t_end_sec", None)
    t_end = None if t_end is None else float(t_end)


    # Optional array geometry (for reproducibility / depth-aware extensions)
    positions_path = ds.get("positions_vla_path", None)
    reverse_positions = bool(ds.get("reverse_positions_for_sio", True))
    channel_depths_m: Optional[np.ndarray] = None
    if positions_path is not None:
        try:
            depths_all = load_vla_depths_m(Path(positions_path), reverse_for_sio=reverse_positions)
            # Map selected channels to depths
            ch_idx = []
            for c in channels:
                idx0 = int(c) - 1 if channels_one_indexed else int(c)
                ch_idx.append(idx0)
            if max(ch_idx) < depths_all.size:
                channel_depths_m = depths_all[np.asarray(ch_idx, dtype=int)]
        except Exception as e:
            # Geometry is optional for the current tonal RL experiments; proceed with a warning.
            print(f"[WARN] Could not load positions_vla file '{positions_path}': {e}")
            channel_depths_m = None

    # Determine output NPZ path (always stored under outputs/ unless overridden).
    ds0 = exp_cfg.get("dataset", {}) or {}
    npz_override = ds0.get("processed_npz", None)
    if isinstance(npz_override, str) and npz_override.strip():
        out_path = Path(npz_override)
    else:
        out_path = out_dir / f"swellex96_processed_{tone_set.lower()}.npz"
        ds0["processed_npz"] = str(out_path)
        exp_cfg["dataset"] = ds0

    # Idempotency: if a valid processed dataset already exists, reuse it.
    if out_path.exists() and (not force):
        stale, reason = _processed_npz_is_stale(exp_cfg, out_path)
        if not stale:
            print(f"[INFO] Processed dataset exists: {out_path} (skip).")
            return out_path
        print(f"[INFO] Processed dataset exists but is stale ({reason}); regenerating.")

    reader = SIOReader(sio_path)
    # Optional safety check: ensure we are reading the expected array file (e.g., VLA 21-channel subset).
    expected_nc = ds.get("expected_nc", None)
    if expected_nc is not None:
        exp_nc = int(expected_nc)
        if int(reader.header.nc) != exp_nc:
            raise ValueError(
                f"SIO channel count mismatch: header.nc={reader.header.nc} but expected_nc={exp_nc}. "
                f"Check that you are pointing to the correct uncompressed .sio file: {sio_path}"
            )

    prep_cfg = TonalExtractionConfig(**_resolve_preprocess_config(exp_cfg, ds))
    if prep_cfg.subtract_noise and (prep_cfg.noise_freqs_hz is None):
        prep_cfg.noise_freqs_hz = list(NOISE_FREQS)

    t_frames, f_hz, rl_db = extract_tonal_rl_db(
        reader,
        prep_cfg,
        channels=channels,
        tones_hz=tones_hz,
        t_start_sec=t_start,
        t_end_sec=t_end,
        channels_are_one_indexed=channels_one_indexed,
    )

    # Range alignment (interpolate).
    #
    # The range tables shipped with the dataset are typically sampled once per minute.
    # Their time column may be either:
    #   - relative seconds from the event start, or
    #   - absolute time-of-day tokens (hh:mm[:ss]).
    # To make this robust, we align by subtracting the first range-table timestamp.
    range_col = int(ds.get("range_column", 1))
    time_col = int(ds.get("time_column", 0))
    range_scale = float(ds.get("range_scale_to_m", 1.0))
    time_scale = float(ds.get("time_scale_to_sec", 1.0))
    time_offset = float(ds.get("time_offset_sec", 0.0))
    file_hint = ds.get("range_file_hint", None)

    t_range, r_range = load_range_table(
        range_path,
        range_col=range_col,
        time_col=time_col,
        range_scale_to_m=range_scale,
        time_scale_to_sec=time_scale,
        time_offset_sec=time_offset,
        file_hint=file_hint,
    )
    t_range_rel = (t_range - float(t_range[0])).astype(np.float64)

    # ------------------------------------------------------------
    # Optional: automatically align the range-table time axis to the
    # recording start time encoded in the SIO filename. This is
    # particularly useful for Event S59 where the SIO recording may
    # start about 60 s before the first range-table row.
    #
    # Convention: we add ``range_time_offset_sec`` to the STFT frame
    # times before interpolating in the (relative) range table.
    # A negative value shifts queries earlier in the range table.
    # ------------------------------------------------------------
    rto_raw = ds.get("range_time_offset_sec", 0.0)
    range_time_offset_mode = "explicit"
    if rto_raw is None or (isinstance(rto_raw, str) and rto_raw.strip().lower() in ("auto", "infer")):
        from ..data.swellex96 import infer_range_time_offset_sec

        range_time_offset = float(
            infer_range_time_offset_sec(sio_path, range_path, file_hint=file_hint)
        )
        range_time_offset_mode = "auto"
    else:
        range_time_offset = float(rto_raw)

    t_query = (t_frames + range_time_offset).astype(np.float64)
    r_frames = np.interp(t_query, t_range_rel, r_range).astype(np.float64)

    # Track how many tonal samples were marked invalid (set to NaN) by the
    # quality-control gate. Stored in metadata for reproducibility.
    n_invalid_total: Optional[int] = None

    # Optionally drop frames outside the range-table support instead of
    # extrapolating with endpoint values. This avoids training on frames
    # where r(t) is undefined (common for S59 if the recording begins
    # before the first range-table entry).
    drop_outside = bool(ds.get("drop_frames_outside_range_table", True))
    if drop_outside:
        t_min = float(np.min(t_range_rel))
        t_max = float(np.max(t_range_rel))
        keep = (t_query >= t_min) & (t_query <= t_max)
        if not np.all(keep):
            t_frames = t_frames[keep]
            rl_db = rl_db[keep, :]
            r_frames = r_frames[keep]
            t_query = t_query[keep]

    # Optional frame-level gating to remove obvious tone dropouts
    # (e.g., transmitter off / severe SNR collapse), which otherwise dominate
    # RMSE and can obscure striation structure.
    frame_filter = ds.get("frame_filter", {}) or {}
    db_floor = frame_filter.get("db_floor")
    min_valid_frac = frame_filter.get("min_valid_frac")
    if db_floor is not None:
        # For dense "band" frequency grids, most bins have no tone energy and
        # will fall below the threshold; in that case this simple fraction-based
        # gating is usually inappropriate. Users can override via force=true.
        if freq_mode != "tones" and not bool(frame_filter.get("force", False)):
            print(
                f"[prepare] frame_filter disabled for freq_mode='{freq_mode}'. "
                "Set frame_filter.force=true to apply anyway."
            )
        else:
            db_floor = float(db_floor)
            if min_valid_frac is None:
                min_valid_frac = 0.5
            min_valid_frac = float(min_valid_frac)
            # Sample-level validity mask (used both for frame gating and to
            # mark missing tonal detections as NaN for downstream training).
            valid = np.isfinite(rl_db) & (rl_db > db_floor)
            n_invalid = int(np.sum(~valid))
            n_invalid_total = n_invalid
            if n_invalid > 0:
                # Replace invalid samples by NaN to avoid contaminating RMSE
                # with artificial floor values (e.g., -300 dB).
                rl_db = rl_db.copy()
                rl_db[~valid] = np.nan
            frac_valid = np.mean(valid.astype(np.float64), axis=1)
            keep2 = frac_valid >= min_valid_frac
            n_drop = int(np.sum(~keep2))
            if n_drop > 0:
                t_frames = t_frames[keep2]
                rl_db = rl_db[keep2, :]
                r_frames = r_frames[keep2]
                t_query = t_query[keep2]
                print(
                    f"[prepare] frame_filter: kept {int(np.sum(keep2))}/{int(len(keep2))} frames "
                    f"(dropped {n_drop}) with db_floor={db_floor:.1f} dB, min_valid_frac={min_valid_frac:.2f}."
                )

    meta = {
        "sio_path": str(sio_path),
        "sio_size_bytes": int(os.path.getsize(sio_path)) if sio_path.exists() else None,
        "range_table_path": str(range_path),
        "range_file_hint": file_hint,
        "time_column": time_col,
        "range_column": range_col,
        "range_scale_to_m": range_scale,
        "time_scale_to_sec": time_scale,
        "time_offset_sec": time_offset,
        "range_time_offset_sec": range_time_offset,
        "range_time_offset_mode": range_time_offset_mode,
        "drop_frames_outside_range_table": drop_outside,
        "channels": list(channels),
        "channels_are_one_indexed": channels_one_indexed,
        "tone_set": tone_set,
        "tones_hz": [float(x) for x in tones_hz],
        # Quality-control (QC) settings applied to the processed RL tensor.
        "qc_db_floor": float(db_floor) if db_floor is not None else None,
        "qc_min_valid_frac": float(min_valid_frac) if min_valid_frac is not None else None,
        "qc_invalid_samples": int(n_invalid_total) if n_invalid_total is not None else None,
        "prep_cfg": asdict(prep_cfg),
        "preprocess": asdict(prep_cfg),
        "t_start_sec": t_start,
        "t_end_sec": t_end,
        "reader_header": asdict(reader.header),
    }

    # Optional geometry info (useful for depth-aware extensions / reproducibility).
    if positions_path is not None:
        meta["positions_vla_path"] = str(positions_path)
        meta["reverse_positions_for_sio"] = bool(reverse_positions)
    if channel_depths_m is not None:
        meta["channel_depths_m"] = channel_depths_m.astype(np.float64)

    save_processed_npz(out_path, t_sec=t_frames, r_m=r_frames, f_hz=f_hz, rl_db=rl_db, meta=meta)
    return out_path


def _flatten_processed(npz_path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load processed NPZ and flatten into sample pairs.

    Returns
    -------
    t_sec_frames: (T,)
    r_m_frames:   (T,)
    f_hz:         (F,)
    y_db:         (T,F)
    """
    d = np.load(npz_path, allow_pickle=True)
    t = d["t_sec"].astype(np.float64)
    r = d["r_m"].astype(np.float64)
    f = d["f_hz"].astype(np.float64)
    y = d["rl_db"].astype(np.float32)
    return t, r, f, y




def _load_npz_meta(npz_path: Path) -> Dict[str, object]:
    """Load meta dict from a processed NPZ, if present."""
    try:
        d = np.load(npz_path, allow_pickle=True)
        if "meta" not in d.files:
            return {}
        meta_obj = d["meta"]
        # Stored as object array([dict], dtype=object)
        if hasattr(meta_obj, "item"):
            try:
                meta = meta_obj.item()
            except Exception:
                meta = meta_obj
        else:
            meta = meta_obj
        return dict(meta) if isinstance(meta, dict) else {}
    except Exception:
        return {}


def _processed_npz_is_stale(exp_cfg: dict, npz_path: Path) -> Tuple[bool, str]:
    """Return (is_stale, reason) for a processed dataset file."""
    if not npz_path.exists():
        return True, "missing"
    meta = _load_npz_meta(npz_path)
    if not meta:
        # Older/foreign files without metadata are considered stale to avoid silent misuse.
        return True, "no_meta"

    ds = exp_cfg.get("dataset", {}) or {}
    pre = exp_cfg.get("preprocess", {}) or {}

    def _canon(p: str) -> str:
        try:
            return os.path.normcase(os.path.abspath(p))
        except Exception:
            return str(p)

    sio_path = str(ds.get("sio_path", ""))
    meta_sio_path = str(meta.get("sio_path", "")) if meta.get("sio_path") is not None else ""

    # 1) SIO file size (strong check). If size matches, we treat the file as the
    # same dataset even if the project directory moved (path string mismatch).
    size_matches = False
    try:
        if sio_path and os.path.exists(sio_path) and meta.get("sio_size_bytes") is not None:
            size = int(os.path.getsize(sio_path))
            size_matches = int(meta["sio_size_bytes"]) == size
            if not size_matches:
                return True, "sio_size_mismatch"
    except Exception:
        pass

    # 2) SIO path (weak check). Only enforce if we *cannot* confirm by file size.
    if (not size_matches) and meta_sio_path and sio_path:
        if _canon(meta_sio_path) != _canon(sio_path):
            return True, "sio_path_mismatch"

    # 3) Tone set
    tone_set = ds.get("tone_set", None)
    if tone_set is not None and meta.get("tone_set") is not None and str(meta["tone_set"]) != str(tone_set):
        return True, "tone_set_mismatch"

    # 4) Preprocess fingerprint (compare only keys present in config)
    meta_pre = meta.get("preprocess", meta.get("prep_cfg", {})) or {}
    for k in ["fs_hz", "win_sec", "hop_sec", "n_fft", "window", "detrend", "channel_pool", "subtract_noise", "noise_stat"]:
        if k in pre:
            if meta_pre.get(k) != pre.get(k):
                return True, f"preprocess_{k}_mismatch"

    return False, "ok"


def _ensure_processed_npz(exp_cfg: dict) -> Path:
    """Ensure the processed NPZ exists and matches the current experiment config.

    If missing or stale, automatically re-runs `prepare_swellex96`.
    """
    ds = exp_cfg.get("dataset", {}) or {}
    out_cfg = exp_cfg.get("outputs", {}) or {}

    processed_dir = Path(out_cfg.get("processed_dir", "outputs/processed"))
    ensure_dir(processed_dir)

    npz_str = ds.get("processed_npz", "")
    if npz_str:
        npz_path = Path(npz_str)
    else:
        tone_set = str(ds.get("tone_set", "tones")).lower()
        npz_path = processed_dir / f"swellex96_processed_{tone_set}.npz"
        ds["processed_npz"] = str(npz_path)
        exp_cfg["dataset"] = ds

    stale, reason = _processed_npz_is_stale(exp_cfg, npz_path)
    if stale:
        print(f"[INFO] processed dataset '{npz_path}' is stale ({reason}); running swellex96-prepare...")
        prepare_swellex96(exp_cfg, processed_dir)
        if not npz_path.exists():
            raise FileNotFoundError(f"prepare_swellex96 did not create expected NPZ: {npz_path}")

    return npz_path

def train_swellex96(
    exp_cfg: dict,
    run_dir: str | Path,
    *,
    force: bool = False,
    frame_train_mask: Optional[np.ndarray] = None,
    frame_test_mask: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Train the model on a processed SWellEx-96 tonal dataset."""

    run_dir = ensure_dir(run_dir)

    # Idempotency: if key artifacts already exist, reuse them.
    results_path = Path(run_dir) / "results.json"
    comp_path = Path(run_dir) / "components.pt"
    bundle_path = Path(run_dir) / "sonarkad_model.pt"
    if (not force) and results_path.exists() and comp_path.exists() and bundle_path.exists():
        try:
            return json.loads(results_path.read_text(encoding="utf-8"))
        except Exception:
            # Fall through to retrain.
            pass

    # ------------------------------------------------------------------
    # Reproducibility: prefer an explicit experiment seed, otherwise fall
    # back to training.seed.
    # ------------------------------------------------------------------
    train_cfg = exp_cfg.get("training", {}) or {}
    seed = exp_cfg.get("seed", None)
    if seed is None:
        seed = train_cfg.get("seed", 42)
    seed = int(seed)
    set_global_seed(seed)

    # Experiment / dataset metadata (used in bundles and figure titles)
    ds = exp_cfg.get("dataset", {}) or {}
    exp_name = str(exp_cfg.get("name", ds.get("name", "swellex96")))
    event = str(ds.get("event", ""))
    array = str(ds.get("array", ""))
    tone_set = str(ds.get("tone_set", ds.get("tones", "")))

    # ------------------------------------------------------------------
    # Load processed dataset (auto-runs prepare if missing/stale)
    # ------------------------------------------------------------------
    npz_path = _ensure_processed_npz(exp_cfg)
    t_frames, r_frames, f_hz, y_tf = _flatten_processed(npz_path)
    T, F = y_tf.shape

    # ------------------------------------------------------------------
    # Optional environment: CTD-derived effective sound speed for
    # traditional baselines.
    # ------------------------------------------------------------------
    env_cfg = exp_cfg.get("environment", {}) or {}
    water_depth_m = float(env_cfg.get("water_depth_m", 217.0) or 217.0)

    c0_m_per_s = env_cfg.get("c0_m_per_s", None)
    c0_m_per_s = float(c0_m_per_s) if c0_m_per_s is not None else None

    ctd_dir = env_cfg.get("ctd_dir", None)
    ctd_stat = str(env_cfg.get("ctd_statistic", "median"))
    ctd_stations = env_cfg.get("ctd_stations", env_cfg.get("ctd_station", None))

    z_profile_m: Optional[np.ndarray] = None
    c_profile_mps: Optional[np.ndarray] = None

    if ctd_dir is not None:
        try:
            casts = load_ctd_casts(Path(ctd_dir), stations=ctd_stations)

            if c0_m_per_s is None:
                c0_m_per_s = estimate_depth_averaged_c0(
                    casts, water_depth_m=water_depth_m, statistic=ctd_stat
                )
                print(
                    f"[INFO] CTD-derived c0={c0_m_per_s:.2f} m/s (stat={ctd_stat}, depth={water_depth_m:.1f} m)"
                )

            dz_prof = float(env_cfg.get("profile_dz_m", 1.0))
            z_profile_m, c_profile_mps = aggregate_sound_speed_profile(
                casts, water_depth_m=water_depth_m, dz_m=dz_prof, statistic=ctd_stat
            )
            print(
                f"[INFO] CTD profile aggregated: casts={len(casts)} dz={dz_prof:g} m depth={water_depth_m:.1f} m"
            )
        except Exception as e:
            print(
                f"[WARN] Could not load CTD casts from '{ctd_dir}': {e}. "
                f"Using default c0=1500 m/s and skipping profile-based baselines."
            )
            z_profile_m, c_profile_mps = None, None
            if c0_m_per_s is None:
                c0_m_per_s = 1500.0

    if c0_m_per_s is None:
        c0_m_per_s = 1500.0

    # ------------------------------------------------------------------
    # Flatten dataset into sample-wise regression pairs (r,f)->RL.
    # We *drop* invalid tonal detections (NaNs inserted during QC).
    # ------------------------------------------------------------------
    r_rep = np.repeat(r_frames, F)
    f_tile = np.tile(f_hz, T)
    y_flat = y_tf.reshape(-1)

    valid = np.isfinite(y_flat)
    if not bool(np.all(valid)):
        n_drop = int(np.sum(~valid))
        print(f"[INFO] Dropping {n_drop} invalid (NaN) samples from flattened dataset.")

    if int(np.sum(valid)) == 0:
        raise ValueError(
            "No valid tonal detections after QC. "
            "Adjust dataset.frame_filter.db_floor / min_valid_frac or choose a tone_set that is transmitted."
        )

    r_rep = r_rep[valid]
    f_tile = f_tile[valid]
    y_flat = y_flat[valid]

    frame_idx_full = np.repeat(np.arange(T, dtype=np.int64), F)
    frame_idx = frame_idx_full[valid]

    # ------------------------------------------------------------------
    # Normalization to [0,1]^2
    # ------------------------------------------------------------------
    norm_cfg = exp_cfg.get("norm", {}) or {}

    def _as_float_or_default(v: object, default: float) -> float:
        if v is None:
            return float(default)
        if isinstance(v, str) and v.strip().lower() in {"auto", "none", "null", ""}:
            return float(default)
        return float(v)

    r_min = _as_float_or_default(norm_cfg.get("r_min_m", None), float(np.min(r_rep)))
    r_max = _as_float_or_default(norm_cfg.get("r_max_m", None), float(np.max(r_rep)))
    f_min = _as_float_or_default(norm_cfg.get("f_min_hz", None), float(np.min(f_tile)))
    f_max = _as_float_or_default(norm_cfg.get("f_max_hz", None), float(np.max(f_tile)))

    r_norm = (r_rep - r_min) / (r_max - r_min + 1e-12)
    f_norm = (f_tile - f_min) / (f_max - f_min + 1e-12)

    X = np.stack([r_norm, f_norm], axis=1).astype(np.float32)
    y = y_flat.astype(np.float32).reshape(-1, 1)

    # ------------------------------------------------------------------
    # Train/test split
    # - default: contiguous holdout by frame index (time)
    # - optional: explicit masks (used by blocked CV)
    # ------------------------------------------------------------------
    split_cfg = exp_cfg.get("split", {}) or {}

    if frame_train_mask is not None:
        ft = np.asarray(frame_train_mask, dtype=bool).reshape(T)
        if frame_test_mask is None:
            fe = ~ft
        else:
            fe = np.asarray(frame_test_mask, dtype=bool).reshape(T)

        if int(np.sum(ft)) < 2 or int(np.sum(fe)) < 2:
            raise ValueError(
                f"Degenerate split: train_frames={int(np.sum(ft))}, test_frames={int(np.sum(fe))} (need >=2 each)."
            )

        train_mask = ft[frame_idx]
        test_mask = fe[frame_idx]
        n_train_frames = int(np.sum(ft))
        n_test_frames = int(np.sum(fe))
    else:
        frac_train = float(split_cfg.get("train_frac", 0.7))
        n_train_frames = int(np.floor(frac_train * T))
        train_mask = frame_idx < n_train_frames
        test_mask = ~train_mask
        n_test_frames = int(T - n_train_frames)

    # Optional frame-level validation split (for early stopping)
    es_cfg = train_cfg.get("early_stopping", {}) if isinstance(train_cfg.get("early_stopping", {}), dict) else {}
    val_frac_frames = float(es_cfg.get("val_frac_frames", es_cfg.get("val_frac", 0.0)) or 0.0)

    val_mask = np.zeros_like(train_mask, dtype=bool)
    if bool(es_cfg.get("enabled", False)) and val_frac_frames > 0.0 and n_train_frames >= 4:
        n_val_frames = max(1, int(np.floor(val_frac_frames * n_train_frames)))
        if frame_train_mask is not None:
            train_frame_ids = np.where(ft)[0]
            val_frame_ids = train_frame_ids[-n_val_frames:]
        else:
            val_frame_ids = np.arange(max(0, n_train_frames - n_val_frames), n_train_frames)
        val_mask = np.isin(frame_idx, val_frame_ids)

    train_fit_mask = train_mask & (~val_mask)

    # ------------------------------------------------------------------
    # Torch loaders
    # ------------------------------------------------------------------
    X_train = torch.tensor(X[train_fit_mask], dtype=torch.float32)
    y_train = torch.tensor(y[train_fit_mask], dtype=torch.float32)

    X_val = torch.tensor(X[val_mask], dtype=torch.float32) if np.any(val_mask) else None
    y_val = torch.tensor(y[val_mask], dtype=torch.float32) if np.any(val_mask) else None

    X_test = torch.tensor(X[test_mask], dtype=torch.float32)
    y_test = torch.tensor(y[test_mask], dtype=torch.float32)

    device = _device_from_cfg(train_cfg)
    use_amp = bool(train_cfg.get("amp", False)) and (device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(train_cfg.get("allow_tf32", True))

    # Progress bars + optional matmul precision knob (PyTorch 2.x)
    progress = bool(train_cfg.get("progress_bar", True))
    mp = train_cfg.get("matmul_precision", None)
    if mp is not None and hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision(str(mp))
        except Exception:
            pass

    batch_size = int(train_cfg.get("batch_size", 4096))
    num_workers = int(train_cfg.get("num_workers", 0))
    pin_memory = bool(train_cfg.get("pin_memory", device.type == "cuda"))

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = None
    if X_val is not None and y_val is not None and X_val.shape[0] > 0:
        val_loader = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # ------------------------------------------------------------------
    # Model configs
    # ------------------------------------------------------------------
    model_cfg = exp_cfg.get("model", {}) or {}
    spline_cfg = bspline_layer_config_from_dict(model_cfg.get("spline", {}))

    # Optional explicit absorption term configuration.
    abs_cfg_raw = model_cfg.get("absorption", model_cfg.get("absorption_term", {}))
    if isinstance(abs_cfg_raw, bool):
        abs_cfg = {"enabled": bool(abs_cfg_raw)}
    elif isinstance(abs_cfg_raw, dict):
        abs_cfg = dict(abs_cfg_raw)
    else:
        abs_cfg = {}

    abs_spline_cfg = bspline_layer_config_from_dict(abs_cfg.get("spline", {}))
    absorption_term_cfg = AbsorptionTermConfig(
        enabled=bool(abs_cfg.get("enabled", False)),
        mode=str(abs_cfg.get("mode", "thorp_scale")),
        reference_fc=bool(abs_cfg.get("reference_fc", True)),
        init_log_scale=float(abs_cfg.get("init_log_scale", 0.0)),
        spline=abs_spline_cfg,
        alpha_floor_db_per_km=float(abs_cfg.get("alpha_floor_db_per_km", 0.0)),
    )

    sonarkad_cfg = SonarKADConfig(
        spline=spline_cfg,
        physics_init_grid_n=int(model_cfg.get("physics_init_grid_n", 256)),
        SL_db=float(model_cfg.get("SL_db", 0.0)),
        fc_hz=float(model_cfg.get("fc_hz", float(np.mean(f_hz)))),
        use_absorption=bool(model_cfg.get("use_absorption", False)),
        f_min_hz=float(np.min(f_hz)),
        f_max_hz=float(np.max(f_hz)),
        absorption=absorption_term_cfg,
        gauge_fix_each_epoch=bool(model_cfg.get("gauge_fix_each_epoch", True)),
        gauge_fix_grid_n=int(model_cfg.get("gauge_fix_grid_n", 200)),
        gauge_fix_interaction=bool(model_cfg.get("gauge_fix_interaction", True)),
        gauge_fix_normalize_factors=bool(model_cfg.get("gauge_fix_normalize_factors", True)),
        gauge_fix_factor_mode=str(model_cfg.get("gauge_fix_factor_mode", "std")),
        gauge_fix_factor_eps=float(model_cfg.get("gauge_fix_factor_eps", 1e-6)),
        gauge_fix_fix_sign=bool(model_cfg.get("gauge_fix_fix_sign", True)),
        interaction_rank=int(model_cfg.get("interaction_rank", 0)),
    )


    # ------------------------------------------------------------------
    # Optional: cross-event transfer initialization
    # ------------------------------------------------------------------
    transfer_cfg = train_cfg.get("transfer", None)
    if isinstance(transfer_cfg, dict) and transfer_cfg.get("bundle_path"):
        if bool(transfer_cfg.get("disable_gauge_fix_interaction", False)):
            sonarkad_cfg.gauge_fix_interaction = False

    sonarkad = SonarKAD(r_min_m=r_min, r_max_m=r_max, cfg=sonarkad_cfg)
    # Broad TL prior (spreading [+ optional absorption])
    sonarkad.physics_init()

    # Apply transfer init (copy selected parts from a pretrained bundle) and freeze.
    if isinstance(transfer_cfg, dict) and transfer_cfg.get("bundle_path"):
        try:
            bundle_path = Path(str(transfer_cfg.get("bundle_path")))
            load_parts = transfer_cfg.get("load_parts", ["phi_f"])
            freeze_parts = transfer_cfg.get("freeze_parts", load_parts)

            # Load bundle (cpu)
            bundle_obj = torch_load_compat(bundle_path, map_location="cpu")
            state_dict = bundle_obj.get("state_dict", bundle_obj) if isinstance(bundle_obj, dict) else bundle_obj

            if not isinstance(state_dict, dict):
                raise TypeError(f"Unexpected bundle format in {bundle_path}: {type(bundle_obj)}")

            # Decide which parameter prefixes to copy
            prefixes = []
            for part in (load_parts or []):
                p = str(part).strip().lower()
                if p in {"phi_f", "phif", "source"}:
                    prefixes.append("phi_f.")
                elif p in {"phi_r", "phir", "range"}:
                    prefixes.append("phi_r.")
                elif p in {"interaction", "psi", "psi_rf", "coupling"}:
                    prefixes.append("interaction.")
                elif p in {"bias"}:
                    prefixes.append("bias")

            own = sonarkad.state_dict()
            copied = 0
            skipped = 0
            for k, v in state_dict.items():
                if not any(str(k).startswith(pref) for pref in prefixes):
                    continue
                if k in own and tuple(own[k].shape) == tuple(v.shape):
                    own[k] = v
                    copied += 1
                else:
                    skipped += 1
            sonarkad.load_state_dict(own, strict=False)

            # Freeze requested parts
            for part in (freeze_parts or []):
                p = str(part).strip().lower()
                mod = None
                if p in {"phi_f", "phif", "source"}:
                    mod = sonarkad.phi_f
                elif p in {"phi_r", "phir", "range"}:
                    mod = sonarkad.phi_r
                elif p in {"interaction", "psi", "psi_rf", "coupling"}:
                    mod = sonarkad.interaction
                elif p in {"bias"}:
                    # bias is a parameter, not a module
                    sonarkad.bias.requires_grad_(False)
                    continue

                if mod is None:
                    continue
                for pp in mod.parameters():
                    pp.requires_grad_(False)

            print(
                f"[INFO] Transfer init from {bundle_path.name}: copied={copied} skipped={skipped} load_parts={list(load_parts)} freeze_parts={list(freeze_parts)}"
            )
        except Exception as e:
            print(f"[WARN] transfer init requested but failed: {e}. Proceeding without transfer.")

    mlp = SmallMLP(hidden=int(train_cfg.get("hidden_mlp", 32)))

    # ------------------------------------------------------------------
    # Training schedule + regularizers
    # ------------------------------------------------------------------
    lr = float(train_cfg.get("lr", 2e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))

    stage1_epochs = int(train_cfg.get("stage1_epochs", 50))
    stage2_epochs = int(train_cfg.get("stage2_epochs", 150))

    lambda_int = float(train_cfg.get("lambda_interaction_l2", 0.0))

    lambda_spline_roughness = float(train_cfg.get("lambda_spline_roughness", 0.0) or 0.0)
    spline_roughness_order = int(train_cfg.get("spline_roughness_order", 2) or 2)

    # Loss: mse | huber
    loss_type = str(train_cfg.get("loss", "mse")).strip().lower()
    huber_delta = float(train_cfg.get("huber_delta", train_cfg.get("huber_beta", 1.0)))

    # Optional monotonicity prior on φ_r (encourage decreasing with range)
    mono_cfg = train_cfg.get("range_monotonic_reg", train_cfg.get("monotone_r", {}))
    if not isinstance(mono_cfg, dict):
        mono_cfg = {}
    mono_enabled = bool(mono_cfg.get("enabled", False))
    mono_weight = float(mono_cfg.get("weight", 0.0))
    mono_grid_n = int(mono_cfg.get("grid_n", 64))

    # Gradient clipping (helps stability when interaction rank is large)
    grad_clip = float(train_cfg.get("grad_clip_norm", 0.0) or 0.0)

    # Gauge fix frequency (in addition to optional per-epoch gauge fix)
    gauge_fix_every = int(train_cfg.get("gauge_fix_every", 0) or 0)

    # ------------------------------------------------------------------
    # WGI regularization (only meaningful when interaction is enabled)
    # ------------------------------------------------------------------
    wgi_reg_cfg = train_cfg.get("wgi_reg", {})
    wgi_reg_cfg = dict(wgi_reg_cfg) if isinstance(wgi_reg_cfg, dict) else {}

    if bool(wgi_reg_cfg.get("enabled", False)):
        beta_mode = str(wgi_reg_cfg.get("beta_mode", "from_baseline")).strip().lower()
        if beta_mode in {"fixed", "manual"}:
            beta_fixed = wgi_reg_cfg.get("beta", wgi_reg_cfg.get("beta_fixed", 0.6))
            wgi_reg_cfg["beta"] = float(beta_fixed)
            wgi_reg_cfg["_beta_source"] = "fixed"
        else:
            # Estimate β from data via a waveguide-invariant baseline on a de-trended residual.
            try:
                residual = y_flat.astype(np.float64).copy()

                # Optional: de-trend using parametric TL (if enabled in baselines)
                base_cfg = exp_cfg.get("baselines", {}) or {}
                p_base = base_cfg.get("parametric_tl", {}) if isinstance(base_cfg.get("parametric_tl", {}), dict) else {}
                if bool(p_base.get("run", p_base.get("enabled", True))):
                    p_cfg_dict = p_base.get("config", {}) if isinstance(p_base.get("config", {}), dict) else {}
                    p_cfg = ParametricTLConfig(**p_cfg_dict)
                    p_info, p_pred = fit_parametric_tl(r_rep, f_tile, y_flat, train_mask, cfg=p_cfg)
                    residual = residual - p_pred(r_rep, f_tile)

                w_base = base_cfg.get("waveguide_invariant", {}) if isinstance(base_cfg.get("waveguide_invariant", {}), dict) else {}
                w_cfg_dict = w_base.get("config", {}) if isinstance(w_base.get("config", {}), dict) else {}
                w_cfg = WaveguideInvariantConfig(**w_cfg_dict)

                w_info, _ = fit_waveguide_invariant_striation(r_rep, f_tile, residual, train_mask, cfg=w_cfg)
                beta_hat = float(w_info.get("best_beta"))

                wgi_reg_cfg["beta"] = beta_hat
                wgi_reg_cfg["_beta_source"] = "waveguide_invariant_baseline"
                print(f"[INFO] WGI regularizer uses beta={beta_hat:.3f} (source={wgi_reg_cfg['_beta_source']})")
            except Exception as e:
                beta_fallback = float(wgi_reg_cfg.get("beta", wgi_reg_cfg.get("beta_fixed", 0.6)))
                wgi_reg_cfg["beta"] = beta_fallback
                wgi_reg_cfg["_beta_source"] = "fallback"
                print(f"[WARN] WGI beta estimate failed ({e}); using beta={beta_fallback:.3f}")
    else:
        wgi_reg_cfg["beta"] = None
        wgi_reg_cfg["_beta_source"] = "disabled"

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------
    opt_sonarkad = optim.Adam([p for p in sonarkad.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    opt_mlp = optim.Adam(mlp.parameters(), lr=lr, weight_decay=weight_decay)

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _eval_rmse_ev(model: nn.Module, loader: DataLoader) -> Tuple[float, float]:
        model.eval()
        y_true_list = []
        y_pred_list = []
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb)
            y_true_list.append(yb.detach().cpu().numpy())
            y_pred_list.append(pred.detach().cpu().numpy())
        y_true = np.concatenate(y_true_list).ravel()
        y_pred = np.concatenate(y_pred_list).ravel()
        mse = float(np.mean((y_true - y_pred) ** 2))
        rmse = float(np.sqrt(mse))
        v = float(np.var(y_true)) + 1e-12
        ev = float(1.0 - np.var(y_true - y_pred) / v)
        return rmse, ev

    def _loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if loss_type in {"huber", "smoothl1", "smooth_l1"}:
            # Huber / SmoothL1
            d = float(huber_delta)
            err = pred - target
            abs_err = torch.abs(err)
            quad = torch.minimum(abs_err, pred.new_tensor(d))
            # 0.5*quad^2 + d*(abs_err-quad)
            return torch.mean(0.5 * quad * quad + d * (abs_err - quad))
        # default: MSE
        return torch.mean((pred - target) ** 2)

    def _range_monotonic_penalty(model: nn.Module) -> torch.Tensor:
        if (not mono_enabled) or (mono_weight <= 0.0):
            return torch.tensor(0.0, device=device)
        if not hasattr(model, "phi_r"):
            return torch.tensor(0.0, device=device)
        # Use a fixed grid so the penalty is stable and comparable across batches.
        rg = torch.linspace(0.0, 1.0, int(max(8, mono_grid_n)), device=device, dtype=torch.float32).reshape(-1, 1)
        pr = model.phi_r(rg).view(-1)
        dpr = pr[1:] - pr[:-1]
        # Penalize positive slope (φ_r should decrease with range)
        return torch.mean(torch.relu(dpr) ** 2)

    def train_one(
        model: nn.Module,
        optimizer: optim.Optimizer,
        *,
        epochs: int,
        freeze_interaction: bool = False,
        wgi_cfg: Optional[dict] = None,
        model_label: str = "",
    ) -> Dict[str, object]:
        model = model.to(device)

        # Optional torch.compile acceleration (PyTorch 2.x). We keep the original
        # module reference for state_dict compatibility; the optimizer can still
        # update parameters because the compiled wrapper uses the same parameter
        # objects.
        model_train = model
        if bool(train_cfg.get("compile", False)) and hasattr(torch, "compile"):
            try:
                model_train = torch.compile(model)  # type: ignore[attr-defined]
            except Exception:
                model_train = model

        # Freeze/unfreeze interaction parameters if present
        if hasattr(model, "interaction") and getattr(model, "interaction") is not None:
            for p in model.interaction.parameters():
                p.requires_grad = (not freeze_interaction)

        # Early stopping applies only when interaction is trainable (stage-2)
        es = None
        eval_every = int(es_cfg.get("eval_every", 5)) if isinstance(es_cfg, dict) else 5
        if (not freeze_interaction) and (val_loader is not None) and bool(es_cfg.get("enabled", False)):
            es = EarlyStopper(
                patience=int(es_cfg.get("patience", 50)),
                min_delta=float(es_cfg.get("min_delta", 0.0)),
            )

        # WGI regularization (only when interaction is trainable)
        wgi_cfg = dict(wgi_cfg or {})
        wgi_enabled = bool(wgi_cfg.get("enabled", False)) and (not freeze_interaction)
        wgi_weight = float(wgi_cfg.get("weight", 0.0) or 0.0)
        wgi_beta = wgi_cfg.get("beta", None)
        wgi_step_logr = float(wgi_cfg.get("step_logr", 0.02))
        wgi_warmup = int(wgi_cfg.get("warmup_epochs", 0) or 0)

        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        autocast_ctx = torch.cuda.amp.autocast if use_amp else (lambda **kwargs: nullcontext())

        hist: Dict[str, object] = {
            "epoch": [],
            "train_loss": [],
            "val_rmse": [],
            "val_ev": [],
        }

        epoch_iter = tqdm(range(int(epochs)), desc=model_label or "train", disable=not progress)
        for epoch in epoch_iter:
            model_train.train()
            losses = []

            batch_iter = train_loader
            if progress:
                batch_iter = tqdm(train_loader, desc="batches", leave=False, disable=not progress)

            for xb, yb in batch_iter:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with autocast_ctx():
                    pred = model_train(xb)
                    loss = _loss(pred, yb)

                    # Spline roughness regularization (curvature proxy on B-spline coefficients)
                    if lambda_spline_roughness > 0.0 and hasattr(model, 'spline_roughness_penalty'):
                        try:
                            loss = loss + (lambda_spline_roughness * model.spline_roughness_penalty(
                                order=spline_roughness_order,
                                include_interaction=(not freeze_interaction),
                            ))
                        except Exception:
                            pass

                    # Interaction energy regularization
                    if lambda_int > 0.0 and hasattr(model, "forward_components"):
                        comps = model.forward_components(xb)
                        psi = comps.get("psi_rf", None)
                        if psi is not None:
                            loss = loss + (lambda_int * torch.mean(psi ** 2))

                    # Range monotonicity prior (applies to SonarKAD only)
                    if mono_enabled and mono_weight > 0.0:
                        loss = loss + (mono_weight * _range_monotonic_penalty(model))

                    # Waveguide-invariant regularization on ψ(r,f)
                    if (
                        wgi_enabled
                        and wgi_weight > 0.0
                        and (wgi_beta is not None)
                        and hasattr(model, "forward_components")
                    ):
                        if wgi_warmup > 0 and epoch < wgi_warmup:
                            ramp = float(epoch + 1) / float(wgi_warmup)
                        else:
                            ramp = 1.0
                        wgi_pen = wgi_invariance_penalty(
                            model,
                            xb,
                            beta=float(wgi_beta),
                            r_min_m=float(r_min),
                            r_max_m=float(r_max),
                            f_min_hz=float(f_min),
                            f_max_hz=float(f_max),
                            step_logr=wgi_step_logr,
                        )
                        loss = loss + (ramp * wgi_weight * wgi_pen)

                if use_amp:
                    scaler.scale(loss).backward()
                    if grad_clip > 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                    optimizer.step()

                losses.append(float(loss.detach().cpu().item()))

            # Periodic gauge fix
            if gauge_fix_every > 0 and hasattr(model, "gauge_fix") and (epoch + 1) % gauge_fix_every == 0:
                try:
                    model.gauge_fix()
                except Exception as e:
                    print(f"[WARN] gauge_fix failed: {e}")

            hist["epoch"].append(int(epoch))
            hist["train_loss"].append(float(np.mean(losses)) if losses else float("nan"))

            # Update outer progress bar
            try:
                epoch_iter.set_postfix(train_loss=f"{hist['train_loss'][-1]:.3g}")
            except Exception:
                pass

            if val_loader is not None and (epoch % eval_every == 0 or epoch == epochs - 1):
                model_train.eval()
                val_rmse, val_ev = _eval_rmse_ev(model_train, val_loader)
                hist["val_rmse"].append(float(val_rmse))
                hist["val_ev"].append(float(val_ev))

                try:
                    epoch_iter.set_postfix(
                        train_loss=f"{hist['train_loss'][-1]:.3g}",
                        val_rmse=f"{float(val_rmse):.3g}",
                    )
                except Exception:
                    pass

                if es is not None:
                    if es.step(float(val_rmse), model, epoch):
                        print(
                            f"[early-stop] {model_label} stop at epoch={epoch} "
                            f"(best={es.best:.4f} @ {es.best_epoch})"
                        )
                        break

            # Optional: gauge-fix every epoch (for interpretability)
            if hasattr(model, "gauge_fix") and getattr(model, "cfg", None) is not None:
                if bool(getattr(model.cfg, "gauge_fix_each_epoch", False)):
                    try:
                        model.gauge_fix()
                    except Exception:
                        pass

        if es is not None:
            es.restore_best(model)

        return hist

    # ------------------------------------------------------------------
    # Fit models
    # ------------------------------------------------------------------
    hist_sonarkad_stage1 = train_one(
        sonarkad,
        opt_sonarkad,
        epochs=stage1_epochs,
        freeze_interaction=True,
        wgi_cfg=wgi_reg_cfg,
        model_label="SonarKAD stage-1",
    )

    hist_sonarkad_stage2 = train_one(
        sonarkad,
        opt_sonarkad,
        epochs=stage2_epochs,
        freeze_interaction=False,
        wgi_cfg=wgi_reg_cfg,
        model_label="SonarKAD stage-2",
    )

    hist_mlp = train_one(
        mlp,
        opt_mlp,
        epochs=stage1_epochs + stage2_epochs,
        freeze_interaction=False,
        wgi_cfg=None,
        model_label="MLP",
    )

    rmse_sonarkad, ev_sonarkad = _eval_rmse_ev(sonarkad, test_loader)
    rmse_mlp, ev_mlp = _eval_rmse_ev(mlp, test_loader)

    # Optional ablation: additive-only SonarKAD (interaction_rank=0)
    base_cfg = exp_cfg.get("baselines", {}) or {}
    ab_cfg = base_cfg.get("additive_ablation", {}) if isinstance(base_cfg.get("additive_ablation", {}), dict) else {}
    run_additive_ablation = bool(ab_cfg.get("run", sonarkad_cfg.interaction_rank > 0))

    rmse_additive = None
    ev_additive = None
    if run_additive_ablation:
        sonarkad_cfg0 = SonarKADConfig(**{**sonarkad_cfg.__dict__, "interaction_rank": 0})
        sonarkad_add = SonarKAD(r_min_m=r_min, r_max_m=r_max, cfg=sonarkad_cfg0)
        sonarkad_add.physics_init()
        opt0 = optim.Adam([p for p in sonarkad_add.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
        _ = train_one(
            sonarkad_add,
            opt0,
            epochs=stage1_epochs + stage2_epochs,
            freeze_interaction=False,
            wgi_cfg=None,
            model_label="SonarKAD-additive",
        )
        rmse_additive, ev_additive = _eval_rmse_ev(sonarkad_add, test_loader)

    # Coupling energy fraction on the test set
    coupling_energy_frac_test = None
    coupling_energy_test = None
    total_energy_test = None
    if getattr(sonarkad, "interaction", None) is not None:
        sonarkad.eval()
        with torch.no_grad():
            comps = sonarkad.forward_components(X_test.to(device))
            phi = (comps["phi_r"] + comps["phi_f"]).detach().cpu().numpy().reshape(-1)
            psi = comps["psi_rf"].detach().cpu().numpy().reshape(-1)

        psi_c = psi - float(np.mean(psi))
        total_c = (phi + psi) - float(np.mean(phi + psi))
        coupling_energy_test = float(np.mean(psi_c ** 2))
        total_energy_test = float(np.mean(total_c ** 2))
        coupling_energy_frac_test = float(coupling_energy_test / (total_energy_test + 1e-12))

    # ------------------------------------------------------------------
    # Traditional baselines
    # ------------------------------------------------------------------
    y_true_test = y_flat[test_mask].astype(np.float64)
    r_test_m = r_rep[test_mask].astype(np.float64)
    f_test_hz = f_tile[test_mask].astype(np.float64)

    rmse_param = ev_param = None
    param_info = None
    pred_param = None
    if bool(base_cfg.get("parametric_tl", {}).get("run", True)):
        p_cfg = ParametricTLConfig(**base_cfg.get("parametric_tl", {}).get("config", {}))
        param_info, pred_param = fit_parametric_tl(r_rep, f_tile, y_flat, train_mask, cfg=p_cfg)
        yhat_test = pred_param(r_test_m, f_test_hz)
        rmse_param = float(np.sqrt(np.mean((y_true_test - yhat_test) ** 2)))
        ev_param = float(explained_variance(y_true_test, yhat_test))

    rmse_gam = ev_gam = None
    gam_info = None
    pred_gam = None
    yhat_all_gam = None
    if bool(base_cfg.get("gam", {}).get("run", True)):
        g_cfg = GAMConfig(**base_cfg.get("gam", {}).get("config", {}))
        gam_info, pred_gam = fit_gam_spline(r_norm, f_norm, y_flat, train_mask, cfg=g_cfg)
        yhat_test = pred_gam(r_norm[test_mask], f_norm[test_mask])
        rmse_gam = float(np.sqrt(np.mean((y_true_test - yhat_test) ** 2)))
        ev_gam = float(explained_variance(y_true_test, yhat_test))
        yhat_all_gam = pred_gam(r_norm, f_norm)

    # Choose additive base for residuals
    yhat_all_param = None
    if pred_param is not None:
        try:
            yhat_all_param = pred_param(r_rep, f_tile).astype(np.float64)
        except Exception:
            yhat_all_param = None

    base_pref = str(base_cfg.get("striations_base", "parametric_tl")).strip().lower()

    def _is_reasonable_rmse(v: object) -> bool:
        try:
            x = float(v)
            return np.isfinite(x) and (x >= 0.0) and (x < 200.0)
        except Exception:
            return False

    yhat_base = None
    base_name = None

    if base_pref in {"gam", "auto"} and (yhat_all_gam is not None):
        if base_pref == "gam":
            yhat_base = yhat_all_gam.astype(np.float64)
            base_name = "gam"
        else:
            rmse_gam_tr = None
            try:
                yhat_gam_tr = pred_gam(r_norm[train_mask], f_norm[train_mask]) if pred_gam is not None else None
                if yhat_gam_tr is not None:
                    rmse_gam_tr = float(np.sqrt(np.mean((y_flat[train_mask] - yhat_gam_tr) ** 2)))
            except Exception:
                rmse_gam_tr = None

            rmse_param_tr = None
            if yhat_all_param is not None:
                try:
                    yhat_param_tr = yhat_all_param[train_mask]
                    rmse_param_tr = float(np.sqrt(np.mean((y_flat[train_mask] - yhat_param_tr) ** 2)))
                except Exception:
                    rmse_param_tr = None

            yhat_base = yhat_all_param if (yhat_all_param is not None) else yhat_all_gam
            base_name = "parametric_tl" if (yhat_all_param is not None) else "gam"

            if _is_reasonable_rmse(rmse_gam_tr) and _is_reasonable_rmse(rmse_param_tr):
                if rmse_gam_tr <= rmse_param_tr:
                    yhat_base = yhat_all_gam
                    base_name = "gam"

    if yhat_base is None:
        if yhat_all_param is not None:
            yhat_base = yhat_all_param
            base_name = "parametric_tl"
        elif yhat_all_gam is not None:
            yhat_base = yhat_all_gam.astype(np.float64)
            base_name = "gam"
        else:
            yhat_base = np.full_like(y_flat, float(np.mean(y_flat[train_mask])), dtype=np.float64)
            base_name = "mean"

    residual = (y_flat.astype(np.float64) - yhat_base).astype(np.float64)

    # Waveguide-invariant residual fit
    rmse_wg = ev_wg = None
    wg_info = None
    pred_wg = None
    coupling_energy_frac_wg_test = None
    if bool(base_cfg.get("waveguide_invariant", {}).get("run", True)):
        w_cfg = WaveguideInvariantConfig(**base_cfg.get("waveguide_invariant", {}).get("config", {}))
        wg_info, pred_wg = fit_waveguide_invariant_striation(r_rep, f_tile, residual, train_mask, cfg=w_cfg)
        wg_res_test = pred_wg(r_test_m, f_test_hz).astype(np.float64)
        yhat_test = (yhat_base[test_mask] + wg_res_test).astype(np.float64)
        rmse_wg = float(np.sqrt(np.mean((y_true_test - yhat_test) ** 2)))
        ev_wg = float(explained_variance(y_true_test, yhat_test))
        coupling_energy_frac_wg_test = float(
            np.mean((wg_res_test - np.mean(wg_res_test)) ** 2)
            / (np.mean((yhat_test - np.mean(yhat_test)) ** 2) + 1e-12)
        )
        if isinstance(wg_info, dict):
            wg_info["base_model"] = base_name
            wg_info["coupling_energy_frac_test"] = coupling_energy_frac_wg_test

    # Constant-c modal residual fit
    rmse_modal = ev_modal = None
    modal_info = None
    pred_modal = None
    coupling_energy_frac_modal_test = None
    if bool(base_cfg.get("modal_striation", {}).get("run", True)):
        m_cfg_dict = base_cfg.get("modal_striation", {}).get("config", {})
        m_cfg = PekerisModalStriationConfig(**m_cfg_dict)
        if getattr(m_cfg, "water_depth_m", None) is None:
            m_cfg.water_depth_m = water_depth_m
        if getattr(m_cfg, "c0_m_per_s", None) is None:
            m_cfg.c0_m_per_s = c0_m_per_s

        modal_info, pred_modal = fit_pekeris_modal_striation(r_rep, f_tile, residual, train_mask, cfg=m_cfg)
        modal_res_test = pred_modal(r_test_m, f_test_hz).astype(np.float64)
        yhat_test = (yhat_base[test_mask] + modal_res_test).astype(np.float64)
        rmse_modal = float(np.sqrt(np.mean((y_true_test - yhat_test) ** 2)))
        ev_modal = float(explained_variance(y_true_test, yhat_test))
        coupling_energy = float(np.mean((modal_res_test - np.mean(modal_res_test)) ** 2))
        total_energy = float(np.mean((yhat_test - np.mean(yhat_test)) ** 2))
        coupling_energy_frac_modal_test = float(coupling_energy / (total_energy + 1e-12))
        if isinstance(modal_info, dict):
            modal_info["base_model"] = base_name
            modal_info["coupling_energy_frac_test"] = coupling_energy_frac_modal_test

    # CTD-profile modal residual fit
    rmse_modal_profile = ev_modal_profile = None
    modal_profile_info = None
    pred_modal_profile = None
    coupling_energy_frac_modal_profile_test = None
    if bool(base_cfg.get("modal_striation_profile", {}).get("run", True)):
        if (z_profile_m is None) or (c_profile_mps is None):
            print("[WARN] modal_striation_profile requested but no CTD profile is available; skipping.")
        else:
            mp_cfg = ProfileModalStriationConfig(**base_cfg.get("modal_striation_profile", {}).get("config", {}))
            if getattr(mp_cfg, "water_depth_m", None) is None:
                mp_cfg.water_depth_m = water_depth_m
            modal_profile_info, pred_modal_profile = fit_profile_modal_striation(
                r_rep,
                f_tile,
                residual,
                train_mask,
                z_profile_m=z_profile_m,
                c_profile_mps=c_profile_mps,
                cfg=mp_cfg,
            )
            modal_prof_res_test = pred_modal_profile(r_test_m, f_test_hz).astype(np.float64)
            yhat_test = (yhat_base[test_mask] + modal_prof_res_test).astype(np.float64)
            rmse_modal_profile = float(np.sqrt(np.mean((y_true_test - yhat_test) ** 2)))
            ev_modal_profile = float(explained_variance(y_true_test, yhat_test))

            coupling_energy = float(np.mean((modal_prof_res_test - np.mean(modal_prof_res_test)) ** 2))
            total_energy = float(np.mean((yhat_test - np.mean(yhat_test)) ** 2))
            coupling_energy_frac_modal_profile_test = float(coupling_energy / (total_energy + 1e-12))
            if isinstance(modal_profile_info, dict):
                modal_profile_info["base_model"] = base_name
                modal_profile_info["coupling_energy_frac_test"] = coupling_energy_frac_modal_profile_test

    # ------------------------------------------------------------------
    # Learned components on grids (for plotting)
    # ------------------------------------------------------------------
    diag_cfg = exp_cfg.get("diagnostics", {}) or {}
    grid_n_r = int(diag_cfg.get("grid_n_r", 200))
    grid_n_f = int(diag_cfg.get("grid_n_f", 200))

    r_grid_m = np.linspace(r_min, r_max, grid_n_r, dtype=np.float64)
    f_grid_hz = np.linspace(f_min, f_max, grid_n_f, dtype=np.float64)

    r_grid = ((r_grid_m - r_min) / (r_max - r_min + 1e-12)).astype(np.float32).reshape(-1, 1)
    f_grid = ((f_grid_hz - f_min) / (f_max - f_min + 1e-12)).astype(np.float32).reshape(-1, 1)

    sonarkad.eval()
    with torch.no_grad():
        r_t = torch.from_numpy(r_grid).to(device)
        f_t = torch.from_numpy(f_grid).to(device)
        phi_r = sonarkad.phi_r(r_t).detach().cpu().numpy().reshape(-1)
        phi_f = sonarkad.phi_f(f_t).detach().cpu().numpy().reshape(-1)

        phi_abs_rf = None
        if getattr(sonarkad, "absorption_term", None) is not None:
            rr = np.repeat(r_grid, grid_n_f, axis=0)
            ff = np.tile(f_grid, (grid_n_r, 1))
            rr_t = torch.from_numpy(rr.astype(np.float32)).to(device)
            ff_t = torch.from_numpy(ff.astype(np.float32)).to(device)
            phi_abs_rf = (
                sonarkad.absorption_term(rr_t, ff_t).detach().cpu().numpy().reshape(grid_n_r, grid_n_f)
            )

        psi_rf = None
        if sonarkad.interaction is not None:
            rr = np.repeat(r_grid, grid_n_f, axis=0)
            ff = np.tile(f_grid, (grid_n_r, 1))
            rr_t = torch.from_numpy(rr.astype(np.float32)).to(device)
            ff_t = torch.from_numpy(ff.astype(np.float32)).to(device)
            psi_rf = sonarkad.interaction(rr_t, ff_t).detach().cpu().numpy().reshape(grid_n_r, grid_n_f)

    # Test predictions for scatter/diagnostics
    sonarkad.eval()
    with torch.no_grad():
        y_pred_test = sonarkad(X_test.to(device)).detach().cpu().numpy().reshape(-1).astype(np.float32)

    # ------------------------------------------------------------------
    # β diagnostics (structure tensor in log r/log f)
    # ------------------------------------------------------------------
    beta_diag: Dict[str, float] = {}
    try:
        if psi_rf is not None and psi_rf.size > 0:
            beta_diag["beta_interaction_tensor"] = float(
                estimate_beta_structure_tensor(psi_rf, r_grid_m, f_grid_hz)
            )
    except Exception as e:
        print(f"[WARN] beta estimation on interaction term failed: {e}")

    if isinstance(wg_info, dict) and ("best_beta" in wg_info):
        try:
            beta_diag["beta_waveguide_invariant_best"] = float(wg_info["best_beta"])
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Persist artifacts
    # ------------------------------------------------------------------
    components = {
        "grid_r_m": r_grid_m,
        "grid_f_hz": f_grid_hz,
        "phi_r": phi_r.astype(np.float32),
        "phi_f": phi_f.astype(np.float32),
        "phi_abs_rf": (
            phi_abs_rf.astype(np.float32)
            if (phi_abs_rf is not None)
            else np.zeros((0, 0), dtype=np.float32)
        ),
        "psi_rf": (psi_rf.astype(np.float32) if psi_rf is not None else np.zeros((0, 0), dtype=np.float32)),
        "y_pred_test": y_pred_test,
        "y_test": y_flat[test_mask].astype(np.float32),
        "r_test_m": r_test_m.astype(np.float32),
        "f_test_hz": f_test_hz.astype(np.float32),
        "beta_waveguide_invariant_best": float(beta_diag.get("beta_waveguide_invariant_best", float("nan"))),
        "beta_interaction_tensor": float(beta_diag.get("beta_interaction_tensor", float("nan"))),
        "meta": {
            "exp_name": exp_name,
            "event": event,
            "array": array,
            "tone_set": tone_set,
            "seed": seed,
            "n_train_frames": int(n_train_frames),
            "n_test_frames": int(n_test_frames),
            "T": int(T),
            "F": int(F),
        },
    }

    comp_path = Path(run_dir) / "components.pt"
    torch.save(components, comp_path)

    # Deployable model bundle
    bundle = {
        "format": "SonarKAD_model_bundle",
        "exp_name": exp_name,
        "event": event,
        "array": array,
        "tone_set": tone_set,
        "state_dict": sonarkad.state_dict(),
        "model_cfg": asdict(sonarkad_cfg),
        "normalization": {
            "r_min_m": float(r_min),
            "r_max_m": float(r_max),
            "f_min_hz": float(f_min),
            "f_max_hz": float(f_max),
        },
        "training_meta": {
            "seed": seed,
            "device": str(device),
            "amp": bool(use_amp),
            "loss": loss_type,
            "huber_delta": float(huber_delta),
            "range_monotonic_reg": {
                "enabled": bool(mono_enabled),
                "weight": float(mono_weight),
                "grid_n": int(mono_grid_n),
            },
            "early_stopping": es_cfg if isinstance(es_cfg, dict) else {},
            "transfer": transfer_cfg if isinstance(transfer_cfg, dict) else {},
        },
        "wgi_regularization": {k: v for k, v in (wgi_reg_cfg or {}).items() if not str(k).startswith("_")},
    }
    torch.save(bundle, Path(run_dir) / "sonarkad_model.pt")

    # JSON summary (scalars + provenance only; arrays in components.pt)
    results: Dict[str, object] = {
        "exp_name": exp_name,
        "event": event,
        "array": array,
        "tone_set": tone_set,
        "seed": seed,

        "processed_npz": str(npz_path),
        "components_path": str(comp_path),

        "rmse_test_sonarkad": float(rmse_sonarkad),
        "ev_test_sonarkad": float(ev_sonarkad),
        "rmse_test_mlp": float(rmse_mlp),
        "ev_test_mlp": float(ev_mlp),

        "rmse_test_additive": float(rmse_additive) if rmse_additive is not None else None,
        "ev_test_additive": float(ev_additive) if ev_additive is not None else None,

        "rmse_test_parametric_tl": float(rmse_param) if rmse_param is not None else None,
        "ev_test_parametric_tl": float(ev_param) if ev_param is not None else None,
        "rmse_test_gam": float(rmse_gam) if rmse_gam is not None else None,
        "ev_test_gam": float(ev_gam) if ev_gam is not None else None,

        "rmse_test_waveguide_invariant": float(rmse_wg) if rmse_wg is not None else None,
        "ev_test_waveguide_invariant": float(ev_wg) if ev_wg is not None else None,
        "coupling_energy_frac_waveguide_invariant_test": float(coupling_energy_frac_wg_test) if coupling_energy_frac_wg_test is not None else None,

        "rmse_test_modal_striation": float(rmse_modal) if rmse_modal is not None else None,
        "ev_test_modal_striation": float(ev_modal) if ev_modal is not None else None,
        "coupling_energy_frac_modal_striation_test": float(coupling_energy_frac_modal_test) if coupling_energy_frac_modal_test is not None else None,

        "rmse_test_modal_striation_profile": float(rmse_modal_profile) if rmse_modal_profile is not None else None,
        "ev_test_modal_striation_profile": float(ev_modal_profile) if ev_modal_profile is not None else None,
        "coupling_energy_frac_modal_striation_profile_test": float(coupling_energy_frac_modal_profile_test) if coupling_energy_frac_modal_profile_test is not None else None,

        "coupling_energy_frac_test": float(coupling_energy_frac_test) if coupling_energy_frac_test is not None else None,
        "coupling_energy_test": float(coupling_energy_test) if coupling_energy_test is not None else None,
        "total_energy_test": float(total_energy_test) if total_energy_test is not None else None,

        "beta_diagnostics": beta_diag,

        "baseline_details": {
            "parametric_tl": param_info,
            "gam": gam_info,
            "waveguide_invariant": wg_info,
            "modal_striation": modal_info,
            "modal_striation_profile": modal_profile_info,
        },

        "environment": {
            "water_depth_m": float(water_depth_m),
            "c0_m_per_s": float(c0_m_per_s),
            "ctd_dir": str(ctd_dir) if ctd_dir is not None else None,
            "ctd_statistic": ctd_stat,
            "ctd_stations": ctd_stations,
        },

        "params_sonarkad": int(count_parameters(sonarkad)),
        "params_mlp": int(count_parameters(mlp)),

        "normalization": {
            "r_min_m": float(r_min),
            "r_max_m": float(r_max),
            "f_min_hz": float(f_min),
            "f_max_hz": float(f_max),
        },

        "split": {
            "train_frames": int(n_train_frames),
            "test_frames": int(n_test_frames),
            "val_frames": int(np.sum(np.unique(frame_idx[val_mask]))) if np.any(val_mask) else 0,
            "T": int(T),
            "F": int(F),
        },

        "training": {
            "loss": loss_type,
            "huber_delta": float(huber_delta),
            "range_monotonic_reg": {
                "enabled": bool(mono_enabled),
                "weight": float(mono_weight),
                "grid_n": int(mono_grid_n),
            },
            "lambda_interaction_l2": float(lambda_int),
            "grad_clip_norm": float(grad_clip),
            "wgi_reg": {k: v for k, v in (wgi_reg_cfg or {}).items() if not str(k).startswith("_")},
            "early_stopping": es_cfg if isinstance(es_cfg, dict) else {},
            "transfer": transfer_cfg if isinstance(transfer_cfg, dict) else {},
            "history_sonarkad": {"stage1": hist_sonarkad_stage1, "stage2": hist_sonarkad_stage2},
            "history_mlp": hist_mlp,
        },
    }

    (Path(run_dir) / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    return results
def train_swellex96_cv(exp_cfg: dict, run_dir: str | Path, *, force: bool = False) -> Dict[str, object]:
    """Blocked (time-contiguous) cross-validation for SWellEx-96 experiments."""

    run_dir = Path(run_dir)
    ensure_dir(run_dir)

    # Idempotency: if a completed CV summary already exists, reuse it.
    out_json = run_dir / "results_cv.json"
    if out_json.exists() and (not force):
        try:
            return json.loads(out_json.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Load processed dataset once to determine frame count.
    npz_path = _ensure_processed_npz(exp_cfg)
    _, _, _, y_tf = _flatten_processed(npz_path)
    T = int(y_tf.shape[0])

    split_cfg = exp_cfg.get("split", {}) or {}
    cv_cfg = exp_cfg.get("block_cv", None)
    if cv_cfg is None:
        cv_cfg = split_cfg.get("block_cv", {})
    cv_cfg = cv_cfg or {}

    n_folds = int(cv_cfg.get("n_folds", cv_cfg.get("n_splits", 5)))
    mode = str(cv_cfg.get("mode", "kfold")).strip().lower()
    seed = int(cv_cfg.get("seed", 0))

    block_len = int(cv_cfg.get("block_len_frames", 0) or 0)

    # Build contiguous blocks in frame-index space
    blocks: list[np.ndarray] = []
    if block_len > 0:
        n_blocks = int(math.ceil(T / float(block_len)))
        for b in range(n_blocks):
            a = int(b * block_len)
            c = int(min((b + 1) * block_len, T))
            if c > a:
                blocks.append(np.arange(a, c, dtype=int))
    else:
        n_blocks = int(max(2, cv_cfg.get("n_blocks", 10)))
        n_blocks = max(2, min(n_blocks, T))
        edges = np.linspace(0, T, n_blocks + 1).astype(int)
        edges[0] = 0
        edges[-1] = T
        for i in range(n_blocks):
            a, c = int(edges[i]), int(edges[i + 1])
            if c > a:
                blocks.append(np.arange(a, c, dtype=int))

    if len(blocks) < 2:
        raise ValueError(f"Block-CV failed: could not form >=2 non-empty blocks (T={T}).")

    n_blocks_eff = len(blocks)

    # Determine test-block sets
    test_blocks_per_fold = int(cv_cfg.get("test_blocks", cv_cfg.get("test_blocks_per_split", 1)))
    test_blocks_per_fold = max(1, min(test_blocks_per_fold, n_blocks_eff - 1))

    rng = np.random.default_rng(seed)
    test_block_sets: list[list[int]] = []

    if mode == "kfold" and test_blocks_per_fold == 1:
        block_ids = np.arange(n_blocks_eff)
        if bool(cv_cfg.get("shuffle", True)):
            rng.shuffle(block_ids)
        fold_groups = np.array_split(block_ids, max(2, n_folds))
        test_block_sets = [list(map(int, g)) for g in fold_groups if len(g) > 0]
    else:
        n_folds = max(1, n_folds)
        for _ in range(n_folds):
            ids = sorted(rng.choice(n_blocks_eff, size=test_blocks_per_fold, replace=False).tolist())
            test_block_sets.append(ids)

    cv_root = run_dir / "cv"
    ensure_dir(cv_root)

    fold_results: list[Dict[str, object]] = []
    fold_summaries: list[Dict[str, object]] = []

    for k, test_ids in enumerate(test_block_sets):
        test_frames = np.zeros(T, dtype=bool)
        for bid in test_ids:
            test_frames[blocks[bid]] = True
        train_frames = ~test_frames

        fold_dir = cv_root / f"fold_{k:02d}"
        ensure_dir(fold_dir)

        res_k = train_swellex96(
            exp_cfg,
            fold_dir,
            force=force,
            frame_train_mask=train_frames,
            frame_test_mask=test_frames,
        )
        fold_results.append(res_k)
        fold_summaries.append(
            {
                "fold": int(k),
                "test_block_ids": [int(x) for x in test_ids],
                "train_frames": int(np.sum(train_frames)),
                "test_frames": int(np.sum(test_frames)),
                "rmse_test_sonarkad": float(res_k.get("rmse_test_sonarkad", float("nan"))),
                "ev_test_sonarkad": float(res_k.get("ev_test_sonarkad", float("nan"))),
            }
        )

    # Aggregate scalar metrics across folds
    def _is_num(x: object) -> bool:
        try:
            return isinstance(x, (int, float, np.floating)) and np.isfinite(float(x))
        except Exception:
            return False

    agg: Dict[str, object] = {}
    keys = set()
    for r in fold_results:
        for k in r.keys():
            if k.startswith("rmse_") or k.startswith("ev_") or k.startswith("coupling_energy_frac"):
                keys.add(k)

    for k in sorted(keys):
        vals = [float(r.get(k)) for r in fold_results if _is_num(r.get(k))]
        if len(vals) >= 2:
            agg[k] = float(np.mean(vals))
            agg[k + "_std"] = float(np.std(vals, ddof=1))
        elif len(vals) == 1:
            agg[k] = float(vals[0])
            agg[k + "_std"] = 0.0

    beta_keys = set()
    for r in fold_results:
        bd = r.get("beta_diagnostics", {})
        if isinstance(bd, dict):
            for bk, bv in bd.items():
                if _is_num(bv):
                    beta_keys.add(bk)

    beta_diag: Dict[str, float] = {}
    for bk in sorted(beta_keys):
        vals = []
        for r in fold_results:
            bd = r.get("beta_diagnostics", {})
            if isinstance(bd, dict) and _is_num(bd.get(bk)):
                vals.append(float(bd[bk]))
        if len(vals) >= 2:
            beta_diag[bk] = float(np.mean(vals))
            beta_diag[bk + "_std"] = float(np.std(vals, ddof=1))
        elif len(vals) == 1:
            beta_diag[bk] = float(vals[0])
            beta_diag[bk + "_std"] = 0.0

    agg["beta_diagnostics"] = beta_diag
    agg["cv"] = {
        "mode": mode,
        "seed": int(seed),
        "n_frames": int(T),
        "n_blocks": int(n_blocks_eff),
        "n_folds": int(len(test_block_sets)),
        "test_blocks_per_fold": int(test_blocks_per_fold),
        "block_len_frames": int(block_len),
    }
    agg["folds"] = fold_summaries

    out_json = run_dir / "results_cv.json"
    out_json.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    return agg
