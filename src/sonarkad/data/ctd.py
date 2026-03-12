"""CTD (conductivity-temperature-depth) profile utilities for SWellEx-96.

The SWellEx-96 website distributes salinity-despiked and depth-interpolated CTD
casts in plain text files named:

    i96{StationNumber}.prn

Each file contains 5 columns (README in the CTD directory):
    (1) depth [m]
    (2) temperature [deg C]
    (3) salinity [PSU]
    (4) sound speed [m/s]
    (5) sigma-t

This module provides minimal, NumPy-only helpers to:
- load one or many CTD casts;
- compute a representative sound-speed profile (mean/median);
- estimate a single effective sound speed c0 for simple baselines.

These utilities intentionally avoid any dependence on external oceanographic
packages so that the full codebase remains lightweight and easy to run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import re
import numpy as np


@dataclass
class CTDCast:
    """A single CTD cast."""

    station: int
    depth_m: np.ndarray
    temperature_c: np.ndarray
    salinity_psu: np.ndarray
    sound_speed_mps: np.ndarray
    sigma_t: np.ndarray
    meta: Dict[str, object]


def parse_ctd_station_number(path: str | Path) -> Optional[int]:
    """Extract station number from a CTD filename.

    Supports 'i9601.prn' -> 1, 'i9612.prn' -> 12.
    """
    name = Path(path).name
    m = re.search(r"i96(\d{2})\.prn$", name, flags=re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1))


def parse_ctd_readme(readme_path: str | Path) -> Dict[int, Dict[str, object]]:
    """Parse the CTD README file to obtain per-station metadata.

    The README distributed with SWellEx-96 lists station numbers and approximate
    date/time/lat/lon. We treat parsing as best-effort and store raw strings.

    Returns
    -------
    meta_by_station:
        Dict mapping station -> metadata dict.
    """
    p = Path(readme_path)
    if not p.exists():
        raise FileNotFoundError(p)

    txt = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    meta: Dict[int, Dict[str, object]] = {}

    # Find lines that begin with an integer station number.
    for ln in txt:
        s = ln.strip()
        if not s:
            continue
        if not re.match(r"^\d+\s+", s):
            continue

        # Heuristic tokenization
        parts = s.split()
        try:
            stn = int(parts[0])
        except Exception:
            continue

        # Expected columns:
        # Stn  Date(DD Mon)  Time(hh:mm)  Latitude  Longitude  Notes...
        # Latitude/Longitude appear as two tokens each: "32d39.76'" "N"
        # We'll join them.
        date_str = " ".join(parts[1:3]) if len(parts) >= 3 else ""
        time_str = parts[3] if len(parts) >= 4 else ""
        lat_str = " ".join(parts[4:6]) if len(parts) >= 6 else ""
        lon_str = " ".join(parts[6:8]) if len(parts) >= 8 else ""
        notes = " ".join(parts[8:]) if len(parts) >= 9 else ""

        meta[stn] = {
            "date": date_str,
            "time_local": time_str,
            "latitude": lat_str,
            "longitude": lon_str,
            "notes": notes,
            "raw": s,
        }

    return meta


def read_ctd_prn(path: str | Path, *, meta: Optional[Dict[str, object]] = None) -> CTDCast:
    """Load a single CTD cast file (i96XX.prn).

    The file contains whitespace-separated numeric columns with no header.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    stn = parse_ctd_station_number(p)
    if stn is None:
        # Still allow reading, but station stays -1.
        stn = -1

    arr = np.loadtxt(p, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError(f"Unexpected CTD format in {p}: got shape {arr.shape}")

    # Some files might omit sigma-t; handle gracefully.
    depth = arr[:, 0]
    temp = arr[:, 1]
    sal = arr[:, 2]
    c = arr[:, 3]
    sig = arr[:, 4] if arr.shape[1] >= 5 else np.full_like(depth, np.nan)

    # Ensure depth is increasing for interpolation.
    if depth.size >= 2 and depth[0] > depth[-1]:
        depth = depth[::-1].copy()
        temp = temp[::-1].copy()
        sal = sal[::-1].copy()
        c = c[::-1].copy()
        sig = sig[::-1].copy()

    return CTDCast(
        station=int(stn),
        depth_m=depth,
        temperature_c=temp,
        salinity_psu=sal,
        sound_speed_mps=c,
        sigma_t=sig,
        meta=dict(meta or {}),
    )


def load_ctd_casts(
    ctd_dir: str | Path,
    *,
    stations: Optional[Sequence[int]] = None,
    readme_name: str = "README",
    pattern: str = "i96*.prn",
) -> List[CTDCast]:
    """Load CTD casts from a directory.

    Parameters
    ----------
    ctd_dir:
        Directory containing i96*.prn files and a README.
    stations:
        Optional list of station numbers to load.
    """
    d = Path(ctd_dir)
    if not d.exists() or not d.is_dir():
        raise FileNotFoundError(d)

    meta_by_station: Dict[int, Dict[str, object]] = {}
    readme_path = d / readme_name
    if readme_path.exists():
        try:
            meta_by_station = parse_ctd_readme(readme_path)
        except Exception:
            meta_by_station = {}

    casts: List[CTDCast] = []
    for fp in sorted(d.glob(pattern)):
        stn = parse_ctd_station_number(fp)
        if stations is not None and stn is not None and int(stn) not in set(int(x) for x in stations):
            continue
        meta = meta_by_station.get(int(stn), {}) if stn is not None else {}
        casts.append(read_ctd_prn(fp, meta=meta))

    if not casts:
        raise FileNotFoundError(f"No CTD files matching {pattern!r} in {d}")

    return casts


def resample_cast_to_grid(depth_grid_m: np.ndarray, cast: CTDCast) -> np.ndarray:
    """Interpolate sound speed of a cast onto a given depth grid."""
    z = np.asarray(cast.depth_m, dtype=np.float64)
    c = np.asarray(cast.sound_speed_mps, dtype=np.float64)
    # np.interp requires increasing x.
    if z.size >= 2 and z[0] > z[-1]:
        z = z[::-1]
        c = c[::-1]
    return np.interp(depth_grid_m, z, c, left=c[0], right=c[-1])


def representative_sound_speed_profile(
    casts: Sequence[CTDCast],
    *,
    depth_grid_m: Optional[np.ndarray] = None,
    statistic: str = "median",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a representative sound-speed profile from multiple casts."""
    if not casts:
        raise ValueError("casts must be non-empty")

    if depth_grid_m is None:
        # Use the intersection of depth ranges to avoid extrapolation.
        z_min = max(float(np.min(c.depth_m)) for c in casts)
        z_max = min(float(np.max(c.depth_m)) for c in casts)
        # A lot of SWellEx casts are on a 0.5 m grid.
        depth_grid_m = np.arange(z_min, z_max + 1e-9, 0.5, dtype=np.float64)

    Z = np.asarray(depth_grid_m, dtype=np.float64).reshape(-1)
    C = np.stack([resample_cast_to_grid(Z, c) for c in casts], axis=0)  # (n_casts, n_depth)

    stat = statistic.strip().lower()
    if stat == "median":
        c_rep = np.median(C, axis=0)
    elif stat == "mean":
        c_rep = np.mean(C, axis=0)
    else:
        raise KeyError(f"Unknown statistic: {stat!r}")

    return Z, c_rep


def estimate_depth_averaged_c0(
    casts: Sequence[CTDCast],
    *,
    water_depth_m: float,
    statistic: str = "median",
) -> float:
    """Estimate an effective constant sound speed c0 for baselines.

    We compute the per-cast mean sound speed over depths [0, water_depth_m] and
    then apply the requested statistic across casts (median by default).
    """
    if water_depth_m <= 0:
        raise ValueError("water_depth_m must be positive")

    per_cast_means: List[float] = []
    for c in casts:
        z = np.asarray(c.depth_m, dtype=np.float64)
        cc = np.asarray(c.sound_speed_mps, dtype=np.float64)
        m = z <= float(water_depth_m) + 1e-9
        if not np.any(m):
            continue
        per_cast_means.append(float(np.mean(cc[m])))

    if not per_cast_means:
        raise ValueError("No casts contain depth samples within the specified water depth")

    stat = statistic.strip().lower()
    arr = np.asarray(per_cast_means, dtype=np.float64)
    if stat == "median":
        return float(np.median(arr))
    if stat == "mean":
        return float(np.mean(arr))
    raise KeyError(f"Unknown statistic: {stat!r}")


def aggregate_sound_speed_profile(
    casts: Sequence[CTDCast],
    *,
    water_depth_m: float,
    dz_m: float = 1.0,
    statistic: str = "median",
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate multiple CTD casts into a representative c(z) profile.

    Parameters
    ----------
    casts:
        List of CTD casts.
    water_depth_m:
        Water depth used to truncate/extend the profile.
    dz_m:
        Uniform grid spacing (meters) for the returned profile.
    statistic:
        'median' (default) or 'mean' across casts at each depth.

    Returns
    -------
    z_grid_m:
        Uniform depth grid from 0 to water_depth_m.
    c_grid_mps:
        Aggregated sound speed profile on z_grid_m.
    """
    if dz_m <= 0:
        raise ValueError("dz_m must be positive.")

    H = float(water_depth_m)
    z_grid = np.arange(0.0, H + 0.5 * dz_m, float(dz_m), dtype=np.float64)
    if z_grid.size < 2:
        raise ValueError("water_depth_m too small for the requested dz_m.")

    # Interpolate each cast onto the common grid.
    C = []
    for c in casts:
        z = np.asarray(c.depth_m, dtype=np.float64).reshape(-1)
        cc = np.asarray(c.sound_speed_mps, dtype=np.float64).reshape(-1)
        if z.size < 2:
            continue
        # Ensure monotonic increasing depth for interpolation.
        order = np.argsort(z)
        z = z[order]
        cc = cc[order]
        # Clamp/extrapolate using end values.
        ci = np.interp(z_grid, z, cc, left=cc[0], right=cc[-1])
        C.append(ci)

    if not C:
        raise ValueError("No valid casts to aggregate.")

    C = np.stack(C, axis=0)  # (n_casts, n_depth)
    stat = statistic.strip().lower()
    if stat == "mean":
        c_grid = np.mean(C, axis=0)
    elif stat == "median":
        c_grid = np.median(C, axis=0)
    else:
        raise KeyError(f"Unknown statistic: {stat!r}")

    return z_grid.astype(np.float64), c_grid.astype(np.float64)
