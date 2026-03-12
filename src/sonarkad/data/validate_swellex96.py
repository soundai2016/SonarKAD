"""SWellEx-96 dataset validation helpers.

These utilities are meant to prevent the most common "silent" dataset mistakes:
- pointing the config at the wrong *.sio file (wrong event, wrong array, wrong channel count),
- forgetting that VLA/TLA channels are reversed for Events S5 and S59,
- using a range table that starts later than the audio file without compensating,
- missing CTD casts.

The validator is intentionally conservative: it fails fast on critical errors
(e.g., missing files, channel-count mismatch), and emits warnings for
non-fatal issues.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .sio import read_sio_header
from .swellex96 import (
    find_range_table_file,
    infer_range_time_offset_sec,
    load_element_depths_m,
    _parse_sio_start_from_filename,
    _parse_first_time_from_native_range_table,
)


# Canonical event start times used by this repo (from the official event pages).
# Format: (Julian day, HHMM)
_CANONICAL_EVENT_START = {
    "S5": (131, 2315),
    "S59": (134, 1145),
}


@dataclass
class ValidationReport:
    ok: bool
    errors: Dict[str, str]
    warnings: Dict[str, str]
    info: Dict[str, object]


def _hhmm_to_min_of_day(hhmm: int) -> int:
    hh = int(hhmm) // 100
    mm = int(hhmm) % 100
    return int(hh * 60 + mm)


def validate_positions_vla(path: str | Path) -> Tuple[bool, Dict[str, object], Dict[str, str]]:
    """Validate `positions_vla.txt`.

    Returns
    -------
    ok, info, warnings
    """
    p = Path(path)
    if not p.exists():
        return False, {}, {"positions": f"missing positions file: {p}"}

    depths = load_element_depths_m(p)
    info: Dict[str, object] = {
        "path": str(p),
        "n_rows": int(depths.size),
        "depth_min_m": float(np.min(depths)) if depths.size else None,
        "depth_max_m": float(np.max(depths)) if depths.size else None,
        "order": "deep-to-shallow" if (depths.size >= 2 and depths[0] > depths[-1]) else "shallow-to-deep",
    }

    warnings: Dict[str, str] = {}
    # The official VLA processed subset has 21 elements.
    if depths.size != 21:
        warnings["positions_rows"] = (
            f"positions_vla.txt has {depths.size} rows; expected 21 for the processed VLA subset. "
            "(This may be OK if you intentionally changed the array subset.)"
        )

    # Depth sanity (meters)
    if depths.size and (np.min(depths) < 0 or np.max(depths) > 400):
        warnings["positions_depth_range"] = (
            f"positions_vla.txt depth values look unusual: min={np.min(depths):.3f} m max={np.max(depths):.3f} m"
        )

    return True, info, warnings


def validate_sio(
    sio_path: str | Path,
    *,
    expected_nc: Optional[int] = None,
    event: Optional[str] = None,
) -> Tuple[bool, Dict[str, object], Dict[str, str], Dict[str, str]]:
    """Validate a SWellEx-96 *.sio file by parsing the header and filename.

    Returns
    -------
    ok, info, warnings, errors
    """
    p = Path(sio_path)
    if not p.exists():
        return False, {}, {}, {"sio": f"missing SIO file: {p}"}

    hdr = read_sio_header(p)

    info: Dict[str, object] = {
        "path": str(p),
        "nc": int(hdr.nc),
        "rl": int(hdr.rl),
        "sl": int(hdr.sl),
        "np_per_channel": int(hdr.np_per_channel),
        "endian": str(hdr.endian),
        "filename_in_header": hdr.filename,
    }

    warnings: Dict[str, str] = {}
    errors: Dict[str, str] = {}

    if expected_nc is not None and int(hdr.nc) != int(expected_nc):
        errors["sio_nc"] = (
            f"SIO channel count mismatch: header.nc={hdr.nc} expected_nc={expected_nc}. "
            "This usually means you pointed to the wrong .sio file (wrong array or wrong event)."
        )

    # Filename-based start-time validation
    start = _parse_sio_start_from_filename(p)
    if start is not None:
        jday, minute_of_day = start
        info["start_jday"] = int(jday)
        info["start_minute_of_day"] = int(minute_of_day)
        info["start_hhmm"] = int((minute_of_day // 60) * 100 + (minute_of_day % 60))

        if event is not None:
            ev = str(event).strip().upper()
            if ev in _CANONICAL_EVENT_START:
                j0, hhmm0 = _CANONICAL_EVENT_START[ev]
                if int(jday) != int(j0) or int(minute_of_day) != int(_hhmm_to_min_of_day(hhmm0)):
                    warnings["sio_start_time"] = (
                        f"SIO filename encodes J{jday} {minute_of_day//60:02d}:{minute_of_day%60:02d} GMT, "
                        f"but canonical start for {ev} is J{j0} {hhmm0//100:02d}:{hhmm0%100:02d} GMT. "
                        "If you intentionally trimmed/shifted the recording this may be OK; otherwise check the filename."
                    )
    else:
        warnings["sio_name"] = "Could not parse start time from SIO filename prefix 'J{jday}{HHMM}'."

    return (len(errors) == 0), info, warnings, errors


def validate_range_table(
    range_path: str | Path,
    *,
    hint: Optional[str] = None,
) -> Tuple[bool, Dict[str, object], Dict[str, str], Dict[str, str]]:
    """Validate the extracted range table (file or directory)."""
    p = Path(range_path)
    if not p.exists():
        return False, {}, {}, {"range": f"missing range path: {p}"}

    if p.is_dir():
        try:
            fp = find_range_table_file(p, hint=hint)
        except Exception as e:
            return False, {}, {}, {"range_find": f"could not find a range table under {p}: {e}"}
    else:
        fp = p

    info: Dict[str, object] = {"path": str(fp), "hint": hint}
    warnings: Dict[str, str] = {}
    errors: Dict[str, str] = {}

    first = _parse_first_time_from_native_range_table(fp)
    if first is not None:
        jday, minute_of_day = first
        info["first_row_jday"] = int(jday)
        info["first_row_minute_of_day"] = int(minute_of_day)
    else:
        warnings["range_first_row"] = (
            "Could not parse (Jday, Time) from the first row. "
            "If you use a converted generic (t,r) table this may be fine."
        )

    # Quick size sanity
    try:
        sz = int(fp.stat().st_size)
        info["size_bytes"] = sz
        if sz < 128:
            warnings["range_size"] = f"range table looks very small ({sz} bytes): {fp}"
    except Exception:
        pass

    return (len(errors) == 0), info, warnings, errors


def validate_swellex96_dataset(
    *,
    sio_path: str | Path,
    range_table_path: str | Path,
    positions_vla_path: Optional[str | Path] = None,
    expected_nc: Optional[int] = 21,
    event: Optional[str] = None,
    range_file_hint: Optional[str] = None,
) -> ValidationReport:
    """Validate the three key inputs for our SWellEx-96 experiments."""

    errors: Dict[str, str] = {}
    warnings: Dict[str, str] = {}
    info: Dict[str, object] = {}

    ok_sio, sio_info, sio_warn, sio_err = validate_sio(sio_path, expected_nc=expected_nc, event=event)
    info["sio"] = sio_info
    warnings.update({f"sio::{k}": v for k, v in sio_warn.items()})
    errors.update({f"sio::{k}": v for k, v in sio_err.items()})

    ok_rng, rng_info, rng_warn, rng_err = validate_range_table(range_table_path, hint=range_file_hint)
    info["range_table"] = rng_info
    warnings.update({f"range::{k}": v for k, v in rng_warn.items()})
    errors.update({f"range::{k}": v for k, v in rng_err.items()})

    if positions_vla_path is not None:
        ok_pos, pos_info, pos_warn = validate_positions_vla(positions_vla_path)
        info["positions_vla"] = pos_info
        warnings.update({f"positions::{k}": v for k, v in pos_warn.items()})
        if not ok_pos:
            errors["positions"] = f"missing or invalid positions_vla file: {positions_vla_path}"

    # If we have both (SIO start time) and (range table first time), estimate offset.
    try:
        off = infer_range_time_offset_sec(sio_path, range_table_path, file_hint=range_file_hint)
        info["inferred_range_time_offset_sec"] = float(off)
        # Warn if offset seems surprisingly large.
        if abs(float(off)) > 5 * 60:
            warnings["range_offset_large"] = (
                f"Inferred range_time_offset_sec={off:.1f} s. "
                "This is larger than a few minutes; double-check that you selected the correct range table."
            )
    except Exception as e:
        warnings["range_offset_infer"] = f"Could not infer range-time offset: {e}"

    ok = (len(errors) == 0) and ok_sio and ok_rng
    return ValidationReport(ok=ok, errors=errors, warnings=warnings, info=info)
