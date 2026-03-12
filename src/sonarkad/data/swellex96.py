"""SWellEx-96 dataset helpers (public benchmark for shallow-water acoustics).

This module does *not* attempt to fully replicate the MATLAB toolchain used in the
SWellEx-96 website. Instead, it provides a pragmatic pipeline that supports the
paper's needs:

- Extract received level (RL) / tonal power vs time and frequency from the array
  ``*.sio`` files (after the user downloads & gunzips them).
- Align each time frame with source-to-array range ``r(t)`` (from the provided
  ``range.tar`` data, exported to a simple CSV/TSV).
- Build a dataset of samples (r, f) -> RL_dB suitable for SonarKAD training.

Design choices for reviewer-aligned experiments
-----------------------------------------------
- Use **tonal sets** listed in the SWellEx-96 event pages (e.g., T-49-13) so the
  recovered φ_f(f) can be validated against known transmitted tone frequencies.
- Include an optional **low-rank interaction ψ(r,f)** in the model to capture
  range–frequency striations seen in real shallow-water propagation.

Notes
-----
- Array sampling rate is 1500 Hz for the VLA/TLA tonal data commonly used in the
  literature; confirm with the event/array documentation for your files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import re
import numpy as np

from .sio import SIOReader


# ---------------------------------------------------------------------
# Canonical tone lists (from the SWellEx-96 event pages)
# ---------------------------------------------------------------------


T49_13_HIGH = [49, 64, 79, 94, 112, 130, 148, 166, 201, 235, 283, 338, 388]
T49_13_SET2 = [52, 67, 82, 97, 115, 133, 151, 169, 204, 238, 286, 341, 391]
T49_13_SET3 = [55, 70, 85, 100, 118, 136, 154, 172, 207, 241, 289, 344, 394]
T49_13_SET4 = [58, 73, 88, 103, 121, 139, 157, 175, 210, 244, 292, 347, 397]
T49_13_SET5 = [61, 76, 91, 106, 124, 142, 160, 178, 213, 247, 295, 350, 400]
T49_13_ALL = T49_13_HIGH + T49_13_SET2 + T49_13_SET3 + T49_13_SET4 + T49_13_SET5

C109_9S = [109, 127, 145, 163, 198, 232, 280, 335, 385]

NOISE_FREQS = [62, 77, 92, 107, 125, 143, 161, 179, 214, 248, 296, 351, 401]


# ---------------------------------------------------------------------
# Extraction configuration
# ---------------------------------------------------------------------


@dataclass
class TonalExtractionConfig:
    """Configuration for tonal received-level extraction from SWellEx-96 SIO files.

    Notes
    -----
    - Sampling rate is typically 1500 Hz for the VLA/TLA tonal datasets used in
      SWellEx-96 Events S5 and S59; confirm against the event documentation for
      your specific files.
    - We work in the log-power domain (dB). The returned values are intended as
      a robust *intensity* proxy (LOFARgram-style), not a coherent field.
    """

    fs_hz: float = 1500.0
    win_sec: float = 2.0
    hop_sec: float = 2.0
    n_fft: int = 4096
    window: str = 'hann'

    detrend: bool = True
    channel_pool: str = 'mean'  # 'mean' or 'median'

    subtract_noise: bool = False
    noise_freqs_hz: Optional[List[float]] = None
    noise_stat: str = 'median'  # 'median' or 'mean'




# ---------------------------------------------------------------------
# Array geometry helpers (VLA/TLA positions)
# ---------------------------------------------------------------------

def load_element_depths_m(path: str | Path) -> np.ndarray:
    """Load element depths (meters) from a SWellEx-96 positions_*.txt file.

    The public SWellEx-96 website provides ``positions_vla.txt`` and similar files.
    In the wild, these files appear either as:
      (i) one depth per line, or
      (ii) two columns: <index> <depth_m>.

    This loader is intentionally forgiving: it extracts numeric tokens and returns
    the last token on each data line as the depth.

    Returns
    -------
    depths_m : (N,) float array
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    depths: List[float] = []
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        # Skip separators / obvious headers
        if s.startswith("#") or s.startswith("%") or s.startswith("//"):
            continue
        if set(s) <= {"-", " "}:
            continue

        # Extract floats/ints from the line
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        if not nums:
            continue

        # If the line looks like "idx depth", take the last number as depth
        depth = float(nums[-1])
        depths.append(depth)

    if not depths:
        raise ValueError(f"No numeric depths found in positions file: {p}")

    return np.asarray(depths, dtype=np.float64)


def load_vla_depths_m(path: str | Path, *, reverse_for_sio: bool = True) -> np.ndarray:
    """Load VLA element depths with the correct channel ordering.

    Dataset note (important for reproducibility)
    -------------------------------------------
    The SWellEx-96 documentation states that for the VLA/TLA data files used in
    Events S5 and S59, the *channel order in the .sio files is reversed relative
    to the published element position tables*.

    Practically, this means that when you map a channel index to a physical depth
    using ``positions_vla.txt``, you must reverse the depth list to match the
    .sio channel ordering (e.g., channel 1 corresponds to ~94.125 m and channel 21
    corresponds to ~212.25 m for the 21-element VLA subset).

    Parameters
    ----------
    path:
        Path to ``positions_vla.txt``.
    reverse_for_sio:
        If True (recommended), reverse the loaded depths to match .sio channel
        ordering.

    Returns
    -------
    depths_m : (N,) float array
    """
    depths = load_element_depths_m(path)
    if reverse_for_sio:
        # Robust handling of the documented VLA/TLA channel-order reversal in the
        # S5/S59 SIO files: reverse the table only if it is in descending depth
        # order (deep-to-shallow). If the table is already shallow-to-deep, we
        # leave it unchanged to avoid double-reversing user-corrected files.
        if depths.size >= 2 and float(depths[0]) > float(depths[-1]):
            depths = depths[::-1].copy()
    return depths


def get_tone_frequencies(name: str) -> List[float]:
    """Return a list of tone frequencies by a symbolic name."""
    key = name.strip().lower()
    if key in ("t49_13", "t49-13", "t49_13_all", "t-49-13"):
        return [float(x) for x in T49_13_ALL]
    if key in ("t49_13_high", "t49-13-high", "high"):
        return [float(x) for x in T49_13_HIGH]
    if key in ("c109_9s", "c-109-9s", "shallow"):
        return [float(x) for x in C109_9S]
    raise KeyError(f"Unknown tone set name: {name!r}")


# ---------------------------------------------------------------------
# Range/track helpers
# ---------------------------------------------------------------------


def _parse_time_token(tok: str) -> Optional[float]:
    """Parse 'hh:mm' or 'hh:mm:ss' into seconds. Returns None if not time-like."""
    if ":" not in tok:
        return None
    parts = tok.split(":")
    if len(parts) not in (2, 3):
        return None
    try:
        hh = int(parts[0])
        mm = int(parts[1])
        ss = int(parts[2]) if len(parts) == 3 else 0
    except ValueError:
        return None
    return float(hh * 3600 + mm * 60 + ss)


def find_range_table_file(dir_path: str | Path, *, hint: Optional[str] = None) -> Path:
    """Find a likely range-table file inside a directory.

    The official SWellEx-96 ``range.tar`` archive can be extracted into a folder that
    contains multiple text files (ranges to different arrays, helper MATLAB scripts,
    etc.). Because users may extract this archive in different ways, we support
    passing either a direct file path or a directory path.

    This helper searches for a numeric-looking text file and picks the best match
    using a simple filename-based scoring heuristic.

    Parameters
    ----------
    dir_path:
        Directory that contains candidate range/track text files.
    hint:
        Optional substring to prefer (e.g., ``"S5"``, ``"S59"``, ``"VLA"``).

    Returns
    -------
    path:
        Path to the selected range table file.
    """
    p = Path(dir_path)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(p)

    hint_l = (hint or "").strip().lower()
    exts = {".txt", ".dat", ".csv", ".tsv", ".prn"}

    best: Optional[Path] = None
    best_score = -1e9

    for fp in sorted(p.rglob("*")):
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in exts:
            continue
        name = fp.name.lower()
        # Skip obvious non-data files
        if "readme" in name or "note" in name:
            continue
        if fp.stat().st_size < 32:
            continue

        score = 0.0
        # Prefer explicit per-event range tables (often named SproulTo*.S5/S59.txt)
        if "range" in name:
            score += 5.0
        if "sproulto" in name:
            score += 5.0
        if "tovla" in name or ("sproulto" in name and "vla" in name):
            score += 3.0
        if "tohla" in name or ("sproulto" in name and "hla" in name):
            score += 2.0
        if "totla" in name or ("sproulto" in name and "tla" in name):
            score += 2.0

        if "s5" in name:
            score += 1.0
        if "s59" in name:
            score += 1.0
        if "vla" in name:
            score += 1.0
        if hint_l and hint_l in name:
            score += 10.0

        # Prefer smaller tables if scores tie (typically the per-event range table)
        score += -1e-9 * float(fp.stat().st_size)

        if score > best_score:
            best_score = score
            best = fp

    if best is None:
        raise FileNotFoundError(f"No candidate range-table text files found under: {p}")

    return best


def _parse_sio_start_from_filename(sio_path: Union[str, Path]) -> Optional[Tuple[int, int]]:
    """Infer (Julian day, minutes-of-day) from a SWellEx-96 SIO filename.

    The public SWellEx-96 array recordings are commonly named like::

        J1312315.vla.21els.sio

    where the prefix encodes ``J{jday}{HHMM}``.

    Returns
    -------
    (jday, minute_of_day) if parsing succeeds; otherwise ``None``.
    """
    name = Path(sio_path).name
    m = re.match(r"^J(?P<jday>\d{3})(?P<hhmm>\d{4})", name, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        jday = int(m.group("jday"))
        hhmm = m.group("hhmm")
        hh = int(hhmm[:2])
        mm = int(hhmm[2:])
        return jday, hh * 60 + mm
    except Exception:
        return None


def _parse_first_time_from_native_range_table(range_table_path: Union[str, Path]) -> Optional[Tuple[int, int]]:
    """Parse first (Jday, HH:MM) row from a native SWellEx-96 range table."""
    p = Path(range_table_path)
    if not p.exists() or p.is_dir():
        return None
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 2:
        return None
    # Find first data row after the header.
    for ln in lines[1:]:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#") or s.startswith("%") or s.startswith("//"):
            continue
        parts = s.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            jday = int(parts[0])
        except Exception:
            continue
        tv = _parse_time_token(parts[1])
        if tv is None:
            continue
        minute_of_day = int(round(float(tv) / 60.0))
        return jday, minute_of_day
    return None


def infer_range_time_offset_sec(
    sio_path: Union[str, Path],
    range_table_path: Union[str, Path],
    *,
    file_hint: Optional[str] = None,
) -> float:
    """Infer a range-table alignment offset (seconds) from filename time stamps.

    In our pipeline, STFT frame times ``t_frames`` are measured from the beginning
    of the SIO file. Native SW96 range tables include a time-of-day column in
    addition to a ``Duration`` column. In some events (notably S59), the recording
    start time encoded in the SIO filename precedes the first range-table row by
    about one minute.

    If the first range-table row occurs ``delta`` seconds after the SIO start time,
    the correct alignment is

        r(t) = r_table(t - delta),

    which is implemented by adding a *negative* query offset:

        t_query = t_frames + range_time_offset_sec,
        range_time_offset_sec = -delta.

    This function estimates ``delta`` from (i) the SIO filename prefix
    ``J{jday}{HHMM}`` and (ii) the first row of the selected native range table.

    Returns
    -------
    range_time_offset_sec:
        Offset in seconds to add to ``t_frames`` before interpolating the range
        table. If parsing fails, returns 0.0.
    """
    sio_start = _parse_sio_start_from_filename(sio_path)
    if sio_start is None:
        return 0.0

    rp = Path(range_table_path)
    if rp.is_dir():
        rp = find_range_table_file(rp, hint=file_hint)

    first = _parse_first_time_from_native_range_table(rp)
    if first is None:
        return 0.0

    j_sio, m_sio = sio_start
    j_rng, m_rng = first
    delta_min = (j_rng - j_sio) * 1440 + (m_rng - m_sio)
    delta_sec = float(delta_min) * 60.0
    return float(-delta_sec)



def load_range_table(
    path: str | Path,
    *,
    range_col: int = 1,
    time_col: int = 0,
    range_scale_to_m: float = 1.0,
    time_scale_to_sec: float = 1.0,
    time_offset_sec: float = 0.0,
    file_hint: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a (time, range) table and return arrays (t_sec, r_m).

    This utility supports both the generic ``(t, r, ...)`` text tables you may
    create from the official MATLAB scripts, and the *native SWellEx-96 per-event*
    range tables such as ``SproulToVLA.S5.txt`` and ``SproulToVLA.S59.txt``.

    Native SWellEx-96 range-table format
    -----------------------------------
    The per-event tables shipped with the dataset have a header like::

        Jday Time  Duration Range(km)

    where ``Duration`` is in minutes from the event start and ``Range(km)`` is in km.
    For these files we automatically parse:
        t_sec = Duration * 60
        r_m   = Range(km) * 1000

    Generic formats
    ---------------
    1) Two or more columns (whitespace or comma separated): ``t  r  ...``
    2) Time token in the specified time column: ``hh:mm[:ss]  r  ...``

    Parameters
    ----------
    path:
        File path to a range table, or a directory containing range tables.
    range_col:
        Column index (0-based after tokenization) holding the range values for
        generic tables.
    time_col:
        Column index (0-based) holding the time values/tokens for generic tables.
    range_scale_to_m:
        Multiply parsed range values by this factor to convert to meters.
        For native ``Range(km)`` tables, if this is left at 1.0 we default to 1000.
    time_scale_to_sec:
        Multiply parsed time values by this factor to convert to seconds (e.g., 60 if
        a table stores minutes). For native ``Duration`` tables, if left at 1.0 we
        default to 60.
    time_offset_sec:
        Add this constant offset (seconds) to all parsed time stamps. This can be used
        to align range tables to the audio start time when the range table begins later
        than the recording.
    file_hint:
        If ``path`` is a directory, use this hint to pick the right file
        (e.g., ``'SproulToVLA'``).

    Returns
    -------
    t_sec, r_m:
        1D arrays of equal length, sorted by time.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.is_dir():
        p = find_range_table_file(p, hint=file_hint)

    raw_lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    # Find first non-empty, non-comment line (potential header)
    header = ""
    for ln in raw_lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#") or s.startswith("%") or s.startswith("//"):
            continue
        header = s
        break

    header_l = header.lower()

    # ----------------------------
    # Native SWellEx-96 range file
    # ----------------------------
    if ("jday" in header_l) and ("duration" in header_l) and ("range" in header_l):
        # Heuristic defaults consistent with the header tokens
        rs = float(range_scale_to_m)
        ts = float(time_scale_to_sec)
        if ("km" in header_l) and abs(rs - 1.0) < 1e-12:
            rs = 1000.0
        if abs(ts - 1.0) < 1e-12:
            ts = 60.0

        rows: List[Tuple[float, float]] = []
        for ln in raw_lines[1:]:
            s = ln.strip()
            if not s:
                continue
            parts = s.replace(",", " ").split()
            if len(parts) < 4:
                continue
            # Expect: Jday, Time, Duration, Range(km)
            try:
                dur = float(parts[2]) * ts + float(time_offset_sec)
                rng = float(parts[3]) * rs
            except ValueError:
                continue
            rows.append((dur, rng))

        if not rows:
            raise ValueError(f"Could not parse any (Duration, Range) rows from native SWellEx table: {p}")

        rows.sort(key=lambda x: x[0])
        t = np.asarray([r[0] for r in rows], dtype=np.float64)
        r = np.asarray([r[1] for r in rows], dtype=np.float64)
        return t, r

    # ----------------------------
    # Generic table parsing
    # ----------------------------
    rows_g: List[Tuple[float, float]] = []
    for line in raw_lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith("#") or s.startswith("%") or s.startswith("//"):
            continue

        parts = s.replace(",", " ").split()
        if len(parts) <= max(time_col, range_col):
            continue

        tok_t = parts[time_col]
        t_val: Optional[float] = _parse_time_token(tok_t)
        if t_val is None:
            try:
                t_val = float(tok_t)
            except ValueError:
                continue

        try:
            r_val = float(parts[range_col]) * float(range_scale_to_m)
        except ValueError:
            continue

        t_val = float(t_val) * float(time_scale_to_sec) + float(time_offset_sec)
        rows_g.append((t_val, r_val))

    if not rows_g:
        raise ValueError(
            f"Could not parse any (time, range) rows from {p}. "
            f"Try setting time_col/range_col and time_scale_to_sec/range_scale_to_m."
        )

    rows_g.sort(key=lambda x: x[0])
    t = np.asarray([rr[0] for rr in rows_g], dtype=np.float64)
    r = np.asarray([rr[1] for rr in rows_g], dtype=np.float64)
    return t, r


def _get_window(name: str, n: int) -> np.ndarray:
    key = name.strip().lower()
    if key in ("hann", "hanning"):
        return np.hanning(n)
    if key in ("rect", "boxcar", "none"):
        return np.ones(n)
    raise KeyError(f"Unknown window: {name!r}")


def extract_tonal_rl_db(
    reader: SIOReader,
    cfg: TonalExtractionConfig,
    *,
    channels: Sequence[int],
    tones_hz: Sequence[float],
    t_start_sec: float = 0.0,
    t_end_sec: Optional[float] = None,
    channels_are_one_indexed: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (pooled-channel) tonal received level in dB vs time.

    This is an intensity-domain (LOFAR-type) feature extractor: we compute a short-time
    FFT of each channel, form per-channel power spectra, pool across channels
    (mean/median), then sample the pooled spectrum at the requested tone frequencies.

    Parameters
    ----------
    reader:
        ``SIOReader`` over the uncompressed array file.
    cfg:
        Extraction configuration.
    channels:
        Channel list to use (e.g., 1..21 for the SWellEx VLA subset).
    tones_hz:
        Tone frequencies (Hz). Although the function name says "tonal", any list of
        frequencies can be passed (e.g., a dense band grid).
    t_start_sec, t_end_sec:
        Time interval to process in seconds relative to file start.
    channels_are_one_indexed:
        If True, channels are interpreted as 1..nc.

    Returns
    -------
    t_frames_sec:
        Frame center times (seconds).
    f_hz:
        Frequencies (Hz), as float array.
    rl_db:
        Array (n_frames, n_freq) with pooled-channel power in dB.
    """
    fs = float(cfg.fs_hz)
    win_n = int(round(cfg.win_sec * fs))
    hop_n = int(round(cfg.hop_sec * fs))
    n_fft = int(cfg.n_fft)

    if win_n <= 0 or hop_n <= 0:
        raise ValueError("win_sec/hop_sec too small.")
    if n_fft < win_n:
        # Zero-padding is allowed but FFT length must be >= window length.
        n_fft = 1
        while n_fft < win_n:
            n_fft *= 2

    tones = np.asarray(list(tones_hz), dtype=np.float64).reshape(-1)
    if tones.size == 0:
        raise ValueError("tones_hz is empty: please provide at least one frequency.")

    w = _get_window(cfg.window, win_n).astype(np.float32)

    # Determine total samples. For SWellEx this is stored in the header; our reader
    # also derives it from file size when needed.
    total_samples = int(reader.header.np_per_channel)
    if total_samples <= 0:
        raise ValueError(
            f"Could not determine np_per_channel from SIO header (np={total_samples}). "
            f"File: {getattr(reader, 'path', '<unknown>')}"
        )

    duration_sec = float(total_samples) / fs
    start_samp = int(round(float(t_start_sec) * fs))
    if t_end_sec is None:
        end_samp = total_samples
        t_end_eff = duration_sec
    else:
        end_samp = min(total_samples, int(round(float(t_end_sec) * fs)))
        t_end_eff = float(end_samp) / fs

    # Validate bounds with an informative error (this is a common failure mode when
    # a user points the config at the wrong .sio file).
    if start_samp < 0:
        raise ValueError(f"Invalid t_start_sec={t_start_sec}: must be >= 0.")
    if start_samp >= total_samples:
        raise ValueError(
            f"Invalid extraction window: t_start_sec={t_start_sec} s is beyond file duration "
            f"{duration_sec:.2f} s (np_per_channel={total_samples}, fs={fs})."
        )
    if end_samp <= start_samp:
        raise ValueError(
            f"Invalid extraction window: t_end_sec={t_end_sec} (effective {t_end_eff:.2f} s) "
            f"must be > t_start_sec={t_start_sec}."
        )
    if (end_samp - start_samp) < win_n:
        raise ValueError(
            f"Extraction interval too short for one frame: [{t_start_sec:.2f},{t_end_eff:.2f}] s "
            f"contains {end_samp-start_samp} samples but win_n={win_n} samples."
        )

    # Precompute FFT bin indices for requested freqs
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)  # (n_bins,)
    tone_bins = np.array([int(np.argmin(np.abs(freqs - f))) for f in tones], dtype=np.int64)

    noise_bins: Optional[np.ndarray] = None
    if cfg.subtract_noise:
        noise_list = cfg.noise_freqs_hz if cfg.noise_freqs_hz is not None else NOISE_FREQS
        noise = np.asarray(list(noise_list), dtype=np.float64).reshape(-1)
        noise_bins = np.array([int(np.argmin(np.abs(freqs - f))) for f in noise], dtype=np.int64)

    # Iterate frames
    starts = np.arange(start_samp, end_samp - win_n + 1, hop_n, dtype=np.int64)
    if starts.size == 0:
        raise ValueError(
            f"No frames to process: start_samp={start_samp}, end_samp={end_samp}, win_n={win_n}, hop_n={hop_n}. "
            f"Check (t_start_sec,t_end_sec) and fs_hz."
        )

    n_frames = int(starts.size)
    rl_db = np.empty((n_frames, tones.size), dtype=np.float32)
    t_frames = (starts + win_n / 2.0) / fs

    for i, s0 in enumerate(starts):
        x = reader.read_segment(
            int(s0),
            win_n,
            channels,
            channels_are_one_indexed=channels_are_one_indexed,
            as_float32=True,
        )  # (win_n, n_ch)

        if cfg.detrend:
            x = x - np.mean(x, axis=0, keepdims=True)

        xw = x * w[:, None]
        X = np.fft.rfft(xw, n=n_fft, axis=0)  # (n_bins, n_ch)
        P = (np.abs(X) ** 2).astype(np.float64)  # power

        # Pool channels at each freq bin
        if cfg.channel_pool.strip().lower() == "median":
            P_pool = np.median(P, axis=1)
        else:
            P_pool = np.mean(P, axis=1)

        # Tone power
        P_tones = P_pool[tone_bins]  # (n_tones,)

        if noise_bins is not None:
            P_noise = P_pool[noise_bins]
            if cfg.noise_stat.strip().lower() == "mean":
                noise_level = float(np.mean(P_noise))
            else:
                noise_level = float(np.median(P_noise))
            P_tones = np.maximum(P_tones - noise_level, 1e-30)

        rl_db[i, :] = (10.0 * np.log10(np.maximum(P_tones, 1e-30))).astype(np.float32)

    return t_frames.astype(np.float64), tones.astype(np.float64), rl_db
def save_processed_npz(
    out_path: str | Path,
    *,
    t_sec: np.ndarray,
    r_m: np.ndarray,
    f_hz: np.ndarray,
    rl_db: np.ndarray,
    meta: Optional[Dict[str, object]] = None,
) -> Path:
    """Save a processed tonal dataset to NPZ."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if meta is None:
        meta = {}

    np.savez(
        p,
        t_sec=np.asarray(t_sec, dtype=np.float64),
        r_m=np.asarray(r_m, dtype=np.float64),
        f_hz=np.asarray(f_hz, dtype=np.float64),
        rl_db=np.asarray(rl_db, dtype=np.float32),
        meta=np.array([meta], dtype=object),
    )
    return p
