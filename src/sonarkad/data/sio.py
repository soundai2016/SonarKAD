"""Minimal SIO (.sio) reader for SWellEx-96 style files.

Why this exists
---------------
The public SWellEx-96 experiment distributes raw array recordings as ``*.sio`` (often
compressed as ``*.sio.gz``). The community MATLAB utility ``sioread.m`` reads these
record-interleaved files. For reproducible experiments we need a small, dependency-
light Python reader that supports *random access* (extract time windows without
loading the full file into RAM).

SIO record layout (SWellEx-96)
------------------------------
The file is organized into fixed-length records of ``rl`` bytes:

- Record 0: header record (length = ``rl`` bytes).
- Records 1..: data records, interleaved by channel.
  For ``nc`` channels, record ordering is:

    record 1      -> channel 1, block 0
    record 2      -> channel 2, block 0
    ...
    record nc     -> channel nc, block 0
    record nc+1   -> channel 1, block 1
    ...

Each data record contains ``ptrec = rl/sl`` samples for one channel, where ``sl`` is
the sample length in bytes (2 -> int16, 4 -> float32).

Endianness note
---------------
Many SWellEx-96 SIO files are big-endian. The header contains a byte-swapping marker
``bs`` that should equal 32677 when interpreted with the correct endianness. We
therefore try both big- and little-endian interpretations and pick the most
plausible one using multiple consistency checks (marker + record-size + file-size
consistency).

This robustness also helps catch user mistakes (e.g., accidentally pointing the
pipeline at a tiny synthetic test file while the config expects the full 21-channel
VLA file).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

BS_MARKER = 32677  # SWellEx-96 / sioread.m byte-swapping marker


@dataclass(frozen=True)
class SIOHeader:
    """Parsed SIO header with additional sanity-check metadata."""

    # Core header fields (as used by the reader)
    ide: int
    nr: int  # number of records (used; may be corrected from file size)
    rl: int  # record length (bytes)
    nc: int  # number of channels
    sl: int  # sample length (bytes): 2=int16, 4=float32
    f0: int  # 0=int, 1=real (kept for completeness)
    np_per_channel: int  # points per channel (used; may be corrected from file size)
    bs: int  # byte-swap marker (should be 32677)
    filename: str
    comment: str
    endian: str  # '>' big-endian, '<' little-endian

    # Extra bookkeeping for debugging / reproducibility
    file_size_bytes: int
    nr_header: int
    np_per_channel_header: int
    nr_derived: int
    np_per_channel_derived: int

    @property
    def points_per_record(self) -> int:
        return int(self.rl // self.sl)

    @property
    def records_per_channel(self) -> int:
        # Includes the header record in nr, consistent with sioread.m conventions.
        return int(np.ceil(self.nr / max(1, self.nc)))


def _read_u32(f, endian: str) -> int:
    dt = np.dtype(endian + "u4")
    v = np.fromfile(f, dt, 1)
    if v.size != 1:
        raise EOFError("Unexpected EOF while reading SIO header.")
    return int(v[0])


def _read_header_candidate(p: Path, endian: str) -> Tuple[int, int, int, int, int, int, int, int, str, str]:
    with p.open("rb") as f:
        f.seek(0)
        ide = _read_u32(f, endian)
        nr = _read_u32(f, endian)
        rl = _read_u32(f, endian)
        nc = _read_u32(f, endian)
        sl = _read_u32(f, endian)
        f0 = _read_u32(f, endian)
        np_per_channel = _read_u32(f, endian)
        bs = _read_u32(f, endian)
        filename = f.read(24).split(b"\x00", 1)[0].decode("ascii", errors="ignore").strip()
        comment = f.read(72).split(b"\x00", 1)[0].decode("ascii", errors="ignore").strip()
    return ide, nr, rl, nc, sl, f0, np_per_channel, bs, filename, comment


def _derive_from_file_size(file_size: int, *, rl: int, nc: int, sl: int) -> Tuple[int, int]:
    """Derive (nr, np_per_channel) from file size if possible.

    Returns
    -------
    nr_derived:
        Total number of records including the header record.
    np_per_channel_derived:
        Points per channel implied by the number of data records.
    """
    if rl <= 0 or nc <= 0 or sl <= 0 or (rl % sl) != 0:
        return 0, 0
    total_records = file_size // rl
    if total_records < 1:
        return 0, 0
    # Record 0 is header, remaining are data records.
    data_records = max(0, total_records - 1)
    ptrec = rl // sl
    # Full channel blocks only (should divide evenly for SWellEx distribution).
    n_blocks = data_records // nc
    np_per_channel = int(n_blocks * ptrec)
    return int(total_records), int(np_per_channel)


def _plausibility_score(p: Path, cand: dict) -> Tuple[int, int, int]:
    """Score a candidate header interpretation.

    Returns
    -------
    score, nr_derived, np_derived
    """
    file_size = int(p.stat().st_size)
    rl = int(cand["rl"])
    nc = int(cand["nc"])
    sl = int(cand["sl"])
    bs = int(cand["bs"])
    np_hdr = int(cand["np_per_channel"])
    nr_hdr = int(cand["nr"])

    nr_der, np_der = _derive_from_file_size(file_size, rl=rl, nc=nc, sl=sl)

    score = 0

    # Marker is the strongest signal.
    if bs == BS_MARKER:
        score += 100

    # Typical SWellEx values.
    if sl in (2, 4):
        score += 10
    if rl in (512, 1024, 2048, 4096, 8192, 16384):
        score += 10
    if rl > 0 and sl > 0 and (rl % sl) == 0:
        score += 5

    # Channel count plausibility.
    if 1 <= nc <= 512:
        score += 5

    # File-size consistency checks.
    if nr_der > 0:
        score += 3
        if (file_size % max(1, rl)) == 0:
            score += 2
        # data records should be multiples of nc.
        if nc > 0 and (max(0, nr_der - 1) % nc) == 0:
            score += 5

    # Consistency between header fields and derived values.
    if nr_der > 0 and nr_hdr > 0:
        if abs(nr_hdr - nr_der) <= 2:  # allow small off-by-one conventions
            score += 5
    if np_der > 0 and np_hdr > 0:
        rel_err = abs(np_hdr - np_der) / max(1.0, float(np_der))
        if rel_err < 1e-3:
            score += 10
        elif rel_err < 1e-2:
            score += 5

    # Filename heuristics help catch common mis-pointing errors.
    name = p.name.lower()
    if "21els" in name and nc == 21:
        score += 15
    if "vla" in name and nc >= 16:
        score += 2

    return int(score), int(nr_der), int(np_der)


def read_sio_header(path: str | Path) -> SIOHeader:
    """Read and sanity-check an SIO header.

    The function tries both big- and little-endian interpretations and selects the
    most plausible one using the byte-swap marker and file-size consistency.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    file_size = int(p.stat().st_size)

    candidates = []
    for endian in (">", "<"):
        try:
            ide, nr, rl, nc, sl, f0, np_per_channel, bs, filename, comment = _read_header_candidate(p, endian)
        except Exception:
            continue

        cand = dict(
            ide=int(ide),
            nr=int(nr),
            rl=int(rl),
            nc=int(nc),
            sl=int(sl),
            f0=int(f0),
            np_per_channel=int(np_per_channel),
            bs=int(bs),
            filename=str(filename),
            comment=str(comment),
            endian=endian,
        )
        score, nr_der, np_der = _plausibility_score(p, cand)
        cand["_score"] = int(score)
        cand["_nr_derived"] = int(nr_der)
        cand["_np_derived"] = int(np_der)
        candidates.append(cand)

    if not candidates:
        raise ValueError(f"Failed to parse SIO header: {p}")

    # Pick best candidate
    candidates.sort(key=lambda d: d["_score"], reverse=True)
    best = candidates[0]

    # Require at least *some* plausibility.
    if best["_score"] < 20:
        # Provide debugging info for both candidates
        dbg = ", ".join([f"{c['endian']}:score={c['_score']} bs={c['bs']} rl={c['rl']} nc={c['nc']} sl={c['sl']} np={c['np_per_channel']}" for c in candidates])
        raise ValueError(f"Unrecognized/implausible SIO header for {p}. Candidates: {dbg}")

    # Decide what values to *use* for length. Prefer derived if header looks inconsistent.
    nr_header = int(best["nr"])
    np_header = int(best["np_per_channel"])
    nr_der = int(best["_nr_derived"])
    np_der = int(best["_np_derived"])

    nr_used = nr_header
    if nr_der > 0 and (nr_header <= 0 or abs(nr_header - nr_der) > 4):
        nr_used = nr_der

    np_used = np_header
    if np_der > 0 and (np_header <= 0 or abs(np_header - np_der) > (best["rl"] // max(1, best["sl"]))):
        np_used = np_der

    return SIOHeader(
        ide=int(best["ide"]),
        nr=int(nr_used),
        rl=int(best["rl"]),
        nc=int(best["nc"]),
        sl=int(best["sl"]),
        f0=int(best["f0"]),
        np_per_channel=int(np_used),
        bs=int(best["bs"]),
        filename=str(best["filename"]),
        comment=str(best["comment"]),
        endian=str(best["endian"]),
        file_size_bytes=int(file_size),
        nr_header=int(nr_header),
        np_per_channel_header=int(np_header),
        nr_derived=int(nr_der),
        np_per_channel_derived=int(np_der),
    )


class SIOReader:
    """Random-access reader for SIO files."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.header = read_sio_header(self.path)

        if self.header.sl == 2:
            self._dtype = np.dtype(self.header.endian + "i2")  # int16
        elif self.header.sl == 4:
            self._dtype = np.dtype(self.header.endian + "f4")  # float32
        else:
            raise ValueError(f"Unsupported sample length sl={self.header.sl} in {self.path}")

    @property
    def fs_hz(self) -> float | None:
        """Sampling rate is not reliably encoded in SWellEx SIO headers; use config."""
        return None

    def _normalize_channels(self, channels: Sequence[int], *, channels_are_one_indexed: bool) -> List[int]:
        ch = [int(c) for c in channels]
        if not ch:
            raise ValueError("channels must be a non-empty sequence")
        if channels_are_one_indexed:
            ch0 = [c - 1 for c in ch]
        else:
            ch0 = ch
        if min(ch0) < 0 or max(ch0) >= self.header.nc:
            raise ValueError(f"channels out of range. nc={self.header.nc}, requested={channels}")
        return ch0

    def _read_segment_all_channels(
        self,
        start_sample: int,
        num_samples: int,
        *,
        as_float32: bool,
    ) -> np.ndarray:
        """Fast path: read a segment for *all* channels with a single seek+read."""
        ptrec = self.header.points_per_record
        r1 = start_sample // ptrec
        r2 = (start_sample + num_samples - 1) // ptrec
        n_blocks = r2 - r1 + 1
        offset_in_first = start_sample % ptrec

        # Read n_blocks * nc records contiguously.
        # Record 0 is the header; data starts at record 1.
        rec_no0 = 1 + r1 * self.header.nc
        n_values = n_blocks * self.header.nc * ptrec

        with self.path.open("rb") as f:
            f.seek(rec_no0 * self.header.rl, 0)
            raw = np.fromfile(f, self._dtype, n_values)

        if raw.size != n_values:
            raise EOFError(
                f"Unexpected EOF when reading {self.path}. Requested {n_values} values, got {raw.size}."
            )

        # raw ordering: (block, channel, ptrec)
        blk = raw.reshape(n_blocks, self.header.nc, ptrec)
        # Convert to (time, channel)
        x_tc = blk.transpose(0, 2, 1).reshape(n_blocks * ptrec, self.header.nc)
        seg = x_tc[offset_in_first : offset_in_first + num_samples, :]
        if as_float32 and seg.dtype != np.float32:
            seg = seg.astype(np.float32, copy=False)
        return seg

    def read_segment(
        self,
        start_sample: int,
        num_samples: int,
        channels: Sequence[int],
        *,
        channels_are_one_indexed: bool = True,
        as_float32: bool = True,
        prefer_fast: bool = True,
    ) -> np.ndarray:
        """Read a time segment.

        Parameters
        ----------
        start_sample:
            0-based start index within each channel.
        num_samples:
            Number of samples to read.
        channels:
            Channel indices to read. By default they are interpreted as 1-based
            (matching MATLAB/website conventions).
        channels_are_one_indexed:
            If True, channels are interpreted as 1..nc. If False, 0..nc-1.
        as_float32:
            If True, return float32 even when underlying data is int16.
        prefer_fast:
            If True (default), use a fast single-seek read path when requesting
            many channels (e.g., the full VLA subarray).

        Returns
        -------
        x:
            Array of shape (num_samples, len(channels)).
        """
        start_sample = int(start_sample)
        num_samples = int(num_samples)
        if start_sample < 0:
            raise ValueError("start_sample must be >= 0")
        if num_samples <= 0:
            raise ValueError("num_samples must be > 0")

        n_tot = int(self.header.np_per_channel) if int(self.header.np_per_channel) > 0 else None
        if n_tot is not None and (start_sample + num_samples > n_tot):
            raise ValueError(
                f"Requested segment exceeds file length: start={start_sample} num={num_samples} "
                f"np_per_channel={n_tot} (file={self.path})"
            )

        ch0 = self._normalize_channels(channels, channels_are_one_indexed=channels_are_one_indexed)

        # Heuristic: if we request at least half of the channels, read all channels
        # in one shot and then slice. This avoids O(n_channels * n_records) seeks.
        if prefer_fast and len(ch0) >= max(1, self.header.nc // 2):
            seg_all = self._read_segment_all_channels(start_sample, num_samples, as_float32=as_float32)
            return seg_all[:, ch0]

        # Fallback: per-channel random access (good for a small subset of channels)
        ptrec = self.header.points_per_record
        r1 = start_sample // ptrec
        r2 = (start_sample + num_samples - 1) // ptrec
        n_records = r2 - r1 + 1
        offset_in_first = start_sample % ptrec

        out = np.empty((num_samples, len(ch0)), dtype=np.float32 if as_float32 else self._dtype)
        with self.path.open("rb") as f:
            for j, c in enumerate(ch0):
                buf = np.empty(n_records * ptrec, dtype=self._dtype)
                for rr in range(n_records):
                    rec_no = 1 + c + (r1 + rr) * self.header.nc
                    f.seek(rec_no * self.header.rl, 0)
                    blk = np.fromfile(f, self._dtype, ptrec)
                    if blk.size != ptrec:
                        raise EOFError(f"Unexpected EOF while reading record {rec_no} from {self.path}")
                    buf[rr * ptrec : (rr + 1) * ptrec] = blk

                seg = buf[offset_in_first : offset_in_first + num_samples]
                if as_float32:
                    out[:, j] = seg.astype(np.float32, copy=False)
                else:
                    out[:, j] = seg

        return out
