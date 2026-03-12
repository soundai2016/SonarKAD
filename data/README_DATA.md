# Expected `data/` layout (SWellEx-96)

This repository expects you to place **raw SWellEx-96** data under `data/`.

> **Important**: The large array recordings are distributed as `*.sio.gz`.
> For this codebase, **gunzip** them to `*.sio` (random access is required).

Recommended structure (matching the SWellEx website downloads)

```text
data/
  ctds/
    README
    i9601.prn ... i9651.prn

  RangeEventS5/
    runS5.m
    SproulToVLA.S5.txt
    ...

  RangeEventS59/
    runS59.m
    SproulToVLA.S59.txt
    ...

  # Array recordings (uncompressed)
  J1312315.vla.21els.sio    # Event S5, VLA, 21-element processed subarray
  J1341145.vla.21els.sio    # Event S59, VLA, 21-element processed subarray

  # VLA element positions (tab-delimited ASCII)
  positions_vla.txt

  # (optional) other arrays / gps / etc.
  # J1312315.tla.22els.sio
  # J1341145.tla.22els.sio
  # gps/
```

## What each file/folder is used for

### `J1312315.vla.21els.sio` (Event S5)
- **What**: VLA recording for Event **S5**, 21-element processed subarray.
- **Why we use it**: S5 is a "clean" tow (no loud interferer), used as the primary validation event.
- **Sanity check**: the filename prefix `J1312315` encodes *Julian day 131, 23:15 (GMT)*.

### `J1341145.vla.21els.sio` (Event S59)
- **What**: VLA recording for Event **S59**, 21-element processed subarray.
- **Why we use it**: S59 includes a **loud interferer** and is used to stress-test robustness.
- **Sanity check**: the filename prefix `J1341145` encodes *Julian day 134, 11:45 (GMT)*.

### `RangeEventS5/` and `RangeEventS59/`
- These folders contain MATLAB helpers (`runS5.m`, `runS59.m`) and **range tables** produced from the official range-track archive.
- The pipeline consumes the per-event text file (e.g. `SproulToVLA.S5.txt`, `SproulToVLA.S59.txt`) to map STFT frame times to source-to-array range `r(t)`.
- **Alignment note (S59)**: in many distributions the first range-table row occurs about **1 minute** after the recording start time. The pipeline supports `range_time_offset_sec: auto` to infer and correct this using the SIO filename time stamp and the range-table first row.

### `positions_vla.txt`
- Tab-delimited element positions for the VLA.
- For Events **S5** and **S59**, the SWellEx documentation states that the **channel order in the VLA/TLA SIO files is reversed relative to the published element position tables**.
- This repo handles it via `dataset.reverse_positions_for_sio: true`, which reverses the depth list only if the table is in deep-to-shallow order.

### `ctds/`
- CTD casts in `i96{StationNumber}.prn` format.
- Used for (i) an effective constant sound speed `c0` in traditional baselines and (ii) a profile-based modal baseline.
- The CTD README lists station metadata in **local time (GMT-7)**.

## Quick reproducibility checklist

1. Download the two VLA files from the SWellEx website:
   - `J1312315.vla.21els.sio.gz` (S5)
   - `J1341145.vla.21els.sio.gz` (S59)
2. `gunzip` them to `*.sio` and place them in `data/`.
3. Extract `range.tar` (or the per-event range directories) into `data/RangeEventS5/` and `data/RangeEventS59/`.
4. Extract `ctds.tar.gz` into `data/ctds/`.
5. Run the built-in validator:

```bash
python scripts/run.py data-validate --config configs/config.yaml
```

If the validator succeeds, `swellex96-prepare` and `swellex96-train-cv` should run end-to-end.
