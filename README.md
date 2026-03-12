# SonarKAD: physics-aligned kolmogorov–arnold decomposition

## Installation

You can install SonarKAD directly via pip:

```bash
pip install SonarKAD
```
## Dataset Download

The SWellEx-96 dataset used in this repository is publicly available and hosted by the Marine Physical Laboratory (MPL) at UC San Diego. You can download the required acoustic and environmental data from the links below:

* **Official SWellEx-96 Website:** [http://swellex96.ucsd.edu/](http://swellex96.ucsd.edu/)
* **UCSD Library Digital Collections:** [SWellEx-96 Experiment Acoustic Data](https://library.ucsd.edu/dc/collection/bb3312136z)

**Note on required files:** To run the standard workflow, please download the acoustic data for **Event S5** and **Event S59** (specifically the `.sio` VLA files and `positions_vla.txt`), along with the `ctds` environmental data. Place all downloaded files directly into the `data/` directory as specified in the layout below.

## Data layout

See `data/README_DATA.md`.

```text
data/
  J1312315.vla.21els.sio
  J1341145.vla.21els.sio
  positions_vla.txt
  RangeEventS5/
  RangeEventS59/
  ctds/
```

> **Important**: for S5 and S59, the `.sio` channel order is reversed relative to `positions_vla.txt`; the config handles this via `reverse_positions_for_sio: true`.

## Optimal workflow

From the repository root:

```bash
# 0) Validate dataset wiring
python scripts/run.py data-validate --config configs/config.yaml

# 1) Prepare tonal datasets
python scripts/run.py swellex96-prepare --exp swellex96_s5_vla
python scripts/run.py swellex96-prepare --exp swellex96_s59_vla

# 2) Train one model per event (used for decomposition figures)
python scripts/run.py swellex96-train --exp swellex96_s5_vla
python scripts/run.py swellex96-train --exp swellex96_s59_vla

# 3) Fixed-capacity blocked-CV reference results
python scripts/run.py swellex96-train-cv --exp swellex96_s5_vla
python scripts/run.py swellex96-train-cv --exp swellex96_s59_vla

# 4) Rank selection for BOTH events
python scripts/run.py swellex96-select-rank --exp swellex96_s5_vla --ranks 0,1,2,4,8,16 --selection-rule 1se
python scripts/run.py swellex96-select-rank --exp swellex96_s59_vla --ranks 0,1,2,4,8,16 --selection-rule 1se

# 5) Build paper assets 
python scripts/run.py paper-assets

# 6) Optional: cross-event transfer study 
python scripts/run.py swellex96-transfer --source swellex96_s5_vla --target swellex96_s59_vla
```

A convenience wrapper is available:

```bash
bash run_train.sh
bash run_plot.sh
```
