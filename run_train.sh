#!/bin/bash

if [ "$1" != "--bg" ]; then
    LOG_FILE="run_pipeline.log"
    echo "🚀 Submitting task to the background..."

    nohup "$0" --bg > "$LOG_FILE" 2>&1 &

    echo "✅ Task successfully started in the background!"
    echo "📄 All output will be logged to: $LOG_FILE"
    echo "💡 Tip: You can safely close this terminal. To check progress, use: tail -f $LOG_FILE"
    exit 0
fi

set -e

echo "========== Starting SonarKAD resubmission pipeline =========="
date

echo ">>> 0) Validating dataset wiring..."
python scripts/run.py data-validate --config configs/config.yaml

echo ">>> 1) Preparing data..."
python scripts/run.py swellex96-prepare --exp swellex96_s5_vla
python scripts/run.py swellex96-prepare --exp swellex96_s59_vla

echo ">>> 2) Fitting event models for decomposition figures..."
python scripts/run.py swellex96-train --exp swellex96_s5_vla
python scripts/run.py swellex96-train --exp swellex96_s59_vla

echo ">>> 3) Blocked cross-validation for fixed-capacity references..."
python scripts/run.py swellex96-train-cv --exp swellex96_s5_vla
python scripts/run.py swellex96-train-cv --exp swellex96_s59_vla

echo ">>> 4) Automatic rank selection..."
python scripts/run.py swellex96-select-rank --exp swellex96_s5_vla --ranks 0,1,2,4,8,16 --selection-rule 1se
python scripts/run.py swellex96-select-rank --exp swellex96_s59_vla --ranks 0,1,2,4,8,16 --selection-rule 1se

echo ">>> 5) Paper assets (uses selected ranks)..."
python scripts/run.py paper-assets

echo ">>> 6) Optional cross-event transfer study..."
python scripts/run.py swellex96-transfer --source swellex96_s5_vla --target swellex96_s59_vla || true

echo "========== SonarKAD pipeline completed =========="
date
