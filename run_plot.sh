#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${ROOT_DIR}/outputs/results"
OUT_DIR="${ROOT_DIR}/outputs/figures"

rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"

python "${ROOT_DIR}/scripts/plot.py" \
  --results-dir "${RESULTS_DIR}" \
  --out-dir "${OUT_DIR}"

printf '[OK] refreshed assets in %s\n' "${OUT_DIR}"