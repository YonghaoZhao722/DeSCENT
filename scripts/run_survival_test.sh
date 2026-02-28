#!/bin/bash
# Test survival CV with local BRCA data (all 5 folds, 2 epochs for quick test)
# Usage: conda activate descent  # or scdiff
# Uses bundled redeconv in scgep_generation/redeconv/ (do NOT pip install redeconv)
set -e
DESCENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DESCENT_ROOT"

echo "=== DeSCENT Survival Test (BRCA) ==="
echo "Config: config/path_local.json"
echo "Data: data/BRCA/bulk, data/BRCA/surv_label (all 5 folds)"
echo ""

# Use conda's libstdc++ if available (fixes CXXABI_1.3.15 on older systems)
[[ -n "$CONDA_PREFIX" && -d "$CONDA_PREFIX/lib" ]] && export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python survival_prediction/scrna_bulk_sc_survival_cv.py \
  --cancer BRCA \
  --config config/path_local.json \
  --epochs 300 \
  --num_folds 5

echo ""
echo "=== Test complete. Check output/survival_cv/BRCA/ ==="
