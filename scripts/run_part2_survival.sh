#!/bin/bash
# Part 2: Multimodal survival analysis (5-fold CV, C-index)
# Uses: sc_npz (pre-generated), bulk, surv_label, deg_dir from config.
# If fraction from Part 1 is missing, copies a repo-local precomputed celltype CSV when available.
#
# Usage: conda activate descent
#        export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
#        ./scripts/run_part2_survival.sh [BRCA]
set -e
DESCENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DESCENT_ROOT"

[[ -n "$CONDA_PREFIX" && -d "$CONDA_PREFIX/lib" ]] && export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Clear GPU memory after each Python step
gpu_cleanup() { python -c "import gc; import torch; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true; }

CANCER="${1:-BRCA}"
CONFIG_PATH="config/path_local.json"
echo "=== Part 2: Multimodal Survival (${CANCER}) ==="

# Copy pre-computed fraction for Part 2 if Part 1 output missing
FRAC_DIR="output/redeconv_fraction/${CANCER}"
FRAC_CSV="${FRAC_DIR}/${CANCER}_ratios_redeconv_10.csv"
REPO_FRAC_CSV="data/${CANCER}/redeconv/${CANCER}_celltypes.csv"
if [[ ! -f "$FRAC_CSV" && -f "$REPO_FRAC_CSV" ]]; then
  echo "Part 1 fraction not found. Copying repo-local ${REPO_FRAC_CSV}..."
  mkdir -p "$FRAC_DIR"
  cp "$REPO_FRAC_CSV" "$FRAC_CSV"
elif [[ ! -f "$FRAC_CSV" ]]; then
  echo "Part 1 fraction not found and no repo-local fallback exists at ${REPO_FRAC_CSV}."
fi

echo ""
echo "=== Survival_DIR="$(python - <<PY
import json
import os

root = os.path.abspath("${DESCENT_ROOT}")
config_path = os.path.join(root, "${CONFIG_PATH}")
with open(config_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)
entry = cfg.get("${CANCER}", {})
deg_dir = entry.get("deg_dir", "")
if deg_dir and not os.path.isabs(deg_dir):
    deg_dir = os.path.normpath(os.path.join(root, deg_dir))
print(deg_dir)
PY
)"

if [[ -z "$DEG_DIR" ]]; then
  echo "Missing deg_dir for ${CANCER} in ${CONFIG_PATH}."
  exit 1
fi
if [[ ! -d "$DEG_DIR" ]]; then
  echo "Per-fold DEG directory not found: ${DEG_DIR}"
  exit 1
fi
if ! ls "${DEG_DIR}"/degs_fold*.csv >/dev/null 2>&1; then
  echo "No per-fold DEG files found in ${DEG_DIR}"
  exit 1
fi
echo "Using per-fold DEG from ${DEG_DIR}"

python survival_prediction/scrna_bulk_sc_survival_cv.py \
  --cancer "${CANCER}" \
  --config "${CONFIG_PATH}" \
  --deg_dir "${DEG_DIR}" \
  --epochs 300 \
  --num_folds 5
gpu_cleanup
echo "  -> output/survival_cv/${CANCER}/"

echo ""
echo "=== Part 2 complete ==="
