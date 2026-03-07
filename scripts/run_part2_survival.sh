#!/bin/bash
# Part 2: Multimodal survival analysis (5-fold CV, C-index)
# Uses: sc_npz (pre-generated), bulk, surv_label, deg from config.
# If fraction from Part 1 not present, copies pre-computed BRCA_celltypes.tsv for reference.
#
# Usage: conda activate descent
#        export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
#        ./scripts/run_part2_survival.sh [BRCA]
set -e
DESCENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DESCENT_ROOT"

[[ -n "$CONDA_PREFIX" && -d "$CONDA_PREFIX/lib" ]] && export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Use GPU 1 (override with CUDA_VISIBLE_DEVICES if already set)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# Clear GPU memory after each Python step
gpu_cleanup() { python -c "import gc; import torch; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true; }

CANCER="${1:-BRCA}"
echo "=== Part 2: Multimodal Survival (${CANCER}) ==="

# Copy pre-computed fraction for Part 2 if Part 1 output missing
FRAC_DIR="output/redeconv_fraction/${CANCER}"
FRAC_CSV="${FRAC_DIR}/${CANCER}_ratios_redeconv_10.csv"
if [[ ! -f "$FRAC_CSV" && "$CANCER" == "BRCA" ]]; then
  echo "Part 1 fraction not found. Copying pre-computed BRCA_celltypes.tsv..."
  mkdir -p "$FRAC_DIR"
  # ReDeconv outputs TSV; convert to CSV format for downstream
  python -c "
import pandas as pd
df = pd.read_csv('/data/zhaoyh/ReDeconv/BRCA_celltypes.tsv', sep='\t')
if df.columns[0] != 'sample/cell_type':
    df = df.rename(columns={df.columns[0]: 'sample/cell_type'})
df.to_csv('$FRAC_CSV', index=False)
print('Copied to', '$FRAC_CSV')
"
fi

echo ""
echo "=== Survival CV ==="
DEG_DIR="output/deg_cv/${CANCER}"
DEG_ARGS=""
if [[ -d "$DEG_DIR" ]]; then
  DEG_ARGS="--deg_dir $DEG_DIR"
  echo "Using per-fold DEG from $DEG_DIR"
else
  echo "deg_dir not found; using global deg from config"
fi
python survival_prediction/scrna_bulk_sc_survival_cv.py \
  --cancer "${CANCER}" \
  --config config/path_local.json \
  $DEG_ARGS \
  --epochs 300 \
  --num_folds 5
gpu_cleanup
echo "  -> output/survival_cv/${CANCER}/"

echo ""
echo "=== Part 2 complete ==="
