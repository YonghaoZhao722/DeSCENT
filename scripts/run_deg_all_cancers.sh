#!/bin/bash
# Run 5-fold DEG (tumor vs normal) for all cancers with tcga_umi data.
# Uses: /data/youzy/tcga_umi/{cancer}.csv, surv_label from config.
#
# Usage: conda activate descent
#        ./scripts/run_deg_all_cancers.sh              # all cancers
#        ./scripts/run_deg_all_cancers.sh COAD LUAD    # specific cancers
#        ./scripts/run_deg_all_cancers.sh --skip BRCA   # all except BRCA
set -e
DESCENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DESCENT_ROOT"

[[ -n "$CONDA_PREFIX" && -d "$CONDA_PREFIX/lib" ]] && export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

TCGA_UMI_DIR="${TCGA_UMI_DIR:-/data/youzy/tcga_umi}"
CONFIG="${CONFIG:-config/path.json}"
GENE_LIST="${GENE_LIST:-}"

# Cancers with tcga_umi data (lowercase filenames)
ALL_CANCERS=(BRCA COAD HNSC KIRC LGG LIHC LUAD STAD)

# Parse args: --skip X Y or cancer names
SKIP=()
CANCERS=()
for arg in "$@"; do
  if [[ "$arg" == "--skip" ]]; then
    SKIP_MODE=1
  elif [[ -n "$SKIP_MODE" ]]; then
    SKIP+=("$arg")
    SKIP_MODE=""
  else
    CANCERS+=("$arg")
  fi
done

if [[ ${#CANCERS[@]} -eq 0 ]]; then
  CANCERS=("${ALL_CANCERS[@]}")
fi

# Filter out skipped
for s in "${SKIP[@]}"; do
  CANCERS=("${CANCERS[@]/$s}")
done
CANCERS=($(printf '%s\n' "${CANCERS[@]}" | grep -v '^$'))

# tcga_umi has ENSG row index; must use gene list with ENSG (17948genes_with_symbol.csv)
# Config gene_list (e.g. BRCA_gene_order) often has symbol only -> 0 matches
if [[ -z "$GENE_LIST" || ! -f "$GENE_LIST" ]]; then
  for p in "$DESCENT_ROOT/data/BRCA/refs/17948genes_with_symbol.csv" \
           "/data/zhaoyh/scDiffusion-main/17948genes_with_symbol.csv" \
           "data/BRCA/refs/17948genes_with_symbol.csv"; do
    if [[ -f "$p" ]]; then
      GENE_LIST="$p"
      break
    fi
  done
fi

echo "=== 5-fold DEG (tumor vs normal) ==="
echo "  tcga_umi_dir: $TCGA_UMI_DIR"
echo "  config: $CONFIG"
echo "  gene_list: ${GENE_LIST:-none}"
echo "  cancers: ${CANCERS[*]}"
echo ""

for CANCER in "${CANCERS[@]}"; do
  UMI_FILE="$TCGA_UMI_DIR/$(echo "$CANCER" | tr '[:upper:]' '[:lower:]').csv"
  if [[ ! -f "$UMI_FILE" ]]; then
    echo "  [SKIP] $CANCER: $UMI_FILE not found"
    continue
  fi

  SURV=$(python3 -c "
import json
c = json.load(open('$CONFIG'))
print(c.get('$CANCER',{}).get('surv_label',''))
" 2>/dev/null || true)

  if [[ -z "$SURV" || ! -d "$SURV" ]]; then
    echo "  [SKIP] $CANCER: surv_label not found in config"
    continue
  fi

  echo ""
  echo "=== $CANCER ==="
  GENE_ARG=""
  [[ -n "$GENE_LIST" && -f "$GENE_LIST" ]] && GENE_ARG="--gene_list_path $GENE_LIST"

  python survival_prediction/run_deg_cv.py \
    --cancer "$CANCER" \
    --config "$CONFIG" \
    --tcga_umi_dir "$TCGA_UMI_DIR" \
    --surv_label_dir "$SURV" \
    --out_dir "output/deg_cv/${CANCER}" \
    $GENE_ARG \
    --num_folds 5
done

echo ""
echo "=== Done. Output: output/deg_cv/{CANCER}/degs_fold{1..5}.csv ==="
