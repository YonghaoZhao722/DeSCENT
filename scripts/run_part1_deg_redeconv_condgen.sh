#!/bin/bash
# Part 1: ReDeconv (reference + bulk -> fraction) -> scDiffusion training -> condgen (fraction -> scGEP)
# Per-fold DEG should already be available under data/${CANCER}/refs/deg_cv/ for Part 2.
# Paths from config/path_local.json (synced with scDiffusion-main/path.json).
# ReDeconv needs: redeconv_ref (Meta_data_new.tsv, scRNA_seq_new_noShift.tsv), bulk_tpm
#
# Usage: conda activate descent
#        export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
#        ./scripts/run_part1_deg_redeconv_condgen.sh [CANCER] [CONDGEN_SAMPLE_LIMIT]
set -e
DESCENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DESCENT_ROOT"

[[ -n "$CONDA_PREFIX" && -d "$CONDA_PREFIX/lib" ]] && export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Clear GPU memory after each Python step (avoids OOM when running full pipeline)
gpu_cleanup() { python -c "import gc; import torch; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true; }

CANCER="${1:-BRCA}"
CONDGEN_SAMPLE_LIMIT="${2:-${CONDGEN_SAMPLE_LIMIT:-4}}"
SCDIFFUSION_TRAIN_MODE="${SCDIFFUSION_TRAIN_MODE:-auto}"
SCDIFFUSION_QUIET="${SCDIFFUSION_QUIET:-1}"
QUIET_ARGS=()
if [[ "${SCDIFFUSION_QUIET}" == "1" ]]; then
  QUIET_ARGS=(--quiet)
fi
echo "=== Part 1: ReDeconv + scDiffusion Training + Condgen (${CANCER}) ==="
echo "  scDiffusion train mode: ${SCDIFFUSION_TRAIN_MODE}"
echo "  scDiffusion quiet: ${SCDIFFUSION_QUIET}"
echo "  DEG: data/${CANCER}/refs/deg_cv/ (used by Part 2)"

echo ""
echo "=== 1. ReDeconv (reference + bulk -> fraction) ==="
mkdir -p "output/redeconv_fraction/${CANCER}"
cd scgep_generation
python pipeline_scripts/run_redeconv_full.py \
  --cancer "${CANCER}" \
  --config "$DESCENT_ROOT/config/path_local.json" \
  --out_dir "$DESCENT_ROOT/output/redeconv_fraction/${CANCER}"
gpu_cleanup
cd ..
echo "  -> output/redeconv_fraction/${CANCER}/${CANCER}_ratios_redeconv_10.csv"

echo ""
echo "=== 2. scDiffusion training / fine-tuning ==="
python scgep_generation/pipeline_scripts/run_scdiffusion_training.py \
  --cancer "${CANCER}" \
  --config config/path_local.json \
  --train_mode "${SCDIFFUSION_TRAIN_MODE}" \
  "${QUIET_ARGS[@]}"
gpu_cleanup

echo ""
echo "=== 3. Condgen (fraction -> scGEP) ==="
FRAC_FULL="output/redeconv_fraction/${CANCER}/${CANCER}_ratios_redeconv_10.csv"
FRAC_INPUT="$DESCENT_ROOT/${FRAC_FULL}"
NUM_CLASS="$(python - <<PY
import pandas as pd
df = pd.read_csv("${FRAC_FULL}")
print(df.shape[1] - 1)
PY
)"
NUM_GENES="$(python - <<PY
import pandas as pd
print(len(pd.read_csv("${DESCENT_ROOT}/data/${CANCER}/refs/${CANCER}_gene_order.csv")))
PY
)"

if [[ "$CONDGEN_SAMPLE_LIMIT" =~ ^[0-9]+$ ]] && (( CONDGEN_SAMPLE_LIMIT > 0 )); then
  FRAC_MINI="output/redeconv_fraction/${CANCER}/${CANCER}_ratios_mini.csv"
  awk -v n="$CONDGEN_SAMPLE_LIMIT" 'NR == 1 || NR <= n + 1' "$FRAC_FULL" > "$FRAC_MINI"
  FRAC_INPUT="$DESCENT_ROOT/${FRAC_MINI}"
  echo "  Using first ${CONDGEN_SAMPLE_LIMIT} samples for condgen -> ${FRAC_MINI}"
else
  echo "  Using full fraction file for condgen -> ${FRAC_FULL}"
fi

cd scgep_generation
python generate_bulk_from_diffusion.py \
  --model_path "$(python -c "import json,os; c=json.load(open('$DESCENT_ROOT/config/path_local.json')); p=c['${CANCER}']['diffusion_backbone']; print(os.path.normpath(os.path.join('$DESCENT_ROOT', p)) if p and not os.path.isabs(p) else p)")" \
  --classifier_path "$(python -c "import json,os; c=json.load(open('$DESCENT_ROOT/config/path_local.json')); p=c['${CANCER}']['classifier']; print(os.path.normpath(os.path.join('$DESCENT_ROOT', p)) if p and not os.path.isabs(p) else p)")" \
  --vae_path "$(python -c "import json,os; c=json.load(open('$DESCENT_ROOT/config/path_local.json')); p=c['${CANCER}']['VAE']; print(os.path.normpath(os.path.join('$DESCENT_ROOT', p)) if p and not os.path.isabs(p) else p)")" \
  --cell_ratios_file "$FRAC_INPUT" \
  --num_class "$NUM_CLASS" \
  --out_dir "$DESCENT_ROOT/output/scgep_condgen/${CANCER}/redeconv" \
  --cell_counts 2048 \
  --num_genes "$NUM_GENES" \
  --gene_order_file "$DESCENT_ROOT/data/${CANCER}/refs/${CANCER}_gene_order.csv" \
  "${QUIET_ARGS[@]}"
gpu_cleanup
cd ..
echo "  -> output/scgep_condgen/${CANCER}/redeconv/"

echo ""
echo "=== Part 1 complete ==="
