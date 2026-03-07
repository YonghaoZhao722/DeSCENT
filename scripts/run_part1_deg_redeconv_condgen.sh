#!/bin/bash
# Part 1: ReDeconv (reference + bulk -> fraction) -> condgen (fraction -> scGEP)
# DEG: use pre-computed output/deg_cv/${CANCER}/degs_fold{1..5}.csv (not computed here, not in git)
# Paths from config/path_local.json (synced with scDiffusion-main/path.json).
# ReDeconv needs: redeconv_ref (Meta_data_new.tsv, scRNA_seq_new_noShift.tsv), bulk_tpm
#
# Usage: conda activate descent
#        export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
#        ./scripts/run_part1_deg_redeconv_condgen.sh
set -e
DESCENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DESCENT_ROOT"

[[ -n "$CONDA_PREFIX" && -d "$CONDA_PREFIX/lib" ]] && export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Clear GPU memory after each Python step (avoids OOM when running full pipeline)
gpu_cleanup() { python -c "import gc; import torch; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true; }

CANCER="${1:-BRCA}"
echo "=== Part 1: ReDeconv + Condgen (${CANCER}) ==="
echo "  DEG: output/deg_cv/${CANCER}/ (local only, used by Part 2)"

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
echo "=== 2. Condgen (fraction -> scGEP) ==="
# Mini: first 4 samples, 64 cells each for quick test
head -4 "output/redeconv_fraction/${CANCER}/${CANCER}_ratios_redeconv_10.csv" > "output/redeconv_fraction/${CANCER}/${CANCER}_ratios_mini.csv"

cd scgep_generation
python generate_bulk_from_diffusion.py \
  --model_path "$(python -c "import json,os; c=json.load(open('$DESCENT_ROOT/config/path_local.json')); p=c['${CANCER}']['diffusion_backbone']; print(os.path.normpath(os.path.join('$DESCENT_ROOT', p)) if p and not os.path.isabs(p) else p)")" \
  --classifier_path "$(python -c "import json,os; c=json.load(open('$DESCENT_ROOT/config/path_local.json')); p=c['${CANCER}']['classifier']; print(os.path.normpath(os.path.join('$DESCENT_ROOT', p)) if p and not os.path.isabs(p) else p)")" \
  --vae_path "$(python -c "import json,os; c=json.load(open('$DESCENT_ROOT/config/path_local.json')); p=c['${CANCER}']['VAE']; print(os.path.normpath(os.path.join('$DESCENT_ROOT', p)) if p and not os.path.isabs(p) else p)")" \
  --cell_ratios_file "$DESCENT_ROOT/output/redeconv_fraction/${CANCER}/${CANCER}_ratios_mini.csv" \
  --num_class 9 \
  --out_dir "$DESCENT_ROOT/output/scgep_condgen/${CANCER}/redeconv" \
  --cell_counts 2048 \
  --num_genes 17930 \
  --gene_order_file "$DESCENT_ROOT/data/${CANCER}/refs/${CANCER}_gene_order.csv"
gpu_cleanup
cd ..
echo "  -> output/scgep_condgen/${CANCER}/redeconv/"

echo ""
echo "=== Part 1 complete ==="
