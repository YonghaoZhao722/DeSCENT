#!/bin/bash
# Full pipeline test: DEG CV -> (ReDeconv) -> condgen -> survival
# Usage: conda activate descent
#        export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
#        ./scripts/run_full_pipeline_test.sh
#
# Note: ReDeconv requires BRCA_mean_std_FC2.0_TOP150.tsv (from single-cell reference).
#       ReDeconv repo has mean_std for LUAD/KIRC/HNSC etc. but not BRCA.
#       For BRCA we use pre-computed ratios (BRCA_celltypes.csv) for condgen.
set -e
DESCENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DESCENT_ROOT"

[[ -n "$CONDA_PREFIX" && -d "$CONDA_PREFIX/lib" ]] && export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "=== 1. DEG CV (per-fold differential genes) ==="
python survival_prediction/run_deg_cv.py \
  --bulk_dir data/BRCA/bulk \
  --surv_label_dir data/BRCA/surv_label \
  --out_dir output/deg_cv/BRCA \
  --num_folds 2
echo "  -> output/deg_cv/BRCA/degs_fold{1,2}.csv"

echo ""
echo "=== 2. ReDeconv (SKIPPED: no BRCA mean_std) ==="
echo "  ReDeconv needs BRCA_mean_std_FC2.0_TOP150.tsv from single-cell reference."
echo "  Using pre-computed output/redeconv_fraction/BRCA/BRCA_ratios_redeconv_10.csv"

echo ""
echo "=== 3. Condgen (fraction -> scGEP, mini test: 3 samples x 64 cells) ==="
mkdir -p output/redeconv_fraction/BRCA
[[ ! -f output/redeconv_fraction/BRCA/BRCA_ratios_redeconv_10.csv ]] && \
  cp /data/zhaoyh/scDiffusion-main/BRCA_celltypes.csv output/redeconv_fraction/BRCA/BRCA_ratios_redeconv_10.csv
head -4 output/redeconv_fraction/BRCA/BRCA_ratios_redeconv_10.csv > output/redeconv_fraction/BRCA/BRCA_ratios_mini.csv

cd scgep_generation
python generate_bulk_from_diffusion.py \
  --model_path /data/zhaoyh/scDiffusion-main/output/checkpoint/backbone/BRCA_diffusion/best_ema_0.9999_epoch=2594_step=804139_loss=0.003250.pt \
  --classifier_path /data/zhaoyh/scDiffusion-main/output/checkpoint/classifier/BRCA_classifier/best_model_epoch=576_step=357119_loss=1.736112.pt \
  --vae_path /data/zhaoyh/scDiffusion-main/VAE/output/BRCA_symbol/best_model_seed=0_epoch=323_step=199999_loss=0.041560.pt \
  --cell_ratios_file "$DESCENT_ROOT/output/redeconv_fraction/BRCA/BRCA_ratios_mini.csv" \
  --num_class 9 \
  --out_dir "$DESCENT_ROOT/output/scgep_condgen/BRCA/redeconv_mini" \
  --cell_counts 64 \
  --num_genes 17930 \
  --gene_order_file "$DESCENT_ROOT/data/BRCA/refs/BRCA_gene_order.csv"
cd ..
echo "  -> output/scgep_condgen/BRCA/redeconv_mini/"

echo ""
echo "=== 4. Survival CV (uses existing sc_npz + deg_dir for per-fold DEG) ==="
python survival_prediction/scrna_bulk_sc_survival_cv.py \
  --cancer BRCA \
  --config config/path_local.json \
  --deg_dir output/deg_cv/BRCA \
  --epochs 2 \
  --num_folds 2
echo "  -> output/survival_cv/BRCA/"

echo ""
echo "=== Full pipeline test complete ==="
