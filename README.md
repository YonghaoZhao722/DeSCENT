# DeSCENT

DeSCENT: Deconvolution-guided Single-Cell Expression for Survival Prediction. This repository contains the code for scGEP (single-cell Gene Expression Profile) generation and multimodal survival prediction.

## Quick Start

```bash
# Create conda environment
conda env create -f environment.yml
conda activate descent
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Test survival CV (2 folds, 2 epochs)
./scripts/run_survival_test.sh
```

## Pipeline Run (Two Scripts)

The pipeline is split into two scripts. Paths are read from `config/path_local.json` (synced with `scDiffusion-main/path.json`).

### Part 1: DEG + ReDeconv + Condgen

DEG CV → ReDeconv (reference + bulk → fraction) → conditional generation (fraction → scGEP).

```bash
./scripts/run_part1_deg_redeconv_condgen.sh [BRCA]
```

- **DEG**: Per-fold differential genes from bulk + survival labels
- **ReDeconv**: 3-step workflow (signature genes → mean/std → deconvolution). Needs `redeconv_ref` (Meta_data_new.tsv, scRNA_seq_new_noShift.tsv) and `bulk_tpm` in config
- **Condgen**: scDiffusion generates scGEP from cell fractions (mini: 4 samples × 64 cells for quick test)

### Part 2: Multimodal Survival

5-fold CV survival analysis using sc_npz (pre-generated) + bulk + per-fold DEG.

```bash
./scripts/run_part2_survival.sh [BRCA]
```

If Part 1 fraction output is missing, Part 2 copies pre-computed `BRCA_celltypes.tsv` from ReDeconv for reference.

### Run Full Pipeline

```bash
./scripts/run_full_pipeline_test.sh [BRCA]
```

Runs Part 1 then Part 2 in sequence.

---

## Chapter 1: scGEP Generation

This chapter covers the pipeline to generate single-cell-like gene expression profiles (scGEP) from bulk RNA-seq: weighted CLTS normalization, cell fraction prediction (ReDeconv), and conditional generation (scDiffusion).

### Installation

We recommend installing the following dependencies:

```bash
# ReDeconv: MUST use the bundled redeconv/ in scgep_generation/ (do NOT pip install redeconv)
# The bundled version contains modifications; pip install would overwrite with vanilla release.

# scDiffusion (VAE, diffusion, conditional generation)
# Dependencies: torch, scanpy, blobfile, guided_diffusion (included)
pip install torch scanpy blobfile
```

The `scgep_generation/` folder contains the ReDeconv package (`redeconv/`) and scDiffusion components (`VAE/`, `guided_diffusion/`). Add the DeSCENT root to `PYTHONPATH` when running:

```bash
export PYTHONPATH=/path/to/DeSCENT:$PYTHONPATH
```

### 1.1 Weighted CLTS Normalization

Weighted CLTS (Cell-type-specific Library size and Transcriptome Size) normalizes single-cell RNA-seq for deconvolution. The implementation is in `scgep_generation/redeconv/ReDeconv_Normalization.py`.

**Input:**
- `fn_meta`: scRNA-seq metadata (cell ID, cell type) – e.g. `sc_meta.tsv`
- `fn_exp`: raw scRNA-seq expression – e.g. `expression_sc.tsv`

**Output:**
- `Ctype_size_means.tsv`, `Ctype_cell_counts.tsv`, `Cell_trans_sizes.tsv`
- Normalized expression for downstream ReDeconv

**How to run:**

```bash
cd scgep_generation
# Edit fn_meta, fn_exp in redeconv/ReDeconv_Normalization.py for your paths
python -m redeconv.ReDeconv_Normalization
# Interactive: choose option 4 for normalization
```

### 1.2 Cell Fraction Prediction (ReDeconv)

ReDeconv predicts cell type fractions from bulk RNA-seq using a 3-step workflow.

**Step 1 – Find signature genes:**
```bash
cd scgep_generation
# Edit fn_meta, fn_exp, fn_ini_sig in redeconv/ReDeconv_Percentage.py
python -m redeconv.ReDeconv_Percentage
# Choose option 1
```

**Step 2 – Compute mean/std of signature genes:**
```bash
# Choose option 2
```

**Step 3 – Deconvolution:**
```bash
# Choose option 3
# Input: fn_mean_std, fn_bulk_RNAseq_raw
# Output: fn_percentage_save (cell fractions)
```

**Non-interactive (batch):**
```bash
python pipeline_scripts/run_redeconv.py \
  --cancer LUAD \
  --path_json ../config/path.json \
  --pseudobulk_root /path/to/pseudobulk \
  --out_root /path/to/redeconv_output \
  --mean_std_dir /path/to/mean_std_files
```

### 1.3 Conditional Generation (scDiffusion)

Given cell fractions from ReDeconv, scDiffusion generates scGEP (single-cell-like profiles) conditioned on cell type composition.

**Prerequisites:** Trained VAE, diffusion backbone, and classifier (see `VAE/VAE_train.py`, `cell_train.py`, `classifier_train.py`).

**Run conditional generation:**
```bash
cd scgep_generation
python pipeline_scripts/run_diffusion_condgen.py \
  --path_json ../config/path.json \
  --out_root /path/to/output/npz \
  --ratio_base /path/to/redeconv_fraction \
  --include_cancers LUAD,HNSC \
  --sources redeconv
```

---

## Chapter 2: Survival Prediction

This chapter uses paired bulk RNA-seq and generated scGEP for multimodal survival analysis (5-fold CV, C-index).

### Data Requirements

- **Bulk expression:** `train_data_1.csv` … `train_data_5.csv`, `val_data_1.csv` … `val_data_5.csv` (rows = samples, cols = genes)
- **Survival labels:** `train_data_1.csv` … `val_data_5.csv` (columns: sample ID, OS, OS.time)
- **scGEP:** Per-sample folders under `sc_npz_root` (e.g. `cells_2048_TCGA-XX-XXXX/`)

Edit `config/path_local.json` (or `path.json`) to set paths for each cancer type. For BRCA, ensure `redeconv_ref` and `bulk_tpm` are set for ReDeconv.

### Run Survival CV

```bash
cd survival_prediction
python scrna_bulk_sc_survival_cv.py --cancer HNSC --config ../config/path.json
```

Or with explicit paths:
```bash
python scrna_bulk_sc_survival_cv.py \
  --sc_npz_root /path/to/cell_embs \
  --gene_list_csv /path/to/genes.csv \
  --deg_csv /path/to/degs.csv \
  --bulk_dir /path/to/bulk \
  --surv_label_dir /path/to/surv_label \
  --vae_ckpt_path /path/to/vae.pt \
  --vae_num_genes 28952 \
  --results_dir /path/to/output
```

### Per-fold DEG (Nested CV)

To avoid data leakage, run DEG per fold using train samples only:

```bash
python run_deg_cv.py \
  --bulk_dir /path/to/bulk \
  --surv_label_dir /path/to/surv_label \
  --out_dir /path/to/degs_per_fold
```

Then run survival CV with per-fold DEG:
```bash
python scrna_bulk_sc_survival_cv.py --cancer HNSC --deg_dir /path/to/degs_per_fold
```

---

## Directory Structure

```
DeSCENT/
├── scgep_generation/       # Chapter 1
│   ├── preprocessing/      # CLTS wrapper
│   ├── redeconv/           # ReDeconv (CLTS, cell fraction)
│   ├── VAE/                # VAE model and training
│   ├── guided_diffusion/   # Diffusion, classifier
│   └── pipeline_scripts/   # run_redeconv, run_diffusion_condgen
├── survival_prediction/    # Chapter 2
│   ├── scrna_bulk_sc_survival_cv.py
│   ├── scrna_bulk_sc_survival.py
│   ├── mil_survival_model.py
│   ├── mil_survival_training.py
│   ├── run_deg_cv.py
│   ├── survival_data.py   # Data prep utilities
│   └── bulk_sample.py     # Bulk prep for DEG
├── config/
│   ├── path.json           # Per-cancer paths (from scDiffusion-main)
│   └── path_local.json     # Project paths (bulk, surv_label, redeconv_ref, bulk_tpm)
├── scripts/
│   ├── run_part1_deg_redeconv_condgen.sh  # DEG + ReDeconv + condgen
│   ├── run_part2_survival.sh             # Multimodal survival
│   └── run_full_pipeline_test.sh         # Part 1 + Part 2
└── README.md
```
