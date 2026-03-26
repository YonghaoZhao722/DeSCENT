<p align="center">
  <img src="data/fig1.png" width="800" alt="DeSCENT Overview">
</p>

<h1 align="center">DeSCENT</h1>

<p align="center">
  <b>De</b>convolutional <b>S</b>ingle-<b>C</b>ell RNA-seq <b>EN</b>hances <b>T</b>ranscriptome-based Cancer Survival Analysis
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#pipeline">Pipeline</a> &middot;
  <a href="#notebook-demo">Notebook Demo</a> &middot;
  <a href="#configuration">Configuration</a> &middot;
  <a href="#project-structure">Project Structure</a>
</p>

---

## Quick Start

```bash
git clone https://github.com/YonghaoZhao722/DeSCENT.git
cd DeSCENT
conda env create -f environment.yml
conda activate descent
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

> The `LD_LIBRARY_PATH` export fixes `CXXABI_1.3.15` errors on older systems. All shell scripts set it automatically.

The repository ships a repo-local BRCA demo config at `config/path_local.json`.
`config/path.json.example` is also a repo-local BRCA example, so a direct copy is runnable for the BRCA demo as long as the expected data files are present under `data/BRCA/`.
If you want to create or replace a local config from the template, run:

```bash
cp config/path.json.example config/path_local.json
```

Then edit `config/path_local.json` only if you want to change the demo defaults or add another cancer type.

Run:
**[`notebooks/descent_pipeline_demo.ipynb`](notebooks/descent_pipeline_demo.ipynb)**

## Onboarding

For the BRCA demo, the repo expects the following layout:

- `config/path_local.json`: active BRCA config used by the shell scripts and notebook
- `data/BRCA/refs/deg_cv/degs_fold{1..5}.csv`: per-fold DEG inputs for survival CV
- `data/BRCA/bulk/`: bulk expression splits plus `filtered_tpm_BRCA.tsv`, **downloaded from the same Zenodo record as the BRCA single-cell demo data**
- `data/BRCA/single_cell/BRCA_train.symbol_mapped.h5ad` and `data/BRCA/single_cell/BRCA_test.symbol_mapped.h5ad`: single-cell train/test `.h5ad` **downloaded from the same Zenodo record**
- `data/pretrained/annotation_model_v1/`: **downloaded scimilarity pretrained checkpoint**
- `data/BRCA/redeconv_ref/Meta_data_new.tsv` and `data/BRCA/redeconv_ref/scRNA_seq_new_noShift.tsv`: ReDeconv reference files required by part 1

For a new cancer type, you can start from the BRCA repo-local template with:

```bash
cp config/path.json.example config/path_local.json
```

Then edit `config/path_local.json` and keep the same repo-relative directory layout under `data/<CANCER>/...`.

## Training Data

The BRCA demo data is available on [Zenodo](https://zenodo.org/records/19182224). That record now contains both:

- the BRCA single-cell train/test `.h5ad` files
- the BRCA bulk input files used by ReDeconv and survival CV

Bulk files are not stored in Git because several files exceed GitHub's 100 MB file size limit.

After downloading, place the files at:

- `data/BRCA/single_cell/BRCA_train.symbol_mapped.h5ad`
- `data/BRCA/single_cell/BRCA_test.symbol_mapped.h5ad`
- `data/BRCA/bulk/filtered_tpm_BRCA.tsv`
- `data/BRCA/bulk/train_data_1.csv` ... `data/BRCA/bulk/train_data_5.csv`
- `data/BRCA/bulk/val_data_1.csv` ... `data/BRCA/bulk/val_data_5.csv`

## External Checkpoint Requirement

The VAE fine-tuning step requires the downloaded scimilarity pretrained checkpoint `annotation_model_v1`. This checkpoint is not bundled with DeSCENT.

- Download the [scimilarity pretrained weights at Zenodo](https://zenodo.org/records/8286452) and place the extracted `annotation_model_v1` directory at `data/pretrained/annotation_model_v1/`
- DeSCENT uses that directory through `config/path_local.json -> VAE_pretrained`

Training policy in this repo is:

- `VAE`: fine-tune from `annotation_model_v1`
- `diffusion_backbone`: train from scratch
- `classifier`: train from scratch

`./scripts/run_part1_deg_redeconv_condgen.sh` now supports `SCDIFFUSION_TRAIN_MODE`:

- `auto` (default): train only when stable checkpoints are missing
- `force`: always rerun VAE fine-tuning + backbone training + classifier training
- `skip`: never train, only reuse the configured checkpoints

## Pipeline

DeSCENT has four stages: ReDeconv → scDiffusion training/fine-tuning → Condgen → Survival CV.

```
bulk RNA-seq + scRNA-seq reference
        │
        ▼
  ReDeconv ──► cell fractions
        │
        ▼
  scDiffusion training/fine-tuning ──► VAE / diffusion / classifier checkpoints
        │
        ▼
  scDiffusion condgen ──► synthetic scGEP (.npz)
        │
        ▼
  Multimodal survival CV (bulk + scGEP + DEG) ──► C-index
```

### Option A: Shell Scripts

```bash
# Part 1: ReDeconv + scDiffusion training + conditional scGEP generation
./scripts/run_part1_deg_redeconv_condgen.sh BRCA

# Part 2: Multimodal survival prediction (5-fold CV)
./scripts/run_part2_survival.sh BRCA

# Or run the full pipeline in one go
./scripts/run_full_pipeline_test.sh BRCA
```

### Option B: Jupyter Notebook

An interactive notebook covers the same pipeline with inline visualizations:

**[`notebooks/descent_pipeline_demo.ipynb`](notebooks/descent_pipeline_demo.ipynb)**

The notebook runs each step via `!python` shell commands (faithful to the shell scripts) and produces plots for cell fractions, training progress, generated scGEP stats, C-index per fold, and training curves. Change `CANCER = "BRCA"` in the first code cell to switch cancer types, and adjust `CONDGEN_SAMPLE_LIMIT = 4` there to control how many samples the condgen step uses (`None` or `0` runs the full fraction file).

> **Note:** The full pipeline is compute-intensive. scDiffusion training and condgen require GPU time, and survival CV trains for 250 epochs × 5 folds. For a quick demo, the notebook defaults to `SCDIFFUSION_TRAIN_MODE = "auto"` and `CONDGEN_SAMPLE_LIMIT = 4`, and you can reduce `EPOCHS` to 10.

## Notebook Demo

| Step | What it does | Output |
|------|-------------|--------|
| Step 1 — ReDeconv | Estimate cell-type fractions from bulk RNA-seq | `output/redeconv_fraction/{CANCER}/` |
| Step 2 — scDiffusion Training | Fine-tune VAE and train the diffusion backbone + classifier | `output/scdiffusion_training/{CANCER}/` and `output/scdiffusion_models/{CANCER}/` |
| Step 3 — Condgen | Generate synthetic scGEP via scDiffusion | `output/scgep_condgen/{CANCER}/redeconv/` |
| Step 4 — Survival CV | Multimodal 5-fold CV (bulk + scGEP + DEG) | `output/survival_cv/{CANCER}/cv_summary.json` |

Each step includes a visualization cell: stacked bar / box plots for cell fractions, .npz inspection for condgen, and C-index bar chart + training curves for survival.

## Configuration

`config/path_local.json` is the active repo-local config for the BRCA demo. `config/path.json.example` is a runnable BRCA template that can be copied directly and then edited for other cancer entries while keeping all paths relative to the repository root.

| Key | Purpose |
|-----|---------|
| `single_cell_data` | Downloaded train/test `.h5ad` files used by scDiffusion training |
| `VAE_pretrained` | Downloaded scimilarity pretrained directory (`annotation_model_v1`) used for VAE fine-tuning |
| `VAE`, `diffusion_backbone`, `classifier` | Stable trained checkpoints exported by DeSCENT under `output/scdiffusion_models/` |
| `sc_npz` | Generated scGEP directory consumed by part 2 |
| `bulk`, `surv_label` | Bulk expression and survival label directories (5-fold splits) |
| `deg_dir` | Per-fold DEG directory containing `degs_fold{1..5}.csv` for leakage-free survival CV |
| `gene_list` | Gene order CSV for generation |
| `redeconv_ref` | Reference scRNA-seq for ReDeconv |
| `bulk_tpm` | TPM-normalized bulk for ReDeconv input |
| `celltypes` | Cell fraction output from ReDeconv |

Verified cancers: **BRCA**, **COAD**, **HNSC**, **KIRC**, **LGG**, **LIHC**, **LUAD**, **STAD**.

## Project Structure

```
DeSCENT/
├── scgep_generation/           # Chapter 1: scGEP Generation
│   ├── redeconv/               # Bundled ReDeconv (patched fork — do NOT pip install)
│   ├── VAE/                    # VAE encoder/decoder for gene expression latent space
│   ├── guided_diffusion/       # Diffusion backbone + classifier
│   ├── pipeline_scripts/       # ReDeconv + scDiffusion orchestration helpers
│   ├── cell_train.py           # Diffusion backbone training entrypoint
│   ├── classifier_train.py     # Classifier training entrypoint
│   └── generate_bulk_from_diffusion.py
├── survival_prediction/        # Chapter 2: Survival Prediction
│   ├── scrna_bulk_sc_survival_cv.py   # Main entry: 5-fold CV
│   ├── mil_survival_model.py          # MIL-based multimodal model
│   ├── mil_survival_training.py       # Training utilities
│   ├── survival_data.py               # Data loading
│   └── bulk_sample.py                 # Bulk prep
├── notebooks/
│   └── descent_pipeline_demo.ipynb    # Interactive pipeline demo
├── config/
│   ├── path.json               # legacy local config (ignored)
│   ├── path.json.example       # template for repo-relative path config
│   └── path_local.json         # active BRCA demo config
├── scripts/
│   ├── run_part1_deg_redeconv_condgen.sh
│   ├── run_part2_survival.sh
│   ├── run_full_pipeline_test.sh
│   └── run_survival_test.sh
├── output/                     # All outputs (not in git)
│   ├── redeconv_fraction/
│   ├── scdiffusion_models/
│   ├── scdiffusion_training/
│   ├── scgep_condgen/
│   └── survival_cv/
├── data/                       # Input data, DEG folds, and model prerequisites
└── environment.yml
```

## Important Notes

- **Bundled ReDeconv**: `scgep_generation/redeconv/` is a patched fork. **DO NOT RUN** `pip install redeconv`, it will silently replace it with the vanilla version and break the pipeline.
- **GPU memory**: Pipeline scripts call `gpu_cleanup()` between steps to avoid OOM when running sequentially.
- **Per-fold DEG only**: part 2 reads `degs_fold{1..5}.csv` from `data/{CANCER}/refs/deg_cv/`. There is no fallback to a single global DEG file.
- **Stable checkpoint paths**: part 1 exports trained checkpoints to `output/scdiffusion_models/{CANCER}/`, and both condgen and survival read those paths from `config/path_local.json`.

## Citation

If you use DeSCENT in your research, please cite:

```bibtex
@article{zhao2026descent,
  title={DeSCENT: Deconvolutional Single-Cell RNA-seq Enhances Transcriptome-based Cancer Survival Analysis},
  author={Zhao, Yonghao and You, Zeyu and Shen, Yu and Chu, Jielei and Gong, Xun and Li, Tianrui and Wang, Ziqiang and Xu, Chuan and Luo, Zhipeng and He, Yazhou},
  journal={bioRxiv},
  year={2026},
  doi={10.64898/2026.03.15.711877},
  url={https://doi.org/10.64898/2026.03.15.711877}
}
```
