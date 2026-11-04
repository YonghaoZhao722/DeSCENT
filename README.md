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
```

The repository already ships a repo-local BRCA demo config at `config/path_local.json`.

Use either entrypoint:

- Notebook: **[`notebooks/descent_pipeline_demo.ipynb`](notebooks/descent_pipeline_demo.ipynb)**
- Shell scripts:
  ```bash
  ./scripts/run_part1_deg_redeconv_condgen.sh BRCA
  ./scripts/run_part2_survival.sh BRCA
  # or
  ./scripts/run_full_pipeline_test.sh BRCA
  ```

## Prerequisites

Before running the BRCA demo, download the required assets from Zenodo:

- BRCA demo data: [Zenodo](https://zenodo.org/records/19182224)
- scimilarity pretrained checkpoint `annotation_model_v1`: [Zenodo](https://zenodo.org/records/8286452)

The BRCA demo data record contains both:

- the BRCA single-cell train/test `.h5ad` files
- the BRCA bulk input files used by ReDeconv and survival CV

Bulk files are not stored in Git because several files exceed GitHub's 100 MB file size limit.

Place the downloaded files under:

- `data/BRCA/single_cell/`
- `data/BRCA/bulk/`
- `data/pretrained/annotation_model_v1/`

## Onboarding

For the BRCA demo, the repo expects the following layout:

- `config/path_local.json`: active BRCA config used by the shell scripts and notebook
- `data/BRCA/refs/deg_cv/degs_fold{1..5}.csv`: per-fold DEG inputs for survival CV
- `data/BRCA/bulk/`: bulk expression splits plus `filtered_tpm_BRCA.tsv`, **downloaded from the same Zenodo record as the BRCA single-cell demo data**
- `data/BRCA/single_cell/BRCA_train.symbol_mapped.h5ad` and `data/BRCA/single_cell/BRCA_test.symbol_mapped.h5ad`: single-cell train/test `.h5ad` **downloaded from the same Zenodo record**
- `data/pretrained/annotation_model_v1/`: **downloaded scimilarity pretrained checkpoint**
- `data/BRCA/redeconv_ref/Meta_data_new.tsv` and `data/BRCA/redeconv_ref/scRNA_seq_new_noShift.tsv`: ReDeconv reference files required by part 1

For a new cancer type, use `config/path.json.example` as the key/layout reference and edit `config/path_local.json` to match the same repo-relative directory layout under `data/<CANCER>/...`.

Training policy in this repo is:

- `VAE`: fine-tune from `annotation_model_v1`
- `diffusion_backbone`: train from scratch
- `classifier`: train from scratch

`./scripts/run_part1_deg_redeconv_condgen.sh` now supports `SCDIFFUSION_TRAIN_MODE`:

- `auto` (default): train only when stable checkpoints are missing
- `force`: always rerun VAE fine-tuning + backbone training + classifier training
- `skip`: never train, only reuse the configured checkpoints

## Pipeline

DeSCENT has four stages: ReDeconv → scDiffusion training → Condgen → Survival CV.

```
bulk RNA-seq + scRNA-seq reference
             │
             ▼
         ReDeconv          ──► cell fractions
             │
             ▼
    scDiffusion training   ──► VAE / diffusion / classifier checkpoints
             │
             ▼
    scDiffusion condgen    ──► synthetic scGEP (.npz)
             │
             ▼
    Multimodal survival    ──► C-index
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
│   ├── scrna_bulk_sc_survival_cv.py   # Main entry: outer-fold CV driver
│   ├── scrna_bulk_sc_survival.py      # Fusion model + patient bag dataset
│   ├── bulk_mlp_survival_cv.py        # Pure-bulk baseline (the comparison arm)
│   ├── mil_survival_model.py          # MIL-based multimodal model
│   ├── mil_survival_training.py       # Training utilities (losses, time bins)
│   └── run_deg_cv.py                  # Per-fold DEG computation
├── notebooks/
│   └── descent_pipeline_demo.ipynb    # Interactive pipeline demo
├── config/
│   ├── path.json.example              # template for repo-relative path config
│   ├── path_local.json                # BRCA
│   ├── path_coad_scgep.json           # COAD, KIRC, LGG
│   └── path_seetacloud_survival.json  # all 8 cancers, uniform layout
├── reproduce/
│   ├── commands.csv                   # the 72 commands behind the published table
│   └── baseline.md                    # pure-bulk baseline command
├── scripts/
│   ├── evaluate_bulk_mlp_external.py  # external validation, baseline
│   ├── evaluate_fusion_external.py    # external validation, fusion
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

## Reproducing the published results

`reproduce/commands.csv` is the complete, literal command list behind the main table
(8 cancers x 3 survival heads x {C-index, td-AUC, IBS}). Each row carries the metric, the
cancer, the head, the reported value, and the exact shell command that produced it.

```bash
# print every command
python -c "import csv;[print(r['command']) for r in csv.DictReader(open('reproduce/commands.csv'))]"

# run one cell, e.g. BRCA / Cox
awk -F',' '$1=="C-index" && $2=="BRCA" && $3=="cox"' reproduce/commands.csv
```

Each command trains one outer fold per process (`--folds $FOLD`) and writes
`cv_summary.json` under its own `--results_dir`; the reported number is the mean of the
five folds' `test_c_index`.

**Determinism.** Every command passes `--deterministic`, which additionally requires
`CUBLAS_WORKSPACE_CONFIG` to be exported *before* the process starts:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Without it cuBLAS keeps a non-deterministic workspace and repeated runs of the same fold
drift; with it, repeated runs of a fold agree to all printed digits.

**Verified.** Re-running the BRCA / Cox / fold-1 row of `reproduce/commands.csv` with this
code reproduces the published fold bit-for-bit: `best_epoch` 60, held-out
`test_c_index` 0.72798819478603, `test_td_auc` 0.757300615258627,
`test_integrated_brier_score` 0.18168747738618 — identical to 15 significant digits.

**Flags that are load-bearing but never appear on the command line.** `--batch_size`
(default 32) and `--max_cells` (default 1024) are taken from the argparse defaults in
`survival_prediction/scrna_bulk_sc_survival_cv.py`. Changing those defaults changes the
model, not just its speed.

**`--diag` is not optional.** It looks like a pure logging flag — it records per-epoch
validation/test diagnostics and never enters checkpoint selection — but its two extra
evaluation passes per epoch change the CUDA allocation sequence, and under
`torch.use_deterministic_algorithms(..., warn_only=True)` that is enough to change the
trained weights. Measured on BRCA / Cox / fold 1: with `--diag` the run reproduces the
published fold bit-for-bit; without it, epoch 1 still matches exactly but epoch 2 diverges
and the fold lands at 0.6936 instead of 0.7280. Keep the flag on every command.

**External validation.** `scripts/evaluate_bulk_mlp_external.py` and
`scripts/evaluate_fusion_external.py` score trained checkpoints on the held-out GEO/CPTAC
cohorts and report C-index, td-AUC, a follow-up-restricted IBS, and IPA against the
evaluation cohort's own Kaplan-Meier null. They read `model.pt` from each fold directory. The commands
in `reproduce/commands.csv` pass `--discard_ckpt`, which deletes it once the fold has been
scored, so drop that flag on any run you intend to validate externally.

**Baseline.** The pure-bulk comparison arm is `survival_prediction/bulk_mlp_survival_cv.py`;
`reproduce/baseline.md` carries its command and the head-matching table. Compare head to
head only: `--direct_cox_from_fusion` against the baseline's Cox head, the default MLP head
against DeepSurv, and `--loss_fn deephit` against DeepHit.

## Important Notes

- **Bundled ReDeconv**: `scgep_generation/redeconv/` is a patched fork. **DO NOT RUN** `pip install redeconv`, it will silently replace it with the vanilla version and break the pipeline.
- **GPU memory**: Pipeline scripts call `gpu_cleanup()` between steps to avoid OOM when running sequentially.
- **Per-fold DEG only**: part 2 reads `degs_fold{1..5}.csv` from `data/{CANCER}/refs/deg_cv/`. There is no fallback to a single global DEG file.
- **Stable checkpoint paths**: part 1 exports trained checkpoints to `output/scdiffusion_models/{CANCER}/`, and both condgen and survival read those paths from `config/path_local.json`.

## Support

If you run into issues when using DeSCENT, please open a [GitHub issue](https://github.com/YonghaoZhao722/DeSCENT/issues). We check them and will follow up when we see them. Thank you.

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
