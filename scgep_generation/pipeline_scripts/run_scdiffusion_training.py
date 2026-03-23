from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import scanpy as sc


DESCENT_ROOT = Path(__file__).resolve().parents[2]
SCGEP_ROOT = DESCENT_ROOT / "scgep_generation"


def resolve(project_root: Path, p: str | None) -> Path | None:
    if not p:
        return None
    pp = Path(p)
    return pp if pp.is_absolute() else (project_root / pp).resolve()


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full scDiffusion training chain for DeSCENT.")
    parser.add_argument("--cancer", required=True)
    parser.add_argument("--config", default="config/path_local.json")
    parser.add_argument("--train_mode", choices=["auto", "force", "skip"], default="auto")
    parser.add_argument("--vae_max_steps", type=int, default=200000)
    parser.add_argument("--diffusion_lr_anneal_steps", type=int, default=500000)
    parser.add_argument("--classifier_iterations", type=int, default=500000)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def latest_checkpoint(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No checkpoint matching {pattern} found in {directory}")
    return matches[-1]


def export_checkpoint(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"Exported {src} -> {dst}")


def select_best_or_latest(directory: Path, best_pattern: str, latest_pattern: str) -> Path:
    best_matches = sorted(directory.glob(best_pattern))
    if best_matches:
        return best_matches[-1]
    return latest_checkpoint(directory, latest_pattern)


def run_command(cmd: list[str], cwd: Path):
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    args = parse_args()
    with open(DESCENT_ROOT / args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cancer_cfg = cfg[args.cancer.upper()]

    train_h5ad = resolve(DESCENT_ROOT, cancer_cfg["single_cell_data"]["train"])
    test_h5ad = resolve(DESCENT_ROOT, cancer_cfg["single_cell_data"]["test"])
    vae_pretrained = resolve(DESCENT_ROOT, cancer_cfg.get("VAE_pretrained"))
    stable_vae = resolve(DESCENT_ROOT, cancer_cfg["VAE"])
    stable_diffusion = resolve(DESCENT_ROOT, cancer_cfg["diffusion_backbone"])
    stable_classifier = resolve(DESCENT_ROOT, cancer_cfg["classifier"])

    stable_outputs = [stable_vae, stable_diffusion, stable_classifier]
    missing_outputs = [p for p in stable_outputs if not p.exists()]
    if args.train_mode == "skip":
        print("Skipping scDiffusion training (--train_mode=skip).")
        return
    if args.train_mode == "auto" and not missing_outputs:
        print("All stable scDiffusion checkpoints already exist. Skipping training.")
        return

    if not train_h5ad.exists():
        raise FileNotFoundError(f"Training h5ad not found: {train_h5ad}")
    if not test_h5ad.exists():
        raise FileNotFoundError(f"Validation h5ad not found: {test_h5ad}")
    if not vae_pretrained or not vae_pretrained.exists():
        raise FileNotFoundError(
            "Downloaded pretrained scimilarity checkpoint is required for VAE fine-tuning. "
            "Expected config key 'VAE_pretrained' to point to the downloaded annotation_model_v1 directory."
        )

    adata = sc.read_h5ad(train_h5ad)
    num_genes = int(adata.n_vars)
    celltype_col = "celltype" if "celltype" in adata.obs.columns else "Cell_type"
    if celltype_col not in adata.obs.columns:
        raise ValueError("Training h5ad must contain a 'celltype' or 'Cell_type' column.")
    num_class = int(pd.Series(adata.obs[celltype_col].astype(str)).nunique())
    print(f"Training scDiffusion for {args.cancer.upper()}: num_genes={num_genes}, num_class={num_class}")

    training_root = DESCENT_ROOT / "output" / "scdiffusion_training" / args.cancer.upper()
    vae_dir = training_root / "vae"
    diffusion_dir = training_root / "diffusion"
    classifier_dir = training_root / "classifier"
    for path in (vae_dir, diffusion_dir, classifier_dir):
        path.mkdir(parents=True, exist_ok=True)

    python = sys.executable

    run_command(
        [
            python,
            "VAE/VAE_train.py",
            "--data_dir",
            str(train_h5ad),
            "--num_genes",
            str(num_genes),
            "--state_dict",
            str(vae_pretrained),
            "--save_dir",
            str(vae_dir),
            "--max_steps",
            str(args.vae_max_steps),
            *(["--quiet"] if args.quiet else []),
        ],
        cwd=SCGEP_ROOT,
    )
    vae_ckpt = select_best_or_latest(vae_dir, "best_model_*.pt", "model_seed=*_step=*.pt")
    export_checkpoint(vae_ckpt, stable_vae)

    diffusion_model_name = f"{args.cancer.upper()}_diffusion"
    run_command(
        [
            python,
            "cell_train.py",
            "--data_dir",
            str(train_h5ad),
            "--vae_path",
            str(stable_vae),
            "--model_name",
            diffusion_model_name,
            "--save_dir",
            str(diffusion_dir),
            "--lr_anneal_steps",
            str(args.diffusion_lr_anneal_steps),
            *(["--quiet"] if args.quiet else []),
        ],
        cwd=SCGEP_ROOT,
    )
    diffusion_ckpt_dir = diffusion_dir / diffusion_model_name
    diffusion_ckpt = select_best_or_latest(diffusion_ckpt_dir, "best_model_*.pt", "model*.pt")
    export_checkpoint(diffusion_ckpt, stable_diffusion)

    classifier_model_dir = classifier_dir / f"{args.cancer.upper()}_classifier"
    run_command(
        [
            python,
            "classifier_train.py",
            "--data_dir",
            str(train_h5ad),
            "--val_data_dir",
            str(test_h5ad),
            "--vae_path",
            str(stable_vae),
            "--num_class",
            str(num_class),
            "--model_path",
            str(classifier_model_dir),
            "--iterations",
            str(args.classifier_iterations),
            *(["--quiet"] if args.quiet else []),
        ],
        cwd=SCGEP_ROOT,
    )
    classifier_ckpt = select_best_or_latest(classifier_model_dir, "best_model_*.pt", "model*.pt")
    export_checkpoint(classifier_ckpt, stable_classifier)

    print("\nscDiffusion training complete.")
    print(f"  VAE              -> {stable_vae}")
    print(f"  diffusion        -> {stable_diffusion}")
    print(f"  classifier       -> {stable_classifier}")


if __name__ == "__main__":
    main()
