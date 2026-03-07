from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_SCGEP_ROOT = Path(__file__).resolve().parent.parent
_DESCENT_ROOT = _SCGEP_ROOT.parent
sys.path.insert(0, str(_SCGEP_ROOT))

from pipeline_scripts.common import (
    CancerPaths,
    ensure_dir,
    get_cancer_paths,
    infer_num_class_from_ratio_file,
    iter_cancers,
    load_path_json,
    read_gene_list,
)


def run_one(
    base_dir: str,
    cp: CancerPaths,
    ratio_csv: str,
    out_dir: str,
    cell_count: int,
) -> None:
    num_class = infer_num_class_from_ratio_file(ratio_csv)
    gene_order = read_gene_list(cp.gene_list)
    num_genes = len(gene_order)

    ensure_dir(out_dir)

    cmd = [
        "python",
        "generate_bulk_from_diffusion.py",
        "--model_path",
        cp.diffusion_backbone,
        "--classifier_path",
        cp.classifier_path,
        "--vae_path",
        cp.vae_path,
        "--cell_ratios_file",
        ratio_csv,
        "--num_class",
        str(num_class),
        "--out_dir",
        out_dir,
        "--cell_counts",
        str(cell_count),
        "--num_genes",
        str(num_genes),
        "--gene_order_file",
        cp.gene_list,
    ]

    print(f"\n[condgen] cancer={cp.cancer} num_class={num_class} num_genes={num_genes}")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=base_dir)


def get_ratio_path(cancer: str, source: str, ratio_base: str) -> Path:
    """Get the path to ratio CSV file based on source."""
    base = Path(ratio_base)
    
    if source == "gt":
        return base / "pseudobulk" / cancer / f"{cancer}_ratios_gt_10.csv"
    elif source == "redeconv":
        return base / "redeconv_fraction" / cancer / f"{cancer}_ratios_redeconv_10.csv"
    elif source == "b2space":
        return base / "scaden_fraction" / cancer / f"{cancer}_ratios_b2space_10.csv"
    elif source == "tape":
        # Note: TAPE output files are named with "b2space" suffix
        return base / "tape_fraction" / cancer / f"{cancer}_ratios_b2space_10.csv"
    else:
        raise ValueError(f"Unknown source: {source}")


def main():
    parser = argparse.ArgumentParser(description="Launch conditional diffusion generation for multiple cancers and ratio sources.")
    parser.add_argument("--path_json", type=str, default=str(_DESCENT_ROOT / "config" / "path.json"))
    parser.add_argument("--base_dir", type=str, default=str(_SCGEP_ROOT), help="DeSCENT scgep_generation root (cwd for generate_bulk_from_diffusion).")
    parser.add_argument("--ratio_base", type=str, default=str(_DESCENT_ROOT / "output"), help="Root dir for ratio CSVs (redeconv_fraction, pseudobulk, etc.).")
    parser.add_argument("--out_root", type=str, required=True, help="Output root directory for generated NPZs.")
    parser.add_argument("--cell_count", type=int, default=2048)
    parser.add_argument(
        "--sources",
        type=str,
        default="gt,redeconv,tape,b2space",
        help="Comma-separated: gt,redeconv,tape,b2space",
    )
    parser.add_argument(
        "--include_cancers",
        type=str,
        default="",
        help="Optional comma-separated cancers to include (default: all except COAD/BRCA).",
    )
    args = parser.parse_args()

    cfg = load_path_json(args.path_json)
    include = [c.strip() for c in args.include_cancers.split(",") if c.strip()] or None
    cancers = iter_cancers(cfg, include=include)
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]

    for cancer in cancers:
        cp = get_cancer_paths(cfg, cancer)
        for src in sources:
            ratio_csv = get_ratio_path(cancer, src, args.ratio_base)
            if not ratio_csv.exists():
                raise FileNotFoundError(f"Ratio file not found: {ratio_csv}")
            out_dir = str(Path(args.out_root) / cancer / src)
            run_one(args.base_dir, cp, str(ratio_csv), out_dir, args.cell_count)


if __name__ == "__main__":
    main()


