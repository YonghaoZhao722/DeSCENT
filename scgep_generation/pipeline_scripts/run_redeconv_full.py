#!/usr/bin/env python3
"""
Run ReDeconv 3-step workflow non-interactively.
Step 1: get_initial_Signature_Candidates (meta + exp) -> Initial_sig
Step 2: Get_signature_gene_matrix -> mean_std
Step 3: ReDeconv (mean_std + bulk) -> celltypes

Paths from config: redeconv_ref (dir with Meta_data_new.tsv, scRNA_seq_new_noShift.tsv),
                   bulk_tpm (filtered TPM TSV, genes x samples).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_SCGEP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCGEP_ROOT))

from redeconv.__ReDeconv_P import (
    Get_signature_gene_matrix,
    ReDeconv,
    check_meta_and_scRNAseq_data,
    get_initial_Signature_Candidates,
)


def run_redeconv_full(
    cancer: str,
    redeconv_ref: str,
    bulk_tpm: str,
    out_dir: Path,
    *,
    L_max_pv: float = 0.005,
    L_min_fold_change: float = 2.0,
    L_CellType_CellNo_LB: int = 30,
    L_NoSep_sampleNo_UB: int = 2,
    L_topNo: int = 70,
) -> None:
    """Run all 3 ReDeconv steps."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = Path(redeconv_ref)
    fn_meta = ref / "Meta_data_new.tsv"
    fn_exp = ref / "scRNA_seq_new_noShift.tsv"
    if not fn_meta.exists():
        raise FileNotFoundError(f"Meta not found: {fn_meta}")
    if not fn_exp.exists():
        raise FileNotFoundError(f"Expression not found: {fn_exp}")
    if not Path(bulk_tpm).exists():
        raise FileNotFoundError(f"Bulk TPM not found: {bulk_tpm}")

    fn_ini_sig = out_dir / f"{cancer}_Initial_sig_t_test_fd2.0_corr.tsv"
    fn_mean_std = out_dir / f"{cancer}_mean_std_FC2.0_TOP150.tsv"
    fn_heatmap = out_dir / f"{cancer}Heatmap_signature_gene_matrix{L_topNo}.png"
    fn_extra_info = out_dir / f"{cancer}_Signature_genes_extra_information.txt"
    fn_percentage_save = out_dir / f"{cancer}_celltypes.tsv"

    # Step 1
    print(f"[{cancer}] Step 1: Find initial signature genes...")
    status = check_meta_and_scRNAseq_data(str(fn_meta), str(fn_exp))
    if status <= 0:
        raise RuntimeError("Meta and expression data check failed.")
    get_initial_Signature_Candidates(
        str(fn_meta), str(fn_exp), str(fn_ini_sig),
        L_max_pv, L_min_fold_change, L_CellType_CellNo_LB, L_NoSep_sampleNo_UB,
    )
    print(f"  -> {fn_ini_sig}")

    # Step 2
    print(f"[{cancer}] Step 2: Compute mean/std of top signature genes...")
    Get_signature_gene_matrix(
        str(fn_exp), str(fn_meta), str(fn_ini_sig),
        str(fn_mean_std), L_topNo, str(fn_heatmap), str(fn_extra_info),
    )
    print(f"  -> {fn_mean_std}")

    # Step 3
    print(f"[{cancer}] Step 3: Cell type deconvolution...")
    ReDeconv(str(fn_mean_std), bulk_tpm, str(fn_percentage_save))
    print(f"  -> {fn_percentage_save}")

    # Convert to CSV for downstream (sample/cell_type, cell type columns)
    df = pd.read_csv(fn_percentage_save, sep="\t")
    if df.columns[0] != "sample/cell_type":
        df = df.rename(columns={df.columns[0]: "sample/cell_type"})
    fn_csv = out_dir / f"{cancer}_ratios_redeconv_10.csv"
    df.to_csv(fn_csv, index=False)
    print(f"  -> {fn_csv}")


def main():
    parser = argparse.ArgumentParser(description="Run ReDeconv 3-step workflow (reference + bulk -> fraction).")
    parser.add_argument("--cancer", type=str, required=True, help="Cancer type (e.g. BRCA)")
    parser.add_argument("--config", type=str, required=True, help="Path to path.json")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for mean_std, celltypes")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    cancer = args.cancer.upper()
    if cancer not in cfg:
        raise KeyError(f"Cancer {cancer} not in config")
    c = cfg[cancer]
    redeconv_ref = c.get("redeconv_ref")
    bulk_tpm = c.get("bulk_tpm")
    if not redeconv_ref or not bulk_tpm:
        raise ValueError(
            f"Config must have redeconv_ref and bulk_tpm for {cancer}. "
            "redeconv_ref: dir with Meta_data_new.tsv, scRNA_seq_new_noShift.tsv. "
            "bulk_tpm: filtered TPM TSV (genes x samples)."
        )
    run_redeconv_full(cancer, redeconv_ref, bulk_tpm, Path(args.out_dir))
    print(f"Done. Output in {args.out_dir}")


if __name__ == "__main__":
    main()
