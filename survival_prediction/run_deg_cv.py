#!/usr/bin/env python3
"""
Per-fold DEG using PyDESeq2 for nested CV.
For each fold k, reads train sample IDs from surv_label_dir/train_data_{k}.csv,
subsets bulk expression to train samples only, runs PyDESeq2, outputs degs_fold{k}.csv.
"""
import argparse
from pathlib import Path

import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats


def run_deg_one_fold(
    fold_num: int,
    bulk_dir: str,
    surv_label_dir: str,
    out_dir: str,
    padj_thresh: float = 0.05,
    l2fc_thresh: float = 1.5,
    contrast_col: str = "OS",
    id_map_species: str = "human",
) -> str:
    """Run DEG for one fold using train samples only. Returns path to output CSV."""
    surv_path = Path(surv_label_dir) / f"train_data_{fold_num}.csv"
    if not surv_path.exists():
        raise FileNotFoundError(f"Survival labels not found: {surv_path}")
    surv_df = pd.read_csv(surv_path)
    id_col = surv_df.columns[0]
    train_ids = surv_df[id_col].astype(str).tolist()

    bulk_path = Path(bulk_dir) / f"train_data_{fold_num}.csv"
    if not bulk_path.exists():
        raise FileNotFoundError(f"Bulk expression not found: {bulk_path}")
    expr_df = pd.read_csv(bulk_path, index_col=0)
    expr_df.index = expr_df.index.astype(str)
    # Detect orientation: if index overlaps with train_ids, rows=samples; else rows=genes -> transpose
    common_ids = [i for i in train_ids if i in expr_df.index]
    if len(common_ids) == 0 and expr_df.shape[0] < expr_df.shape[1]:
        expr_df = expr_df.T
        expr_df.index = expr_df.index.astype(str)
        common_ids = [i for i in train_ids if i in expr_df.index]
    if len(common_ids) == 0:
        for tid in train_ids:
            for idx in expr_df.index:
                if tid in idx or idx in tid:
                    common_ids.append(idx)
                    break
    expr_df = expr_df.loc[common_ids]

    surv_df[id_col] = surv_df[id_col].astype(str)
    surv_sub = surv_df[surv_df[id_col].isin(common_ids)].set_index(id_col)
    expr_df = expr_df.loc[expr_df.index.intersection(surv_sub.index)]
    meta_df = surv_sub.loc[expr_df.index, [contrast_col]].copy()
    meta_df.columns = ["group"]

    counts_df = expr_df.drop(columns=[c for c in expr_df.columns if c in meta_df.columns], errors="ignore")
    counts_df = counts_df.select_dtypes(include=["number"])
    counts_df = counts_df.fillna(0).astype(int)
    if (counts_df < 0).any().any():
        counts_df = counts_df.clip(lower=0)

    dds = DeseqDataSet(counts=counts_df, metadata=meta_df, design_factors=["group"])
    dds.deseq2()
    ds = DeseqStats(dds, contrast=["group", 0, 1])
    ds.summary()
    res = pd.DataFrame(ds.results_df)
    res = res.sort_values(by=["padj", "log2FoldChange"], ascending=[True, False])

    try:
        from sanbomics.tools import id_map
        mapper = id_map(species=id_map_species)
        res["symbol"] = res.index.map(mapper.mapper)
        mask = res["symbol"].isna()
        res.loc[mask, "symbol"] = res.index[mask]
    except Exception:
        res["symbol"] = res.index

    res_filtered = res[["baseMean", "log2FoldChange", "pvalue", "padj", "symbol"]].copy()
    res_filtered = res_filtered[res_filtered["pvalue"] < padj_thresh]
    res_filtered = res_filtered[res_filtered["padj"] < padj_thresh]
    res_filtered = res_filtered[res_filtered["log2FoldChange"].abs() > l2fc_thresh]

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / f"degs_fold{fold_num}.csv"
    res_filtered.to_csv(out_path)
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="Per-fold DEG for nested CV")
    parser.add_argument("--bulk_dir", type=str, required=True)
    parser.add_argument("--surv_label_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--padj_thresh", type=float, default=0.05)
    parser.add_argument("--l2fc_thresh", type=float, default=1.5)
    parser.add_argument("--contrast_col", type=str, default="OS")
    parser.add_argument("--id_map_species", type=str, default="human")
    args = parser.parse_args()

    for k in range(1, args.num_folds + 1):
        print(f"Running DEG for fold {k}...")
        out_path = run_deg_one_fold(
            fold_num=k,
            bulk_dir=args.bulk_dir,
            surv_label_dir=args.surv_label_dir,
            out_dir=args.out_dir,
            padj_thresh=args.padj_thresh,
            l2fc_thresh=args.l2fc_thresh,
            contrast_col=args.contrast_col,
            id_map_species=args.id_map_species,
        )
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
