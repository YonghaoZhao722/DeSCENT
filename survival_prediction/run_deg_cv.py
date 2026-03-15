#!/usr/bin/env python3
"""
DEG using PyDESeq2 (tumor vs normal).
- Supports: tcga_umi (single CSV per cancer), bulk_recount3_dir, bulk_tpm.
- Matched with surv_label = tumor. Remaining samples 5-fold split, no leakage.
- Output: degs_fold1.csv ... degs_fold5.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from sklearn.model_selection import KFold


def _infer_sample_type_from_barcode(sample_ids: pd.Index) -> pd.Series:
    """
    Infer tumor (1) vs normal (0) from sample barcode.
    GTEx -> 0; TCGA positions 13-14 < 10 (01,02) -> 1; >= 10 (11) -> 0.
    """
    idx = sample_ids
    sample_ids = pd.Series(sample_ids.tolist(), index=idx).astype(str)
    study = sample_ids.str.slice(0, 4)
    sample_code = sample_ids.str.slice(13, 15)
    sample_code = sample_code.replace("-", "10")
    sample_code = pd.to_numeric(sample_code, errors="coerce").fillna(10).astype(int)

    out = pd.Series(0, index=sample_ids.index)
    out[study == "GTEX"] = 0
    out[(study == "TCGA") & (sample_code < 10)] = 1
    out[(study == "TCGA") & (sample_code >= 10)] = 0
    return out


def _map_to_bulk_ids(candidate_ids: list, bulk_index: pd.Index) -> list:
    """Map surv_label IDs to bulk sample IDs (exact or patient prefix + tumor 01 match)."""
    bulk_list = [str(x) for x in bulk_index]
    mapped = []
    for cid in candidate_ids:
        cid = str(cid)
        if cid in bulk_list:
            mapped.append(cid)
            continue
        patient = cid[:12] if len(cid) >= 12 else cid
        candidates = [b for b in bulk_list if b.startswith(patient) or patient in b]
        tumor_candidates = [b for b in candidates if len(b) > 14 and b[13:15] in ("01", "02", "03")]
        if tumor_candidates:
            mapped.append(tumor_candidates[0])
        elif candidates:
            mapped.append(candidates[0])
    return list(dict.fromkeys(mapped))


def _load_gene_list(gene_list_path: str) -> set:
    """Load gene list. Auto-detect ENSG or symbol column."""
    df = pd.read_csv(gene_list_path, header=0)
    ensg_col = None
    symbol_col = None
    for c in df.columns:
        s = df[c].dropna().astype(str)
        if len(s) == 0:
            continue
        sample = s.iloc[0]
        if sample.startswith("ENSG"):
            ensg_col = c
            break
        if sample.isupper() or (sample.replace("-", "").replace(".", "").isalnum() and len(sample) > 2):
            symbol_col = c
    col = ensg_col if ensg_col is not None else symbol_col
    if col is None:
        col = df.columns[0]
    return set(df[col].dropna().astype(str).unique())


def _load_tcga_umi_csv(
    umi_path: str,
    gene_list_path: str | None = None,
    is_log2: bool = False,
) -> pd.DataFrame:
    """
    Load tcga_umi single CSV (genes x samples, columns = TCGA/GTEx barcode).
    Return samples x genes (raw counts).
    If is_log2: values are log2(umi+1), back-transform to raw before returning.
    """
    counts_df = pd.read_csv(umi_path, index_col=0)
    if is_log2:
        # Back-transform log2(umi+1) -> raw: raw = 2^x - 1
        counts_df = np.exp2(counts_df.fillna(0).astype(float)) - 1
        counts_df = counts_df.clip(lower=0)
    if gene_list_path and Path(gene_list_path).exists():
        valid_genes = _load_gene_list(gene_list_path)
        def _base_id(g):
            return str(g).split(".")[0] if "." in str(g) else str(g)
        keep_genes = [g for g in counts_df.index if _base_id(g) in valid_genes or str(g) in valid_genes]
        if keep_genes:
            counts_df = counts_df.loc[keep_genes]
        else:
            # gene_list has wrong format (e.g. symbol vs ENSG); skip filter
            import warnings
            warnings.warn(
                f"Gene list matched 0 genes (likely symbol vs ENSG mismatch). Using all {len(counts_df.index)} genes.",
                UserWarning,
            )
    bulk_df = counts_df.T
    bulk_df.index = bulk_df.index.astype(str)
    return bulk_df


def _load_recount3_counts(
    gene_counts_path: str,
    metadata_path: str,
    gene_list_path: str | None = None,
) -> pd.DataFrame:
    """Load recount3. Return samples x genes."""
    counts_df = pd.read_csv(gene_counts_path, index_col=0)
    meta_df = pd.read_csv(metadata_path)
    barcode_col = "tcga.tcga_barcode" if "tcga.tcga_barcode" in meta_df.columns else "tcga_barcode"
    id_col = "external_id" if "external_id" in meta_df.columns else "rail_id"
    id_to_barcode = dict(zip(meta_df[id_col].astype(str), meta_df[barcode_col].astype(str)))
    valid_cols = [c for c in counts_df.columns if str(c) in id_to_barcode]
    counts_df = counts_df[valid_cols].rename(columns=id_to_barcode)
    if gene_list_path and Path(gene_list_path).exists():
        valid_genes = _load_gene_list(gene_list_path)
        def _base_id(g):
            return str(g).split(".")[0] if "." in str(g) else str(g)
        keep_genes = [g for g in counts_df.index if _base_id(g) in valid_genes or str(g) in valid_genes]
        counts_df = counts_df.loc[keep_genes]
    bulk_df = counts_df.T
    bulk_df.index = bulk_df.index.astype(str)
    return bulk_df


def run_deg_tumor_vs_normal(
    surv_label_dir: str,
    num_folds: int,
    out_dir: str,
    bulk_df: pd.DataFrame,
    padj_thresh: float = 0.05,
    l2fc_thresh: float = 1.5,
    id_map_species: str = "human",
    random_state: int = 42,
) -> None:
    """
    DEG tumor vs normal. bulk_df: samples x genes.
    Matched with surv_label = tumor. Remaining samples 5-fold split (no leakage).
    """
    bulk_samples = bulk_df.index.astype(str).tolist()

    surv_dir = Path(surv_label_dir)
    tumor_per_fold = []
    for k in range(1, num_folds + 1):
        surv_path = surv_dir / f"train_data_{k}.csv"
        if not surv_path.exists():
            raise FileNotFoundError(f"Survival labels not found: {surv_path}")
        surv_df = pd.read_csv(surv_path)
        id_col = surv_df.columns[0]
        train_ids = surv_df[id_col].astype(str).tolist()
        mapped = _map_to_bulk_ids(train_ids, bulk_df.index)
        tumor_per_fold.append(set(mapped))

    all_tumor_ids = set()
    for s in tumor_per_fold:
        all_tumor_ids.update(s)
    remaining_ids = [x for x in bulk_samples if x not in all_tumor_ids]
    sample_type = _infer_sample_type_from_barcode(pd.Index(remaining_ids))
    remaining_normal = [remaining_ids[i] for i in range(len(remaining_ids)) if sample_type.iloc[i] == 0]
    remaining_tumor = [remaining_ids[i] for i in range(len(remaining_ids)) if sample_type.iloc[i] == 1]

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    remaining_all = remaining_normal + remaining_tumor
    remaining_arr = np.array(remaining_all)
    if len(remaining_arr) == 0:
        raise ValueError("No remaining samples (normal or unmatched tumor). Need tumor+normal.")
    fold_indices = list(kf.split(remaining_arr))

    for fold_num in range(1, num_folds + 1):
        train_idx, _ = fold_indices[fold_num - 1]
        remaining_train_k = remaining_arr[train_idx].tolist()
        tumor_train_k = list(tumor_per_fold[fold_num - 1])
        deg_sample_ids = tumor_train_k + remaining_train_k

        expr_df = bulk_df.loc[bulk_df.index.isin(deg_sample_ids)]
        expr_df = expr_df[~expr_df.index.duplicated(keep="first")]
        sample_type_fold = _infer_sample_type_from_barcode(expr_df.index)

        meta_df = sample_type_fold.to_frame(name="group")
        meta_df["group"] = meta_df["group"].astype(int)

        n_tumor = (meta_df["group"] == 1).sum()
        n_normal = (meta_df["group"] == 0).sum()
        if n_tumor == 0 or n_normal == 0:
            raise ValueError(
                f"Fold {fold_num}: need both tumor and normal. Found tumor={n_tumor}, normal={n_normal}."
            )

        counts_df = expr_df.select_dtypes(include=["number"])
        if counts_df.shape[1] == 0:
            raise ValueError(
                f"Fold {fold_num}: no numeric gene columns. "
                "Check gene_list matches UMI format (ENSG required for tcga_umi)."
            )
        counts_df = counts_df.fillna(0).astype(int)
        if (counts_df < 0).any().any():
            counts_df = counts_df.clip(lower=0)

        common_idx = counts_df.index.intersection(meta_df.index)
        counts_df = counts_df.loc[common_idx]
        meta_df = meta_df.loc[common_idx]

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
        print(f"  Saved: {out_path} (tumor={n_tumor}, normal={n_normal}, DEGs={len(res_filtered)})")


def main():
    parser = argparse.ArgumentParser(description="DEG tumor vs normal (5-fold CV)")
    parser.add_argument("--cancer", type=str, default=None, help="Cancer type (e.g. BRCA, COAD)")
    parser.add_argument("--config", type=str, default="config/path_local.json")
    parser.add_argument("--bulk_umi_csv", type=str, default=None,
                        help="Single UMI CSV (genes x samples). Overrides tcga_umi_dir.")
    parser.add_argument("--log2_umi", action="store_true",
                        help="Values are log2(umi+1); back-transform to raw before DESeq2. Use for LGG lgg_gtex_log2.csv.")
    parser.add_argument("--tcga_umi_dir", type=str, default="/data/youzy/tcga_umi",
                        help="Dir with {cancer}.csv. Used when --cancer set and --bulk_umi_csv not set.")
    parser.add_argument("--bulk_recount3_dir", type=str, default=None)
    parser.add_argument("--bulk_tpm", type=str, default=None)
    parser.add_argument("--surv_label_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--gene_list_path", type=str, default=None)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--padj_thresh", type=float, default=0.05)
    parser.add_argument("--l2fc_thresh", type=float, default=1.5)
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        import json
        config = json.load(open(config_path))
    else:
        config = {}

    cancer = args.cancer or (list(config.keys())[0] if config else "BRCA")
    c = config.get(cancer, {}) if cancer else {}

    surv_label_dir = args.surv_label_dir or c.get("surv_label")
    out_dir = args.out_dir
    if not out_dir:
        descent_root = config_path.resolve().parent.parent
        out_dir = str(descent_root / "output" / "deg_cv" / cancer)
    gene_list_path = args.gene_list_path or c.get("gene_list_path") or c.get("gene_list")

    if not surv_label_dir or not Path(surv_label_dir).exists():
        raise FileNotFoundError(f"surv_label_dir not found: {surv_label_dir}")

    # Load bulk data
    bulk_df = None
    log2_umi = args.log2_umi
    if args.bulk_umi_csv and Path(args.bulk_umi_csv).exists():
        print(f"Loading UMI from {args.bulk_umi_csv}" + (" (log2->raw)" if log2_umi else ""))
        bulk_df = _load_tcga_umi_csv(args.bulk_umi_csv, gene_list_path, is_log2=log2_umi)
    elif args.tcga_umi_dir and cancer:
        # LGG: use lgg_gtex_log2.csv (TCGA+GTEx, log2) instead of lgg.csv (no normal)
        if cancer.upper() == "LGG":
            umi_path = Path(args.tcga_umi_dir) / "lgg_gtex_log2.csv"
            log2_umi = True
        else:
            umi_path = Path(args.tcga_umi_dir) / f"{cancer.lower()}.csv"
        if umi_path.exists():
            print(f"Loading UMI from {umi_path}" + (" (log2->raw)" if log2_umi else ""))
            bulk_df = _load_tcga_umi_csv(str(umi_path), gene_list_path, is_log2=log2_umi)
    if args.bulk_recount3_dir:
        r3_dir = Path(args.bulk_recount3_dir)
        if not gene_list_path or not Path(gene_list_path).exists():
            raise FileNotFoundError("bulk_recount3_dir requires --gene_list_path")
        gene_files = list(r3_dir.glob("*_gene_counts.csv"))
        meta_files = list(r3_dir.glob("*_sample_metadata.csv"))
        if gene_files and meta_files:
            bulk_df = _load_recount3_counts(str(gene_files[0]), str(meta_files[0]), gene_list_path)
    if args.bulk_tpm and Path(args.bulk_tpm).exists():
        path = Path(args.bulk_tpm)
        sep = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","
        bulk_df = pd.read_csv(args.bulk_tpm, index_col=0, sep=sep)
        if bulk_df.shape[0] > bulk_df.shape[1]:
            bulk_df = bulk_df.T
        if gene_list_path and Path(gene_list_path).exists():
            valid_genes = _load_gene_list(gene_list_path)
            bulk_df = bulk_df.loc[:, bulk_df.columns.isin(valid_genes)]

    if bulk_df is None:
        raise ValueError(
            "Provide one of: --bulk_umi_csv, --tcga_umi_dir + --cancer, --bulk_recount3_dir, --bulk_tpm"
        )

    print(f"Running DEG ({cancer or 'unknown'}): {bulk_df.shape[0]} samples x {bulk_df.shape[1]} genes")
    run_deg_tumor_vs_normal(
        surv_label_dir=surv_label_dir,
        num_folds=args.num_folds,
        out_dir=out_dir,
        bulk_df=bulk_df,
        padj_thresh=args.padj_thresh,
        l2fc_thresh=args.l2fc_thresh,
    )
    print(f"  -> {out_dir}/degs_fold{{1..{args.num_folds}}}.csv")


if __name__ == "__main__":
    main()
