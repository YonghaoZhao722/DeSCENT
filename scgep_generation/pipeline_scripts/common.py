from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


DEFAULT_EXCLUDE_CANCERS = {"COAD", "BRCA"}


@dataclass(frozen=True)
class CancerPaths:
    cancer: str
    sc_train_h5ad: str
    sc_test_h5ad: str
    vae_path: str
    diffusion_backbone: str
    classifier_path: str
    gene_list: str
    celltypes_csv: str
    gene_mapping: Optional[str] = None  # Optional: ENSG -> Symbol mapping file


def load_path_json(path_json: str) -> dict:
    with open(path_json, "r", encoding="utf-8") as f:
        return json.load(f)


def get_cancer_paths(cfg: dict, cancer: str) -> CancerPaths:
    item = cfg[cancer]
    return CancerPaths(
        cancer=item["cancer"],
        sc_train_h5ad=item["single_cell_data"]["train"],
        sc_test_h5ad=item["single_cell_data"]["test"],
        vae_path=item["VAE"],
        diffusion_backbone=item["diffusion_backbone"],
        classifier_path=item["classifier"],
        gene_list=item["gene_list"],
        celltypes_csv=item["celltypes"],
        gene_mapping=item.get("gene_mapping"),  # Optional field
    )


def iter_cancers(cfg: dict, include: Optional[Iterable[str]] = None) -> list[str]:
    cancers = list(cfg.keys())
    if include is not None:
        include_set = {c.upper() for c in include}
        cancers = [c for c in cancers if c.upper() in include_set]
        return cancers  # when explicitly included, do not apply default exclude
    cancers = [c for c in cancers if c.upper() not in DEFAULT_EXCLUDE_CANCERS]
    return cancers


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def strip_gene_suffix(gene_names: Iterable[str]) -> list[str]:
    # Many h5ad gene IDs can be like "ENSG... .1" etc.
    return [str(g).split(".")[0] for g in gene_names]


def read_gene_list(gene_list_path: str) -> list[str]:
    """
    Read gene order list from CSV/TSV/txt.

    In this repo, many gene list files are CSV with the 2nd column being gene IDs/symbols.
    """
    p = Path(gene_list_path)
    if not p.exists():
        raise FileNotFoundError(f"Gene list file not found: {gene_list_path}")

    if p.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if p.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(gene_list_path, sep=sep)
        if df.shape[1] == 1:
            genes = df.iloc[:, 0].astype(str).tolist()
        else:
            genes = df.iloc[:, 1].astype(str).tolist()
        return strip_gene_suffix(genes)

    # Fallback: one gene per line
    with open(gene_list_path, "r", encoding="utf-8") as f:
        genes = [line.strip() for line in f if line.strip()]
    return strip_gene_suffix(genes)


def read_gene_lengths(gene_length_file: str) -> pd.Series:
    """
    Return a Series mapping gene symbol -> feature_length (bp).
    Accepts files with columns like:
      gene_symbol,feature_length
    or other 2-column TSVs.
    Also supports human_gene_length.txt format: ENSG_ID length (space-separated, no header)
    """
    # Try reading as space-separated first (for human_gene_length.txt format)
    try:
        df = pd.read_csv(gene_length_file, sep=r'\s+', header=None, names=['gene', 'length'], engine='python')
        # If successful and has 2 columns, assume it's ENSG format
        if df.shape[1] == 2:
            s = pd.Series(df['length'].values, index=df['gene'].astype(str))
            s = s[~s.index.duplicated(keep="first")]
            return s
    except Exception:
        pass
    
    # Fallback to standard CSV/TSV parsing
    df = pd.read_csv(gene_length_file, sep=None, engine="python")
    cols = [c.lower() for c in df.columns]

    if "feature_length" not in cols and "length" not in cols:
        # If no length column found, assume second column is length
        if df.shape[1] >= 2:
            s = pd.Series(df.iloc[:, 1].values, index=df.iloc[:, 0].astype(str))
            s = s[~s.index.duplicated(keep="first")]
            return s
        raise ValueError(f"gene_length_file missing 'feature_length' or 'length' column: {gene_length_file}")

    # Try common gene column names
    gene_col = None
    for cand in ["gene_symbol", "gene name", "gene", "symbol"]:
        if cand in cols:
            gene_col = df.columns[cols.index(cand)]
            break
    if gene_col is None:
        # Fallback: first column
        gene_col = df.columns[0]

    length_col = df.columns[cols.index("feature_length")] if "feature_length" in cols else df.columns[cols.index("length")]
    s = pd.Series(df[length_col].values, index=df[gene_col].astype(str))
    s = s[~s.index.duplicated(keep="first")]
    return s


def normalize_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2 = df2.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df2[df2 < 0] = 0.0
    row_sums = df2.sum(axis=1).replace(0, np.nan)
    df2 = df2.div(row_sums, axis=0).fillna(0.0)
    return df2


def save_ratios_csv(df: pd.DataFrame, out_csv: str) -> None:
    """
    Save ratios in the format expected by generate_bulk_from_diffusion.py:
      - first column named 'sample/cell_type'
      - remaining columns are cell types
      - each row sums to 1 (recommended; not enforced here)
    """
    ensure_dir(str(Path(out_csv).parent))
    df_out = df.copy()
    df_out.index = df_out.index.astype(str)
    df_out.to_csv(out_csv, index=True, index_label="sample/cell_type")


def load_ratios_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "sample/cell_type" not in df.columns:
        raise ValueError(f"Ratio file must contain 'sample/cell_type' column: {path}")
    df = df.set_index("sample/cell_type")
    df.index = df.index.astype(str)
    return df


def infer_num_class_from_ratio_file(ratio_csv: str) -> int:
    df = load_ratios_csv(ratio_csv)
    return int(df.shape[1])


