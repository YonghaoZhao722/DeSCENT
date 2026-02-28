from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path so we can import pipeline_scripts
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse

from pipeline_scripts.common import (
    CancerPaths,
    ensure_dir,
    get_cancer_paths,
    iter_cancers,
    load_path_json,
    normalize_ratios,
    read_gene_lengths,
    read_gene_list,
    save_ratios_csv,
    strip_gene_suffix,
)


def _detect_celltype_key(adata) -> str:
    # Prefer repo conventions first.
    for k in ["Cell_type", "celltype", "cell_type", "Cell_Type", "cell_type_final"]:
        if k in adata.obs.columns:
            return k
    raise ValueError(f"Cannot find cell type column in adata.obs. Available: {adata.obs.columns.tolist()}")


def _get_raw_counts_matrix(adata):
    # Prefer raw counts if present; otherwise use X.
    if getattr(adata, "raw", None) is not None and getattr(adata.raw, "X", None) is not None:
        X = adata.raw.X
        genes = adata.raw.var_names
        return X, genes
    return adata.X, adata.var_names


def counts_to_tpm(counts: np.ndarray, gene_lengths_bp: np.ndarray) -> np.ndarray:
    # counts shape: (n_genes,)
    # gene_lengths_bp shape: (n_genes,)
    gene_lengths_bp = gene_lengths_bp.astype(float)
    gene_lengths_bp[~np.isfinite(gene_lengths_bp)] = np.nan
    gene_lengths_bp[gene_lengths_bp <= 0] = np.nan
    if np.all(np.isnan(gene_lengths_bp)):
        raise ValueError("All gene lengths are missing/invalid; cannot compute TPM.")
    fill_val = float(np.nanmedian(gene_lengths_bp))
    gene_lengths_bp = np.nan_to_num(gene_lengths_bp, nan=fill_val)

    rpk = counts * 1000.0 / gene_lengths_bp
    denom = rpk.sum()
    if denom <= 0:
        return np.zeros_like(rpk, dtype=float)
    return rpk / denom * 1e6


def _sample_indices_for_celltype(
    cell_indices: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(cell_indices) == 0 or n <= 0:
        return np.array([], dtype=int)
    replace = len(cell_indices) < n
    return rng.choice(cell_indices, size=n, replace=replace)


def process_one_cancer(
    cancer: str,
    cfg: dict,
    args,
    out_root: Path,
) -> None:
    """Process a single cancer."""
    cp: CancerPaths = get_cancer_paths(cfg, cancer)
    
    h5ad_path = cp.sc_train_h5ad if args.use_split == "train" else cp.sc_test_h5ad
    if not Path(h5ad_path).exists():
        raise FileNotFoundError(f"h5ad file not found: {h5ad_path}")

    out_dir = out_root / cancer
    ensure_dir(str(out_dir))

    gene_order = read_gene_list(cp.gene_list)
    
    # Load gene lengths (may be ENSG format)
    gene_lengths_raw = read_gene_lengths(args.gene_length_file)
    
    # If gene_mapping exists, convert ENSG lengths to Symbol lengths
    if cp.gene_mapping and Path(cp.gene_mapping).exists():
        print(f"  Loading gene mapping for length conversion: {cp.gene_mapping}")
        mapping_df = pd.read_csv(cp.gene_mapping)
        # Build ENSG -> Symbol mapping (column 1 is ENSG, column 2 is Symbol)
        ensg_to_symbol = {}
        for _, row in mapping_df.iterrows():
            ensg = str(row[mapping_df.columns[1]]).strip()
            symbol = str(row[mapping_df.columns[2]]).strip()
            ensg_clean = strip_gene_suffix([ensg])[0]
            ensg_to_symbol[ensg_clean] = symbol
        
        # Convert gene_lengths from ENSG index to Symbol index
        gene_lengths_symbol = pd.Series(dtype=float)
        for ensg, length in gene_lengths_raw.items():
            ensg_clean = strip_gene_suffix([ensg])[0]
            if ensg_clean in ensg_to_symbol:
                symbol = ensg_to_symbol[ensg_clean]
                # If multiple ENSG map to same symbol, take the max length
                if symbol in gene_lengths_symbol.index:
                    gene_lengths_symbol[symbol] = max(gene_lengths_symbol[symbol], length)
                else:
                    gene_lengths_symbol[symbol] = length
        
        gene_lengths = gene_lengths_symbol
        print(f"  Converted {len(gene_lengths)} gene lengths from ENSG to Symbol")
    else:
        gene_lengths = gene_lengths_raw

    rng = np.random.default_rng(args.seed)

    adata = sc.read_h5ad(h5ad_path)
    celltype_key = args.celltype_key or _detect_celltype_key(adata)

    # Get raw counts matrix
    X, genes_raw = _get_raw_counts_matrix(adata)
    
    # Convert single-cell gene names from numeric index to Symbol if mapping file exists
    if cp.gene_mapping and Path(cp.gene_mapping).exists():
        print(f"  Loading gene mapping for SC gene conversion: {cp.gene_mapping}")
        mapping_df = pd.read_csv(cp.gene_mapping)
        # Build index -> Symbol mapping (column 0 is index, column 2 is Symbol)
        index_to_symbol = {}
        for _, row in mapping_df.iterrows():
            idx = str(row[mapping_df.columns[0]]).strip()
            symbol = str(row[mapping_df.columns[2]]).strip()
            index_to_symbol[idx] = symbol
        
        # Convert genes_raw (numeric indices) to Symbols
        genes_symbol = []
        for g in genes_raw:
            g_str = str(g).strip()
            g_clean = strip_gene_suffix([g_str])[0]
            if g_clean in index_to_symbol:
                genes_symbol.append(index_to_symbol[g_clean])
            else:
                genes_symbol.append(g_clean)  # Keep original if no mapping
        
        genes = pd.Index(genes_symbol)
        print(f"  Converted SC gene names: {len(genes)} genes")
        print(f"  Sample SC genes after conversion: {list(genes[:5])}")
        print(f"  Sample gene_order: {list(gene_order[:5])}")
    else:
        # Standardize gene names to symbols without suffix.
        genes = pd.Index(strip_gene_suffix(genes_raw))
        print(f"  No gene mapping file found, using SC genes as-is")

    # Build mapping from gene name (Symbol) -> column index in X (gene dimension).
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    n_genes_sc = len(genes)
    
    # Verify gene matching
    sc_genes_set = set(genes)
    gene_order_set = set(gene_order)
    common_genes = sc_genes_set & gene_order_set
    print(f"  Common genes between SC and gene_order: {len(common_genes)}/{len(gene_order)}")
    if len(common_genes) < len(gene_order) * 0.5:
        print(f"  WARNING: Only {len(common_genes)}/{len(gene_order)} genes match!")
        print(f"    Sample SC genes: {list(sc_genes_set)[:10]}")
        print(f"    Sample gene_order: {list(gene_order_set)[:10]}")
        print(f"    Sample common: {list(common_genes)[:10]}")

    # Index cells by cell type.
    celltypes = adata.obs[celltype_key].astype(str).values
    unique_celltypes = sorted(pd.unique(celltypes).tolist())

    celltype_to_cell_indices: dict[str, np.ndarray] = {}
    for ct in unique_celltypes:
        celltype_to_cell_indices[ct] = np.where(celltypes == ct)[0]

    # Empirical prior (used to shape Dirichlet).
    prior = pd.Series(celltypes).value_counts(normalize=True).reindex(unique_celltypes, fill_value=0.0).values
    alpha = np.maximum(prior * float(args.alpha_scale), 1e-3)

    bulk_tpm_cols: dict[str, np.ndarray] = {}
    gt_ratios_rows: dict[str, dict[str, float]] = {}

    for i in range(args.n_samples):
        sample_id = f"{cancer}_pseudo_{i:02d}"
        ratios = rng.dirichlet(alpha)
        ratios = ratios / ratios.sum()

        # Determine cells per type.
        n_per_type = np.floor(ratios * args.total_cells).astype(int)
        # Fix rounding to sum exactly.
        remainder = int(args.total_cells - n_per_type.sum())
        if remainder > 0:
            # Add remaining cells to the largest ratios.
            for idx in np.argsort(-ratios)[:remainder]:
                n_per_type[idx] += 1

        sampled_cell_indices = []
        sampled_labels = []
        for ct, n_ct in zip(unique_celltypes, n_per_type):
            idxs = _sample_indices_for_celltype(celltype_to_cell_indices[ct], int(n_ct), rng)
            if len(idxs) == 0:
                continue
            sampled_cell_indices.append(idxs)
            sampled_labels.extend([ct] * len(idxs))

        if len(sampled_cell_indices) == 0:
            raise RuntimeError("No cells sampled. Check cell types and ratios.")

        sampled_cell_indices = np.concatenate(sampled_cell_indices, axis=0)

        # Compute gene counts by summing sampled cells.
        if issparse(X):
            summed = np.asarray(X[sampled_cell_indices].sum(axis=0)).ravel()
        else:
            summed = np.asarray(X[sampled_cell_indices].sum(axis=0)).ravel()
        if summed.shape[0] != n_genes_sc:
            raise ValueError(f"Unexpected gene dimension: {summed.shape[0]} vs {n_genes_sc}")

        # Reindex to gene_order (fill missing genes with zeros).
        counts_vec = np.zeros(len(gene_order), dtype=float)
        matched_count = 0
        for j, g in enumerate(gene_order):
            idx = gene_to_idx.get(g)
            if idx is not None:
                counts_vec[j] = float(summed[idx])
                matched_count += 1
        
        if matched_count == 0:
            print(f"  ERROR: No genes matched between gene_order and SC genes!")
            print(f"    Sample gene_order: {list(gene_order[:5])}")
            print(f"    Sample SC genes: {list(genes[:5])}")
            print(f"    Sample gene_to_idx keys: {list(gene_to_idx.keys())[:5]}")
            raise ValueError(f"No genes matched for sample {sample_id}")
        
        if matched_count < len(gene_order) * 0.5:
            print(f"  Warning: Only {matched_count}/{len(gene_order)} genes matched for sample {sample_id}")
        
        if counts_vec.sum() == 0:
            print(f"  ERROR: Counts sum is 0 for sample {sample_id}!")
            print(f"    Matched genes: {matched_count}/{len(gene_order)}")
            print(f"    Original summed shape: {summed.shape}, sum: {summed.sum()}, max: {summed.max()}")
            raise ValueError(f"Counts are all zero for sample {sample_id}")

        # Get gene lengths for gene_order, handling missing genes
        lengths_vec = gene_lengths.reindex(gene_order).values
        
        # Check for missing lengths and fill with median
        missing_mask = np.isnan(lengths_vec)
        if missing_mask.any():
            median_length = np.nanmedian(lengths_vec[~missing_mask])
            if np.isnan(median_length) or median_length <= 0:
                median_length = 1000.0  # Fallback default
            lengths_vec[missing_mask] = median_length
            print(f"  Warning: {missing_mask.sum()} genes missing length info, using median={median_length:.1f}bp")
        
        # Ensure all lengths are positive
        lengths_vec[lengths_vec <= 0] = np.nanmedian(lengths_vec[lengths_vec > 0])
        
        # Compute TPM
        tpm_vec = counts_to_tpm(counts_vec, lengths_vec)
        
        # Debug: check if TPM is all zeros
        if np.all(tpm_vec == 0):
            print(f"  WARNING: TPM vector is all zeros for sample {sample_id}")
            print(f"    Counts sum: {counts_vec.sum()}, max: {counts_vec.max()}")
            print(f"    Lengths range: [{lengths_vec.min():.1f}, {lengths_vec.max():.1f}]")

        bulk_tpm_cols[sample_id] = tpm_vec

        gt_row = {ct: float(r) for ct, r in zip(unique_celltypes, ratios)}
        gt_ratios_rows[sample_id] = gt_row

    bulk_tpm = pd.DataFrame(bulk_tpm_cols, index=gene_order)
    bulk_tpm.index.name = ""  # match filtered_tpm_*.tsv (blank top-left header cell)
    bulk_tpm_path = out_dir / f"{cancer}_pseudobulk_10_TPM.tsv"
    bulk_tpm.to_csv(bulk_tpm_path, sep="\t")

    gt_ratios = pd.DataFrame.from_dict(gt_ratios_rows, orient="index")
    gt_ratios = normalize_ratios(gt_ratios)
    gt_ratios_path = out_dir / f"{cancer}_ratios_gt_10.csv"
    save_ratios_csv(gt_ratios, str(gt_ratios_path))

    print(f"Saved pseudobulk TPM: {bulk_tpm_path}")
    print(f"Saved ground-truth ratios: {gt_ratios_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Randomly sample single-cell data to generate 10 pseudobulks (TPM) and ground-truth ratios."
    )
    parser.add_argument("--path_json", type=str, default="/data/zhaoyh/scDiffusion-main/path.json")
    parser.add_argument("--cancer", type=str, default=None, help="Cancer type (e.g. LUAD). If omitted, process all cancers.")
    parser.add_argument(
        "--include_cancers",
        type=str,
        default="",
        help="Optional comma-separated cancers to include (default: all except COAD/BRCA).",
    )
    parser.add_argument(
        "--use_split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Which h5ad path from path.json to use.",
    )
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--total_cells", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--alpha_scale",
        type=float,
        default=50.0,
        help="Dirichlet concentration. Higher -> ratios closer to empirical celltype prior.",
    )
    parser.add_argument(
        "--gene_length_file",
        type=str,
        default="/data/zhaoyh/scDiffusion-main/human_gene_length.txt",
        help="Gene length file used to compute TPM (ENSG format: ENSG_ID length).",
    )
    parser.add_argument(
        "--celltype_key",
        type=str,
        default=None,
        help="Override cell type column key (auto-detected if omitted).",
    )
    args = parser.parse_args()

    cfg = load_path_json(args.path_json)
    
    # Determine which cancers to process
    if args.cancer:
        cancers = [args.cancer.upper()]
    else:
        include = [c.strip() for c in args.include_cancers.split(",") if c.strip()] or None
        cancers = iter_cancers(cfg, include=include)
    
    out_root = Path(args.out_dir)
    ensure_dir(str(out_root))
    
    print(f"Processing {len(cancers)} cancer(s): {', '.join(cancers)}")
    
    for cancer in cancers:
        print(f"\n{'='*60}")
        print(f"Processing {cancer}")
        print(f"{'='*60}")
        try:
            process_one_cancer(cancer, cfg, args, out_root)
        except Exception as e:
            print(f"ERROR processing {cancer}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"Completed processing {len(cancers)} cancer(s)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()


