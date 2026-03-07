from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
_SCGEP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCGEP_ROOT))

import pandas as pd

from pipeline_scripts.common import ensure_dir, get_cancer_paths, iter_cancers, load_path_json


def process_one_cancer(
    cancer: str,
    bulk_tpm_tsv: str,
    out_dir: Path,
    mean_std_tsv: str = None,
    mean_std_dir: str = None,
) -> None:
    """Process a single cancer."""
    cancer_upper = cancer.upper()
    if mean_std_tsv:
        mean_std = mean_std_tsv
    elif mean_std_dir:
        mean_std = str(Path(mean_std_dir) / f"{cancer_upper}_mean_std_FC2.0_TOP150.tsv")
    else:
        mean_std = str(_SCGEP_ROOT.parent / "output" / "redeconv" / f"{cancer_upper}_mean_std_FC2.0_TOP150.tsv")

    if not Path(mean_std).exists():
        raise FileNotFoundError(f"mean_std file not found: {mean_std}")
    if not Path(bulk_tpm_tsv).exists():
        raise FileNotFoundError(f"bulk_tpm_tsv not found: {bulk_tpm_tsv}")

    ensure_dir(str(out_dir))

    out_tsv = out_dir / f"{cancer_upper}_celltypes.tsv"
    out_csv = out_dir / f"{cancer_upper}_ratios_redeconv_10.csv"

    # Use local redeconv package
    from redeconv.__ReDeconv_P import ReDeconv  # type: ignore

    ReDeconv(mean_std, bulk_tpm_tsv, str(out_tsv))
    print(f"Saved ReDeconv output TSV: {out_tsv}")

    # Convert to CSV with proper format
    df = pd.read_csv(out_tsv, sep="\t")
    # Ensure the first column name matches downstream scripts.
    if df.columns[0] != "sample/cell_type":
        df = df.rename(columns={df.columns[0]: "sample/cell_type"})
    df.to_csv(out_csv, index=False)
    print(f"Saved ReDeconv output CSV: {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="Run ReDeconv (percentage prediction) in a non-interactive way.")
    parser.add_argument("--cancer", type=str, default=None, help="Cancer type (e.g. LUAD). If omitted, process all cancers.")
    parser.add_argument(
        "--include_cancers",
        type=str,
        default="",
        help="Optional comma-separated cancers to include (default: all except COAD/BRCA).",
    )
    parser.add_argument("--path_json", type=str, default=str(_SCGEP_ROOT.parent / "config" / "path.json"))
    parser.add_argument("--mean_std_dir", type=str, default=None, help="Dir containing {CANCER}_mean_std_FC2.0_TOP150.tsv")
    parser.add_argument("--pseudobulk_root", type=str, default=None, help="Root directory containing per-cancer pseudobulk TSVs (e.g., /path/to/pseudobulk).")
    parser.add_argument("--bulk_tsv", type=str, default=None, help="Direct path to bulk TPM TSV (genes x samples). Overrides pseudobulk_root when set.")
    parser.add_argument("--out_root", type=str, required=True, help="Output root directory for ratio CSVs (e.g., /path/to/redeconv_output).")
    parser.add_argument(
        "--mean_std_tsv",
        type=str,
        default=None,
        help="Path to mean_std TSV (overrides --mean_std_dir when processing single cancer)",
    )
    args = parser.parse_args()

    cfg = load_path_json(args.path_json)
    
    # Determine which cancers to process
    if args.cancer:
        cancers = [args.cancer.upper()]
    else:
        include = [c.strip() for c in args.include_cancers.split(",") if c.strip()] or None
        cancers = iter_cancers(cfg, include=include)
    
    out_root = Path(args.out_root)
    ensure_dir(str(out_root))
    pseudobulk_root = Path(args.pseudobulk_root) if args.pseudobulk_root else None
    if not args.bulk_tsv and not pseudobulk_root:
        raise ValueError("Either --pseudobulk_root or --bulk_tsv must be provided.")
    
    print(f"Processing {len(cancers)} cancer(s): {', '.join(cancers)}")
    
    for cancer in cancers:
        print(f"\n{'='*60}")
        print(f"Processing {cancer}")
        print(f"{'='*60}")
        
        if args.bulk_tsv:
            bulk_tpm_tsv = Path(args.bulk_tsv)
        else:
            bulk_tpm_tsv = pseudobulk_root / cancer / f"{cancer}_pseudobulk_10_TPM.tsv"
        out_dir = out_root / cancer
        
        try:
            process_one_cancer(
                cancer, str(bulk_tpm_tsv), out_dir,
                mean_std_tsv=args.mean_std_tsv,
                mean_std_dir=args.mean_std_dir,
            )
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


