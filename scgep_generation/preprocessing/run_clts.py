#!/usr/bin/env python3
"""
Weighted CLTS normalization. Wrapper for ReDeconv_Normalization.
Run: python -m redeconv.ReDeconv_Normalization (interactive)
Or call the underlying functions directly for non-interactive use.
Input: fn_meta (sc_meta.tsv), fn_exp (expression_sc.tsv)
Output: Ctype_size_means.tsv, Ctype_cell_counts.tsv, Cell_trans_sizes.tsv
"""
import argparse
import sys
from pathlib import Path

_SCGEP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCGEP_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Weighted CLTS normalization. For interactive use, run: "
        "python -m redeconv.ReDeconv_Normalization"
    )
    parser.add_argument("--data_dir", type=str, help="Data dir with sc_meta.tsv, expression_sc.tsv")
    parser.add_argument("--fn_meta", type=str, help="Path to sc_meta.tsv")
    parser.add_argument("--fn_exp", type=str, help="Path to expression_sc.tsv")
    parser.add_argument("--out_dir", type=str, help="Output directory")
    args = parser.parse_args()
    norm_module = _SCGEP_ROOT / "redeconv" / "ReDeconv_Normalization.py"
    print(f"CLTS normalization: {norm_module}")
    print("Edit fn_meta, fn_exp in ReDeconv_Normalization.py or run interactively:")
    print(f"  cd {_SCGEP_ROOT} && python -m redeconv.ReDeconv_Normalization")


if __name__ == "__main__":
    main()
