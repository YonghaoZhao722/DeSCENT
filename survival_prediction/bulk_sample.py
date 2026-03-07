"""
Bulk expression sampling utilities for DEG analysis.
Prepare expression_bulk_umi_group.csv with sample labels (0=normal, 1=tumor) for PyDESeq2.
"""
import pandas as pd
from pathlib import Path


def prepare_bulk_for_deg(
    expression_path: str,
    output_path: str,
    genes_list_path: str,
    data_unit: str = "umi",
):
    """
    Prepare bulk expression for DEG (PyDESeq2).
    Adds 'sample' column: 0=normal, 1=tumor (TCGA sample type from barcode).
    """
    expression_df = pd.read_csv(expression_path, index_col=0)
    expression_df.index = [str(g).split(".")[0] for g in expression_df.index]
    if expression_df.index.duplicated().any():
        expression_df = expression_df.groupby(level=0).max()

    expression_df = expression_df.T
    genes_non_zero = expression_df.columns[expression_df.sum(axis=0) != 0]
    expression_df = expression_df[genes_non_zero]
    expression_df = expression_df.loc[:, expression_df.std() != 0]

    expression_df["study"] = expression_df.index.str.slice(0, 4)
    expression_df.loc[(expression_df["study"] == "GTEX"), "sample"] = 0
    expression_df["_sample"] = expression_df.index.str.slice(13, 15)
    expression_df.loc[expression_df["_sample"].str[-1] == "-", "_sample"] = 10
    expression_df["_sample"] = expression_df["_sample"].astype(int)
    expression_df.loc[
        (expression_df["study"] == "TCGA") & (expression_df["_sample"] < 10), "sample"
    ] = 1
    expression_df.loc[
        (expression_df["study"] == "TCGA") & (expression_df["_sample"] >= 10), "sample"
    ] = 0
    expression_df["sample"] = expression_df["sample"].astype(int)
    expression_df.drop(columns=["study", "_sample"], inplace=True)

    sample_df = expression_df.pop("sample")
    if data_unit == "log2":
        expression_df = (pow(2, expression_df) - 1).round().astype(int)
    expression_df.insert(0, "sample", sample_df)

    genes_list = pd.read_csv(genes_list_path, index_col=0)
    if "1" in genes_list.columns:
        genes_list = genes_list.drop(columns=["1"])
    if "0" in genes_list.columns:
        genes_list = genes_list.set_index(["0"])
    expression_df = expression_df.T
    intersected_df = expression_df.merge(genes_list, how="inner", left_index=True, right_index=True).T
    intersected_df = pd.concat([sample_df, intersected_df], axis=1)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    intersected_df.T.to_csv(output_path)
