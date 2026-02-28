"""
Survival data preparation utilities.
Merge expression with clinical data, split into 5-fold CV format.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from pathlib import Path


def prepare_survival_folds(
    expression_path: str,
    clinical_path: str,
    genes_list_path: str,
    output_dir: str,
    deg_path: str = None,
    data_dir: str = "data",
    deg_thresh: float = 1.5,
):
    """
    Prepare 5-fold CV data for survival analysis.
    expression_path: bulk expression CSV (genes x samples or samples x genes)
    clinical_path: clinical CSV with bcr_patient_barcode, OS, OS.time
    genes_list_path: gene list for intersection
    output_dir: output directory for train_data_1..5, val_data_1..5
    deg_path: optional DEG file for gene filtering
    """
    expression_df = pd.read_csv(expression_path, index_col=0)
    expression_df.index = [str(g).split(".")[0] for g in expression_df.index]
    if expression_df.index.duplicated().any():
        expression_df = expression_df.groupby(level=0).max()

    expression_df = expression_df.T
    genes_non_zero = expression_df.columns[expression_df.sum(axis=0) != 0]
    expression_df = expression_df[genes_non_zero]
    expression_df = expression_df.loc[:, expression_df.std() != 0]

    genes_list = pd.read_csv(genes_list_path, index_col=0)
    if "1" in genes_list.columns:
        genes_list = genes_list.drop(columns=["1"])
    if "0" in genes_list.columns:
        genes_list = genes_list.set_index(["0"])
    expression_df = expression_df.T
    intersected_df = expression_df.merge(genes_list, how="inner", left_index=True, right_index=True).T
    intersected_df.insert(0, "barcode", intersected_df.index)
    intersected_df.index = intersected_df.index.str.slice(0, 12)
    intersected_df.index.name = "bcr_patient_barcode"

    clinical_df = pd.read_csv(clinical_path, index_col=0)
    exp_df = intersected_df.merge(clinical_df, how="inner", on="bcr_patient_barcode")
    exp_df.index = exp_df["barcode"]
    exp_df.drop(columns=["barcode"], inplace=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for fold_idx, (train_index, test_index) in enumerate(kf.split(exp_df)):
        train_df = exp_df.iloc[train_index]
        test_df = exp_df.iloc[test_index]
        train_df.T.to_csv(Path(output_dir) / f"train_data_{fold_idx+1}.csv")
        test_df.T.to_csv(Path(output_dir) / f"val_data_{fold_idx+1}.csv")


def filter_by_deg(expression_df: pd.DataFrame, deg_path: str, deg_thresh: float = 1.5) -> pd.DataFrame:
    """Filter expression to DEG genes."""
    filtered_df = pd.read_csv(deg_path, index_col=0)
    intersected = expression_df.merge(filtered_df, how="inner", left_index=True, right_index=True)
    return intersected.drop(columns=filtered_df.columns, errors="ignore")
