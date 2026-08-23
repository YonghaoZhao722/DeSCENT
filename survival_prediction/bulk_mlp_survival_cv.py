#!/usr/bin/env python3
"""Leakage-free bulk-only survival baselines for DeSCENT comparisons.

The outer fold is evaluated once per pre-specified model family. Epoch and
hyperparameter selection use only an inner validation split carved from the
outer training data. No per-epoch test metrics or history files are produced.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pycox.models.data import pair_rank_mat
from pycox.models.loss import DeepHitSingleLoss

from mil_survival_training import cox_ph_loss, deepsurv_like_risk_from_output, make_time_bins
from scrna_bulk_sc_survival import load_survival_fold
from scrna_bulk_sc_survival_cv import (
    build_survival_dataframe,
    compute_integrated_brier_score,
    compute_td_auc,
    harrell_cindex_from_arrays,
    resolve_selected_fold_genes,
    split_train_early_val_ids,
)


@dataclass(frozen=True)
class Candidate:
    hidden: Tuple[int, ...]
    dropout: float
    lr: float
    weight_decay: float
    transform: str = "log1p_zscore"
    num_bins: int = 20
    alpha: float = 0.2
    sigma: float = 0.1


class BulkMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: Sequence[int], dropout: float, output_dim: int):
        super().__init__()
        layers: List[nn.Module] = []
        previous = int(input_dim)
        for width in hidden:
            layers.extend([nn.Linear(previous, int(width)), nn.ReLU(), nn.Dropout(float(dropout))])
            previous = int(width)
        layers.append(nn.Linear(previous, int(output_dim)))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def candidate_grid(model_name: str) -> List[Candidate]:
    if model_name == "cox":
        return [
            Candidate((), 0.0, 1e-3, 1e-3),
            Candidate((), 0.0, 3e-4, 1e-2),
            Candidate((), 0.0, 1e-3, 1e-2, transform="zscore"),
        ]
    if model_name == "deepsurv":
        return [
            Candidate((128, 64), 0.1, 1e-3, 1e-4),
            Candidate((256, 64), 0.2, 5e-4, 1e-4),
            Candidate((128, 64), 0.3, 5e-4, 1e-3),
            Candidate((128, 64), 0.1, 5e-4, 1e-4, transform="zscore"),
        ]
    if model_name == "deephit":
        return [
            Candidate((128, 64), 0.1, 1e-3, 1e-4, num_bins=20),
            Candidate((256, 64), 0.2, 5e-4, 1e-4, num_bins=20),
            Candidate((128, 64), 0.2, 5e-4, 1e-3, num_bins=40),
            Candidate((128, 64), 0.1, 5e-4, 1e-4, transform="zscore", num_bins=20),
        ]
    raise ValueError(f"Unknown model family: {model_name}")


def align_survival_to_bulk(
    survival: pd.DataFrame,
    bulk: pd.DataFrame,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    bulk_index = bulk.index.astype(str)
    bulk_lookup = {sample: sample for sample in bulk_index}
    for sample in bulk_index:
        bulk_lookup.setdefault(sample[:12], sample)

    ids: List[str] = []
    rows: List[str] = []
    durations: List[float] = []
    events: List[int] = []
    for _, row in survival.iterrows():
        patient = str(row["tcga_barcode"])
        bulk_id = bulk_lookup.get(patient, bulk_lookup.get(patient[:12]))
        if bulk_id is None:
            continue
        ids.append(patient)
        rows.append(bulk_id)
        durations.append(float(row["duration"]))
        events.append(int(row["event"]))
    return ids, bulk.loc[rows].to_numpy(np.float32), np.asarray(durations, np.float32), np.asarray(events, np.float32)


def transform_features(
    fit: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    transform: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if transform == "log1p_zscore":
        fit = np.log1p(np.clip(fit, 0.0, None))
        val = np.log1p(np.clip(val, 0.0, None))
        test = np.log1p(np.clip(test, 0.0, None))
    elif transform != "zscore":
        raise ValueError(f"Unknown feature transform: {transform}")
    mean = fit.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = fit.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return (fit - mean) / scale, (val - mean) / scale, (test - mean) / scale, mean, scale


def append_celltype_features(
    bulk: pd.DataFrame,
    celltypes: pd.DataFrame,
    feature_names: Sequence[str],
) -> Tuple[pd.DataFrame, int]:
    """Append cell fractions to bulk rows using exact ID, then patient-prefix matching."""
    exact = {str(sample): str(sample) for sample in celltypes.index}
    prefix: Dict[str, str] = {}
    for sample in celltypes.index.astype(str):
        prefix.setdefault(sample[:12], sample)
    rows = []
    missing = []
    for sample in bulk.index.astype(str):
        celltype_id = exact.get(sample, prefix.get(sample[:12]))
        if celltype_id is None:
            missing.append(sample)
            rows.append(np.zeros(celltypes.shape[1], dtype=np.float32))
        else:
            rows.append(celltypes.loc[celltype_id].to_numpy(np.float32))
    fractions = pd.DataFrame(rows, index=bulk.index, columns=list(feature_names))
    return pd.concat([bulk, fractions], axis=1), len(missing)


def predict_arrays(
    model: nn.Module,
    x: torch.Tensor,
    duration: np.ndarray,
    event: np.ndarray,
    model_name: str,
    bin_edges: Optional[np.ndarray],
) -> Dict[str, np.ndarray]:
    model.eval()
    with torch.no_grad():
        output = model(x)
        if model_name in ("cox", "deepsurv"):
            risk = output.view(-1)
            logits = np.empty((len(duration), 0), dtype=np.float32)
        else:
            risk = deepsurv_like_risk_from_output(output, "deephit", bin_edges)
            logits = output.detach().cpu().numpy()
    return {
        "risk": risk.detach().cpu().numpy(),
        "logits": logits,
        "duration": np.asarray(duration, dtype=np.float64),
        "event": np.asarray(event, dtype=np.int64),
    }


def evaluate_predictions(
    model_name: str,
    train_pred: Dict[str, np.ndarray],
    eval_pred: Dict[str, np.ndarray],
    bin_edges: Optional[np.ndarray],
) -> Dict[str, float]:
    loss_name = "deephit" if model_name == "deephit" else "cox"
    survival = build_survival_dataframe(loss_name, train_pred, eval_pred, bin_edges)
    return {
        "c_index": harrell_cindex_from_arrays(eval_pred["risk"], eval_pred["duration"], eval_pred["event"]),
        "td_auc": compute_td_auc(train_pred, eval_pred),
        "integrated_brier_score": compute_integrated_brier_score(
            survival, eval_pred["duration"], eval_pred["event"]
        ),
    }


def train_candidate(
    model_name: str,
    candidate: Candidate,
    x_fit_raw: np.ndarray,
    d_fit: np.ndarray,
    e_fit: np.ndarray,
    x_val_raw: np.ndarray,
    d_val: np.ndarray,
    e_val: np.ndarray,
    x_test_raw: np.ndarray,
    args: argparse.Namespace,
    fold: int,
) -> Dict[str, object]:
    x_fit_np, x_val_np, x_test_np, feature_mean, feature_scale = transform_features(
        x_fit_raw.copy(), x_val_raw.copy(), x_test_raw.copy(), candidate.transform
    )
    device = torch.device(args.device)
    x_fit = torch.as_tensor(x_fit_np, dtype=torch.float32, device=device)
    x_val = torch.as_tensor(x_val_np, dtype=torch.float32, device=device)
    x_test = torch.as_tensor(x_test_np, dtype=torch.float32, device=device)
    d_fit_t = torch.as_tensor(d_fit, dtype=torch.float32, device=device)
    e_fit_t = torch.as_tensor(e_fit, dtype=torch.float32, device=device)

    torch.manual_seed(int(args.seed) + int(fold))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed) + int(fold))

    bin_edges: Optional[np.ndarray] = None
    output_dim = 1
    deep_hit_loss = None
    idx_fit_t = None
    rank_fit_t = None
    if model_name == "deephit":
        bin_edges = make_time_bins(d_fit, num_bins=candidate.num_bins, strategy="quantile")
        output_dim = candidate.num_bins
        idx_fit = np.searchsorted(bin_edges, d_fit, side="right") - 1
        idx_fit = np.clip(idx_fit, 0, candidate.num_bins - 1).astype(np.int64)
        idx_fit_t = torch.as_tensor(idx_fit, dtype=torch.long, device=device)
        rank_fit_t = torch.as_tensor(pair_rank_mat(idx_fit, e_fit), dtype=torch.float32, device=device)
        deep_hit_loss = DeepHitSingleLoss(alpha=candidate.alpha, sigma=candidate.sigma)

    model = BulkMLP(x_fit.shape[1], candidate.hidden, candidate.dropout, output_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=candidate.lr, weight_decay=candidate.weight_decay)
    val_scores: List[float] = []
    best_state = None
    best_epoch = 0
    best_raw_val = float("nan")
    best_selection = -math.inf
    epochs_without_improvement = 0

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(x_fit)
        if model_name in ("cox", "deepsurv"):
            loss = cox_ph_loss(output.view(-1), d_fit_t, e_fit_t)
        else:
            assert deep_hit_loss is not None and idx_fit_t is not None and rank_fit_t is not None
            loss = deep_hit_loss(output, idx_fit_t, e_fit_t, rank_fit_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        val_pred = predict_arrays(model, x_val, d_val, e_val, model_name, bin_edges)
        raw_val = harrell_cindex_from_arrays(val_pred["risk"], d_val, e_val)
        val_scores.append(float(raw_val))
        selection = float("nan")
        if len(val_scores) >= int(args.selection_smoothing_window):
            recent = val_scores[-int(args.selection_smoothing_window):]
            if all(np.isfinite(recent)):
                selection = float(np.mean(recent))
        if np.isfinite(selection) and selection > best_selection:
            best_selection = selection
            best_raw_val = float(raw_val)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        elif np.isfinite(selection):
            epochs_without_improvement += 1
        if best_state is not None and epochs_without_improvement >= int(args.early_stop_patience):
            break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch
        best_raw_val = float(val_scores[-1])
        best_selection = float(val_scores[-1])
    model.load_state_dict(best_state)
    return {
        "model": model,
        "x_fit": x_fit,
        "x_val": x_val,
        "x_test": x_test,
        "bin_edges": bin_edges,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "best_epoch": best_epoch,
        "best_raw_val_c_index": best_raw_val,
        "best_selection_score": best_selection,
    }


def load_fold_arrays(
    fold: int,
    bulk_dir: str,
    survival_dir: str,
    deg_dir: str,
    gene_list_csv: str,
    gene_list_path: str,
    seed: int,
    early_stop_fraction: float,
    celltype_path: str = "",
) -> Dict[str, object]:
    train_bulk = pd.read_csv(os.path.join(bulk_dir, f"train_data_{fold}.csv"), index_col=0)
    test_bulk = pd.read_csv(os.path.join(bulk_dir, f"val_data_{fold}.csv"), index_col=0)
    genes = resolve_selected_fold_genes(deg_dir, [fold], gene_list_csv, gene_list_path)
    genes = [gene for gene in genes if gene in train_bulk.columns and gene in test_bulk.columns]
    if not genes:
        raise ValueError(f"Fold {fold}: no DEG genes overlap the bulk matrices")
    bulk_genes = list(genes)
    train_bulk = train_bulk[bulk_genes].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    test_bulk = test_bulk[bulk_genes].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    celltype_columns: List[str] = []
    missing_train_celltypes = 0
    missing_test_celltypes = 0
    if celltype_path:
        celltypes = pd.read_csv(celltype_path, sep="\t", index_col=0)
        celltypes.index = celltypes.index.astype(str)
        celltypes = celltypes.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        celltype_columns = [f"celltype::{column}" for column in celltypes.columns.astype(str)]
        train_bulk, missing_train_celltypes = append_celltype_features(
            train_bulk, celltypes, celltype_columns
        )
        test_bulk, missing_test_celltypes = append_celltype_features(
            test_bulk, celltypes, celltype_columns
        )
    genes = bulk_genes + celltype_columns
    train_survival, test_survival = load_survival_fold(fold, survival_dir)
    train_ids, train_x, train_d, train_e = align_survival_to_bulk(train_survival, train_bulk)
    test_ids, test_x, test_d, test_e = align_survival_to_bulk(test_survival, test_bulk)

    event_by_id = dict(zip(train_ids, train_e.astype(int)))
    fit_ids, val_ids = split_train_early_val_ids(
        train_ids, event_by_id, val_fraction=early_stop_fraction, seed=seed + fold
    )
    position = {patient: index for index, patient in enumerate(train_ids)}
    fit_idx = np.asarray([position[patient] for patient in fit_ids], dtype=np.int64)
    val_idx = np.asarray([position[patient] for patient in val_ids], dtype=np.int64)
    return {
        "genes": genes,
        "bulk_genes": bulk_genes,
        "celltype_columns": celltype_columns,
        "missing_train_celltypes": missing_train_celltypes,
        "missing_test_celltypes": missing_test_celltypes,
        "fit_x": train_x[fit_idx],
        "fit_d": train_d[fit_idx],
        "fit_e": train_e[fit_idx],
        "val_x": train_x[val_idx],
        "val_d": train_d[val_idx],
        "val_e": train_e[val_idx],
        "test_x": test_x,
        "test_d": test_d,
        "test_e": test_e,
        "counts": {"fit": len(fit_idx), "inner_val": len(val_idx), "outer_test": len(test_ids)},
        "events": {
            "fit": int(train_e[fit_idx].sum()),
            "inner_val": int(train_e[val_idx].sum()),
            "outer_test": int(test_e.sum()),
        },
    }


def run_fold(fold: int, model_names: Sequence[str], args: argparse.Namespace) -> Dict[str, object]:
    arrays = load_fold_arrays(
        fold,
        args.bulk_dir,
        args.survival_dir,
        args.deg_dir,
        args.gene_list_csv,
        args.gene_list_path,
        args.seed,
        args.early_stop_fraction,
        args.celltype_path,
    )
    fold_results: Dict[str, object] = {
        "fold": fold,
        "feature_count": len(arrays["genes"]),
        "bulk_gene_count": len(arrays["bulk_genes"]),
        "celltype_count": len(arrays["celltype_columns"]),
        "missing_train_celltypes": arrays["missing_train_celltypes"],
        "missing_test_celltypes": arrays["missing_test_celltypes"],
        "counts": arrays["counts"],
        "events": arrays["events"],
        "models": {},
    }
    fold_dir = Path(args.results_dir) / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    for model_name in model_names:
        candidates = candidate_grid(model_name)
        if args.fixed_config:
            candidates = candidates[:1]
        trained: List[Dict[str, object]] = []
        for index, candidate in enumerate(candidates, start=1):
            result = train_candidate(
                model_name,
                candidate,
                arrays["fit_x"], arrays["fit_d"], arrays["fit_e"],
                arrays["val_x"], arrays["val_d"], arrays["val_e"],
                arrays["test_x"], args, fold,
            )
            trained.append(result)
            print(
                f"Fold {fold} {model_name} candidate {index}/{len(candidates)}: "
                f"val_selection={result['best_selection_score']:.4f} epoch={result['best_epoch']}",
                flush=True,
            )
        selected_index = int(np.argmax([float(item["best_selection_score"]) for item in trained]))
        selected = trained[selected_index]
        candidate = candidates[selected_index]
        model = selected["model"]
        bin_edges = selected["bin_edges"]
        train_pred = predict_arrays(model, selected["x_fit"], arrays["fit_d"], arrays["fit_e"], model_name, bin_edges)
        val_pred = predict_arrays(model, selected["x_val"], arrays["val_d"], arrays["val_e"], model_name, bin_edges)
        test_pred = predict_arrays(model, selected["x_test"], arrays["test_d"], arrays["test_e"], model_name, bin_edges)
        val_metrics = evaluate_predictions(model_name, train_pred, val_pred, bin_edges)
        test_metrics = evaluate_predictions(model_name, train_pred, test_pred, bin_edges)
        metrics = {
            "selected_candidate": selected_index + 1,
            "candidate": asdict(candidate),
            "best_epoch": int(selected["best_epoch"]),
            "best_selection_score": float(selected["best_selection_score"]),
            "inner_val": val_metrics,
            "outer_test": test_metrics,
        }
        fold_results["models"][model_name] = metrics
        torch.save(
            {
                "model": model.state_dict(),
                "model_name": model_name,
                "candidate": asdict(candidate),
                "genes": arrays["genes"],
                "bulk_genes": arrays["bulk_genes"],
                "celltype_columns": arrays["celltype_columns"],
                "missing_train_celltypes": arrays["missing_train_celltypes"],
                "missing_test_celltypes": arrays["missing_test_celltypes"],
                "feature_mean": selected["feature_mean"],
                "feature_scale": selected["feature_scale"],
                "bin_edges": bin_edges,
                "metrics": metrics,
            },
            fold_dir / f"{model_name}.pt",
        )
        print(
            f"Fold {fold} {model_name}: inner_val={val_metrics['c_index']:.4f}, "
            f"outer_test={test_metrics['c_index']:.4f}",
            flush=True,
        )

    with open(fold_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(fold_results, handle, indent=2)
    return fold_results


def resolve_config(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    with open(config_path, encoding="utf-8") as handle:
        entry = json.load(handle)[args.cancer.upper()]
    root = config_path.parent.parent

    def path_from_config(key: str, explicit: str) -> str:
        if explicit:
            return explicit
        path = Path(entry[key])
        return str(path if path.is_absolute() else root / path)

    args.bulk_dir = path_from_config("bulk", args.bulk_dir)
    args.survival_dir = path_from_config("surv_label", args.survival_dir)
    args.deg_dir = path_from_config("deg_dir", args.deg_dir)
    args.gene_list_csv = path_from_config("gene_list", args.gene_list_csv)
    args.gene_list_path = path_from_config("gene_list_path", args.gene_list_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk-only Cox/DeepSurv/DeepHit MLP baselines")
    parser.add_argument("--cancer", default="BRCA")
    parser.add_argument("--config", default="config/path_local.json")
    parser.add_argument("--bulk_dir", default="")
    parser.add_argument("--survival_dir", default="")
    parser.add_argument("--deg_dir", default="")
    parser.add_argument("--gene_list_csv", default="")
    parser.add_argument("--gene_list_path", default="")
    parser.add_argument(
        "--celltype_path",
        default="",
        help="Optional TSV of sample-by-celltype fractions appended to the bulk features",
    )
    parser.add_argument("--results_dir", default="output/bulk_mlp_survival_cv/BRCA")
    parser.add_argument("--models", nargs="+", default=["cox", "deepsurv", "deephit"], choices=["cox", "deepsurv", "deephit"])
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--early_stop_fraction", type=float, default=0.2)
    parser.add_argument("--selection_smoothing_window", type=int, default=5)
    parser.add_argument("--early_stop_patience", type=int, default=30)
    parser.add_argument(
        "--fixed_config",
        action="store_true",
        help="Use the first pre-specified candidate for each model family instead of inner hyperparameter search",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    resolve_config(args)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    folds = [run_fold(fold, args.models, args) for fold in args.folds]
    aggregate: Dict[str, object] = {}
    for model_name in args.models:
        test_values = np.asarray(
            [fold["models"][model_name]["outer_test"]["c_index"] for fold in folds], dtype=float
        )
        val_values = np.asarray(
            [fold["models"][model_name]["inner_val"]["c_index"] for fold in folds], dtype=float
        )
        aggregate[model_name] = {
            "mean_outer_test_c_index": float(np.nanmean(test_values)),
            "std_outer_test_c_index": float(np.nanstd(test_values)),
            "mean_inner_val_c_index": float(np.nanmean(val_values)),
            "fold_outer_test_c_index": test_values.tolist(),
        }
    summary = {"cancer": args.cancer, "folds": folds, "aggregate": aggregate, "args": vars(args)}
    with open(Path(args.results_dir) / "cv_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()
