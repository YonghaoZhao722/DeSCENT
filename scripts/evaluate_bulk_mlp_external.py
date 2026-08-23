#!/usr/bin/env python3
"""Re-evaluate frozen bulk-only checkpoints on internal and external cohorts.

The internal outer-test metrics are reconstructed from each checkpoint's original
fit/test split.  External cohorts are evaluated once per frozen fold model; no
external outcome is used for fitting, checkpoint selection, or preprocessing.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SURVIVAL_DIR = ROOT / "survival_prediction"
if str(SURVIVAL_DIR) not in sys.path:
    sys.path.insert(0, str(SURVIVAL_DIR))

from bulk_mlp_survival_cv import (  # noqa: E402
    BulkMLP,
    evaluate_predictions,
    load_fold_arrays,
    predict_arrays,
)
from scrna_bulk_sc_survival import load_survival_fold  # noqa: E402
from scrna_bulk_sc_survival_cv import (  # noqa: E402
    build_survival_dataframe,
    compute_integrated_brier_score,
    compute_ipa,
)


CANCERS = ("BRCA", "COAD", "HNSC", "KIRC", "LGG", "LIHC", "LUAD", "STAD")
MODELS = ("cox", "deepsurv", "deephit")
EXTERNAL_COHORT = {
    "BRCA": "GSE96058",
    "COAD": "GSE39582",
    "HNSC": "GSE65858",
    "KIRC": "CPTAC",
    "LGG": "GSE184941",
    "LIHC": "GSE76427",
    "LUAD": "GSE68465",
    "STAD": "GSE84437",
}
# TCGA OS.time is in DAYS (e.g. BRCA 4047.0, median 847, max 8605). Every GEO external
# cohort here reports OS.time in MONTHS (GSE96058 77.76; GSE84437 23.0/24.0). Checkpoints
# fit their baseline hazard / discrete-time bins on the training day axis, so external
# durations must be converted before any time-dependent metric is computed.
# Symptom of the missing conversion: internal IBS 0.169-0.301 (sane) but external IBS
# 0.054-0.774, scattered in BOTH directions. C-index is rank-based and was unaffected.
DAYS_PER_MONTH = 365.25 / 12.0
EXTERNAL_TIME_IN_MONTHS = set(EXTERNAL_COHORT)

REFERENCE_SAMPLE_SIZE = {
    "BRCA": "3,273 tumor samples",
    "COAD": "566 tumor samples",
    "HNSC": "270 tumor samples",
    "KIRC": "60 deaths; survival time unavailable for non-deaths",
    "LGG": "180 tumor samples",
    "LIHC": "167 tumor samples",
    "LUAD": "443 tumor samples",
    "STAD": "483 tumor samples",
}


def torch_load(path: Path, device: torch.device) -> Dict[str, object]:
    return torch.load(path, map_location=device, weights_only=False)


def read_selected_columns(path: Path, genes: Iterable[str]) -> pd.DataFrame:
    """Read the index and requested genes without materializing the full CSV."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if not header:
        raise ValueError(f"Empty CSV: {path}")
    wanted = set(map(str, genes))
    index_column = header[0]
    available = wanted.intersection(header[1:])
    frame = pd.read_csv(
        path,
        index_col=0,
        usecols=lambda column: column == index_column or column in available,
    )
    frame.index = frame.index.astype(str)
    return frame


def external_paths(cancer: str) -> Tuple[Path, Path]:
    bulk_dir = ROOT / "data" / cancer / "bulk_geo"
    if cancer == "BRCA":
        survival_dir = Path("/data/zhaoyh/SHAP/data/BRCA_GEO")
    else:
        survival_dir = Path("/data/zhaoyh/SHAP/data") / cancer
    return bulk_dir, survival_dir


def normalized_celltype_name(value: str) -> str:
    return "".join(character.lower() for character in str(value) if character.isalnum())


def append_external_celltypes(
    cancer: str,
    frame: pd.DataFrame,
    target_columns: Sequence[str],
) -> Tuple[pd.DataFrame, int, int]:
    if not target_columns:
        return frame, 0, 0
    if cancer == "BRCA":
        path = ROOT / "output/redeconv_fraction/BRCA_GEO/BRCA_GEO_ratios_redeconv_10.csv"
    else:
        path = ROOT / f"output/redeconv_fraction/{cancer}/{cancer}_ratios_redeconv_10.csv"
    fractions = pd.read_csv(path, index_col=0)
    fractions.index = fractions.index.astype(str)
    fractions = fractions.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    source_by_name: Dict[str, str] = {}
    for column in fractions.columns.astype(str):
        key = normalized_celltype_name(column)
        if key in source_by_name:
            raise ValueError(f"{cancer}: ambiguous normalized external celltype name {key}")
        source_by_name[key] = column

    matched_columns: Dict[str, str] = {}
    for target in target_columns:
        original = str(target).split("celltype::", 1)[-1]
        source = source_by_name.get(normalized_celltype_name(original))
        if source is not None:
            matched_columns[target] = source
    if len(matched_columns) != len(target_columns):
        missing = sorted(set(target_columns) - set(matched_columns))
        raise ValueError(f"{cancer}: external celltype columns missing for {missing}")

    exact = {sample: sample for sample in fractions.index}
    prefix: Dict[str, str] = {}
    for sample in fractions.index:
        prefix.setdefault(sample[:12], sample)
    rows = []
    missing_samples = 0
    sources = [matched_columns[target] for target in target_columns]
    for sample in frame.index.astype(str):
        matched = exact.get(sample, prefix.get(sample[:12]))
        if matched is None:
            missing_samples += 1
            rows.append(np.zeros(len(target_columns), dtype=np.float32))
        else:
            rows.append(fractions.loc[matched, sources].to_numpy(np.float32))
    values = pd.DataFrame(rows, index=frame.index, columns=list(target_columns))
    return pd.concat([frame, values], axis=1), len(matched_columns), missing_samples


def external_gene_mapping(cancer: str, genes: Sequence[str]) -> Dict[str, str]:
    """Return external-column -> checkpoint-gene aliases where naming differs."""
    if cancer != "COAD":
        return {}
    mapping_path = Path("/data/zhaoyh/SHAP/coad_degs_deseq2_new.csv")
    mapping = pd.read_csv(mapping_path, index_col=0)
    wanted = set(map(str, genes))
    mapping.index = mapping.index.astype(str).str.split(".").str[0]
    mapping["symbol"] = mapping["symbol"].astype(str)
    selected = mapping.loc[mapping.index.isin(wanted) & mapping["symbol"].ne("nan"), ["symbol"]]
    # Ambiguous symbols cannot safely represent more than one Ensembl feature.
    selected = selected.loc[~selected["symbol"].duplicated(keep=False)]
    return dict(zip(selected["symbol"], selected.index))


def allowed_external_samples(cancer: str) -> set[str] | None:
    """Apply cohort-specific subject/tumor filters from the source metadata."""
    if cancer == "BRCA":
        # The matrix contains 3,273 subjects plus 136 sequencing replicates.
        selected: Dict[str, str] = {}
        paths = Path("/data/zhaoyh/SHAP/data/BRCA_GEO").glob(
            "GSE96058-*series_matrix.txt.gz"
        )
        for path in sorted(paths):
            titles = external_ids = None
            with gzip.open(path, "rt", errors="replace") as handle:
                for line in handle:
                    if line.startswith("!Sample_title"):
                        titles = [item.strip('\"\n') for item in line.split("\t")[1:]]
                    elif (
                        line.startswith("!Sample_characteristics_ch1")
                        and "scan-b external id:" in line.lower()
                    ):
                        external_ids = [
                            item.strip('\"\n').split(": ", 1)[1]
                            for item in line.split("\t")[1:]
                        ]
                    if titles is not None and external_ids is not None:
                        break
            if titles is None or external_ids is None or len(titles) != len(external_ids):
                raise ValueError(f"Could not parse BRCA replicate metadata from {path}")
            for title, external_id in zip(titles, external_ids):
                selected.setdefault(external_id.split(".l", 1)[0], title)
        if len(selected) != 3273:
            raise ValueError(f"Expected 3,273 unique GSE96058 subjects, found {len(selected)}")
        return set(selected.values())

    if cancer == "COAD":
        # GSE39582 contains 566 primary tumors and 19 non-tumoral mucosa samples.
        path = Path("/data/zhaoyh/SHAP/data/COAD/GSE39582_series_matrix.txt.gz")
        accessions = sources = None
        with gzip.open(path, "rt", errors="replace") as handle:
            for line in handle:
                if line.startswith("!Sample_geo_accession"):
                    accessions = [item.strip('\"\n') for item in line.split("\t")[1:]]
                elif line.startswith("!Sample_source_name_ch1"):
                    sources = [item.strip('\"\n') for item in line.split("\t")[1:]]
                if accessions is not None and sources is not None:
                    break
        if accessions is None or sources is None or len(accessions) != len(sources):
            raise ValueError(f"Could not parse COAD tumor metadata from {path}")
        selected = {
            accession
            for accession, source in zip(accessions, sources)
            if "primary colorectal Adenocarcinoma" in source
        }
        if len(selected) != 566:
            raise ValueError(f"Expected 566 GSE39582 tumors, found {len(selected)}")
        return selected
    return None


def load_external_cohort(cancer: str, genes: Sequence[str]) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Recombine prepared fold-1 train/val files into the full external cohort."""
    bulk_dir, survival_dir = external_paths(cancer)
    aliases = external_gene_mapping(cancer, genes)
    source_genes = list(aliases) if aliases else list(genes)
    frames = [
        read_selected_columns(bulk_dir / f"{split}_data_1.csv", source_genes).rename(columns=aliases)
        for split in ("train", "val")
    ]
    bulk = pd.concat(frames, axis=0)
    if bulk.index.duplicated().any():
        duplicates = bulk.index[bulk.index.duplicated()].unique().tolist()[:5]
        raise ValueError(f"{cancer}: duplicate external sample IDs: {duplicates}")

    train_survival, val_survival = load_survival_fold(1, str(survival_dir))
    survival = pd.concat([train_survival, val_survival], ignore_index=True)
    survival["tcga_barcode"] = survival["tcga_barcode"].astype(str)
    exact = {sample: sample for sample in bulk.index}
    prefix = {}
    for sample in bulk.index:
        prefix.setdefault(sample[:12], sample)

    rows: List[str] = []
    durations: List[float] = []
    events: List[int] = []
    allowed = allowed_external_samples(cancer)
    for _, record in survival.iterrows():
        sample = str(record["tcga_barcode"])
        matched = exact.get(sample, prefix.get(sample[:12]))
        duration = pd.to_numeric(record["duration"], errors="coerce")
        event = pd.to_numeric(record["event"], errors="coerce")
        if (
            matched is None
            or (allowed is not None and matched not in allowed)
            or not np.isfinite(duration)
            or not np.isfinite(event)
        ):
            continue
        rows.append(matched)
        durations.append(float(duration))
        events.append(int(event))
    if not rows:
        raise ValueError(f"{cancer}: no external expression/survival overlap")
    duration_array = np.asarray(durations, dtype=np.float32)
    if cancer in EXTERNAL_TIME_IN_MONTHS:
        duration_array = duration_array * np.float32(DAYS_PER_MONTH)
    return (
        bulk.loc[rows],
        duration_array,
        np.asarray(events, dtype=np.float32),
    )


def standardize_complete(raw: np.ndarray, checkpoint: Dict[str, object]) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float32)
    transform = str(checkpoint["candidate"]["transform"])
    if transform == "log1p_zscore":
        values = np.log1p(np.clip(values, 0.0, None))
    elif transform != "zscore":
        raise ValueError(f"Unknown transform: {transform}")
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["feature_scale"], dtype=np.float32)
    return (values - mean) / scale


def standardize_external(
    frame: pd.DataFrame, checkpoint: Dict[str, object]
) -> Tuple[np.ndarray, int, int]:
    """Map external genes to checkpoint order; missing genes receive z-score 0."""
    genes = list(map(str, checkpoint["genes"]))
    positions = {gene: index for index, gene in enumerate(genes)}
    present = [gene for gene in genes if gene in frame.columns]
    values = np.zeros((len(frame), len(genes)), dtype=np.float32)
    if present:
        raw = frame[present].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)
        if str(checkpoint["candidate"]["transform"]) == "log1p_zscore":
            raw = np.log1p(np.clip(raw, 0.0, None))
        elif str(checkpoint["candidate"]["transform"]) != "zscore":
            raise ValueError(f"Unknown transform: {checkpoint['candidate']['transform']}")
        indices = np.asarray([positions[gene] for gene in present], dtype=np.int64)
        mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)[indices]
        scale = np.asarray(checkpoint["feature_scale"], dtype=np.float32)[indices]
        values[:, indices] = (raw - mean) / scale
    return values, len(present), len(genes)


def build_model(checkpoint: Dict[str, object], device: torch.device) -> BulkMLP:
    state = checkpoint["model"]
    weights = [tensor for name, tensor in state.items() if name.endswith(".weight")]
    output_dim = int(weights[-1].shape[0])
    candidate = checkpoint["candidate"]
    model = BulkMLP(
        len(checkpoint["genes"]), candidate["hidden"], candidate["dropout"], output_dim
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def predict(
    model: BulkMLP,
    x: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
    model_name: str,
    bin_edges: np.ndarray | None,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
    return predict_arrays(model, tensor, duration, event, model_name, bin_edges)


def restricted_brier_metrics(
    model_name: str,
    fit_pred: Dict[str, np.ndarray],
    eval_pred: Dict[str, np.ndarray],
    bin_edges: np.ndarray | None,
) -> Dict[str, float]:
    """IBS restricted to the external cohort's own follow-up, plus the null-model reference.

    `evaluate_predictions` integrates the Brier score over the *model's* timeline, which is
    the TCGA event grid (BRCA: to 8605 days). GSE96058 is fully censored by 2474 days, so
    two thirds of that integral is pure extrapolation where the IPCW weight 1/G(t) diverges.
    That inflates external IBS ~5x (cox 0.071 -> 0.354) and makes it swing 6x across folds.
    Restricting the horizon to max(observed external follow-up) is the standard fix, and the
    accompanying IPA says whether the model beats the cohort's marginal Kaplan-Meier at all.
    """
    loss_name = "deephit" if model_name == "deephit" else "cox"
    surv = build_survival_dataframe(loss_name, fit_pred, eval_pred, bin_edges)
    horizon = float(np.max(eval_pred["duration"])) if eval_pred["duration"].size else float("nan")
    ibs = compute_integrated_brier_score(surv, eval_pred["duration"], eval_pred["event"], horizon)
    null_ibs, ipa = compute_ipa(surv, eval_pred["duration"], eval_pred["event"], horizon)
    return {
        "ibs_restricted": ibs,
        "ibs_null_model": null_ibs,
        "ipa": ipa,
        "brier_horizon_days": horizon,
    }


def finite_summary(frame: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    rows = []
    metrics = ("c_index", "td_auc", "integrated_brier_score", "ibs_restricted", "ibs_null_model", "ipa")
    for keys, group in frame.groupby(list(group_columns), sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, keys))
        row["n_folds"] = int(len(group))
        for metric in metrics:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(float)
            row[f"mean_{metric}"] = float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
            row[f"std_{metric}"] = float(np.nanstd(values)) if np.isfinite(values).any() else np.nan
            row[f"n_{metric}"] = int(np.isfinite(values).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_cancer(
    cancer: str, checkpoint_root: Path, device: torch.device, prediction_dir: Path | None = None
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    cancer_root = checkpoint_root / cancer
    with (cancer_root / "cv_summary.json").open(encoding="utf-8") as handle:
        training_summary = json.load(handle)
    args = training_summary["args"]

    checkpoints: Dict[Tuple[int, str], Dict[str, object]] = {}
    gene_union = set()
    for fold in range(1, 6):
        for model_name in MODELS:
            checkpoint = torch_load(cancer_root / f"fold{fold}" / f"{model_name}.pt", device)
            checkpoints[(fold, model_name)] = checkpoint
            gene_union.update(map(str, checkpoint.get("bulk_genes", checkpoint["genes"])))
    external_bulk, external_duration, external_event = load_external_cohort(cancer, sorted(gene_union))
    celltype_columns = list(checkpoints[(1, MODELS[0])].get("celltype_columns", []))
    external_bulk, external_celltypes_present, missing_external_celltypes = append_external_celltypes(
        cancer, external_bulk, celltype_columns
    )

    internal_rows: List[Dict[str, object]] = []
    external_rows: List[Dict[str, object]] = []
    for fold in range(1, 6):
        arrays = load_fold_arrays(
            fold,
            args["bulk_dir"],
            args["survival_dir"],
            args["deg_dir"],
            args["gene_list_csv"],
            args["gene_list_path"],
            int(args["seed"]),
            float(args["early_stop_fraction"]),
            args.get("celltype_path", ""),
        )
        for model_name in MODELS:
            checkpoint = checkpoints[(fold, model_name)]
            checkpoint_genes = list(map(str, checkpoint["genes"]))
            if list(map(str, arrays["genes"])) != checkpoint_genes:
                raise ValueError(f"{cancer} fold{fold}: current internal gene order differs from checkpoint")
            model = build_model(checkpoint, device)
            bin_edges = checkpoint.get("bin_edges")
            x_fit = standardize_complete(arrays["fit_x"], checkpoint)
            x_test = standardize_complete(arrays["test_x"], checkpoint)
            fit_pred = predict(
                model, x_fit, arrays["fit_d"], arrays["fit_e"], model_name, bin_edges, device
            )
            test_pred = predict(
                model, x_test, arrays["test_d"], arrays["test_e"], model_name, bin_edges, device
            )
            internal_metrics = evaluate_predictions(model_name, fit_pred, test_pred, bin_edges)
            # Same restriction applied to the internal test fold, so the size of the
            # horizon effect can be read off directly instead of assumed negligible.
            internal_metrics.update(
                restricted_brier_metrics(model_name, fit_pred, test_pred, bin_edges)
            )
            stored = checkpoint["metrics"]["outer_test"]
            internal_rows.append(
                {
                    "cancer": cancer,
                    "model": model_name,
                    "fold": fold,
                    "n_fit": len(arrays["fit_d"]),
                    "n_test": len(arrays["test_d"]),
                    "test_events": int(np.sum(arrays["test_e"])),
                    **internal_metrics,
                    "stored_c_index": float(stored["c_index"]),
                    "c_index_delta_vs_stored": float(internal_metrics["c_index"] - stored["c_index"]),
                    "stored_td_auc": float(stored["td_auc"]),
                    "td_auc_delta_vs_stored": float(internal_metrics["td_auc"] - stored["td_auc"]),
                    "stored_integrated_brier_score": float(stored["integrated_brier_score"]),
                    "integrated_brier_score_delta_vs_stored": float(
                        internal_metrics["integrated_brier_score"]
                        - stored["integrated_brier_score"]
                    ),
                }
            )

            x_external, _, _ = standardize_external(external_bulk, checkpoint)
            bulk_genes = list(map(str, checkpoint.get("bulk_genes", checkpoint["genes"])))
            present = sum(gene in external_bulk.columns for gene in bulk_genes)
            total = len(bulk_genes)
            external_pred = predict(
                model,
                x_external,
                external_duration,
                external_event,
                model_name,
                bin_edges,
                device,
            )
            external_metrics = evaluate_predictions(model_name, fit_pred, external_pred, bin_edges)
            external_metrics.update(
                restricted_brier_metrics(model_name, fit_pred, external_pred, bin_edges)
            )
            external_rows.append(
                {
                    "cancer": cancer,
                    "external_cohort": EXTERNAL_COHORT[cancer],
                    "model": model_name,
                    "fold": fold,
                    "n_external": len(external_duration),
                    "external_events": int(np.sum(external_event)),
                    "genes_present": present,
                    "genes_total": total,
                    "gene_coverage": present / total,
                    "celltypes_present": external_celltypes_present,
                    "celltypes_total": len(celltype_columns),
                    "missing_external_celltype_samples": missing_external_celltypes,
                    "missing_gene_policy": "training_mean_after_standardization",
                    **external_metrics,
                }
            )
            # Cache the raw per-sample predictions. Every survival metric here is a pure
            # function of these arrays, so any later change to the metric definition
            # (integration horizon, IPA, a different tau) is a seconds-long recompute
            # instead of another pass over 120 checkpoints.
            if prediction_dir is not None:
                np.savez_compressed(
                    prediction_dir / f"{cancer}_{model_name}_fold{fold}.npz",
                    bin_edges=np.asarray(bin_edges if bin_edges is not None else [], dtype=np.float64),
                    **{f"fit_{k}": v for k, v in fit_pred.items()},
                    **{f"test_{k}": v for k, v in test_pred.items()},
                    **{f"external_{k}": v for k, v in external_pred.items()},
                )
            del model, x_fit, x_test, x_external, fit_pred, test_pred, external_pred
        del arrays
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"{cancer}: completed fold {fold}/5", flush=True)
    return internal_rows, external_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=ROOT / "output/bulk_mlp_survival_cv/8cancer_fixed_w1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output/bulk_mlp_survival_cv/8cancer_fixed_w1/frozen_external_metrics",
    )
    parser.add_argument("--cancers", nargs="+", choices=CANCERS, default=list(CANCERS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Cache per-sample risk/logits/duration/event so metrics can be recomputed without the GPU",
    )
    cli = parser.parse_args()
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cli.device)
    prediction_dir = cli.output_dir / "predictions" if cli.save_predictions else None
    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)

    internal_rows: List[Dict[str, object]] = []
    external_rows: List[Dict[str, object]] = []
    for cancer in cli.cancers:
        print(f"{cancer}: loading frozen checkpoints and external cohort", flush=True)
        internal, external = evaluate_cancer(cancer, cli.checkpoint_root, device, prediction_dir)
        internal_rows.extend(internal)
        external_rows.extend(external)

    internal_fold = pd.DataFrame(internal_rows)
    external_fold = pd.DataFrame(external_rows)
    internal_summary = finite_summary(internal_fold, ["cancer", "model"])
    external_summary = finite_summary(external_fold, ["cancer", "external_cohort", "model"])
    combined_summary = internal_summary.merge(
        external_summary, on=["cancer", "model"], suffixes=("_internal", "_external")
    )
    cohort_qc = (
        external_fold.groupby(["cancer", "external_cohort"], as_index=False)
        .agg(
            usable_n=("n_external", "first"),
            usable_events=("external_events", "first"),
            min_gene_coverage=("gene_coverage", "min"),
            max_gene_coverage=("gene_coverage", "max"),
            celltypes_present=("celltypes_present", "first"),
            celltypes_total=("celltypes_total", "first"),
            missing_external_celltype_samples=("missing_external_celltype_samples", "first"),
        )
    )
    cohort_qc.insert(
        2, "reference_sample_size_from_workbook", cohort_qc["cancer"].map(REFERENCE_SAMPLE_SIZE)
    )
    notes = pd.DataFrame(
        {
            "item": [
                "checkpoint_root", "internal_evaluation", "external_evaluation",
                "external_checkpoint_selection", "missing_external_genes", "COAD_gene_mapping",
                "BRCA_subject_filter", "COAD_tumor_filter", "celltype_features",
                "missing_internal_celltypes", "tdAUC", "IBS", "KIRC_tdAUC",
            ],
            "description": [
                str(cli.checkpoint_root.resolve()),
                "Original five outer test folds reconstructed from each checkpoint; fit partition supplies the training reference distribution.",
                "Each frozen TCGA fold checkpoint evaluated on the full prepared external cohort (prepared fold-1 train + val recombined).",
                "External outcomes were never used for fitting, preprocessing estimates, epoch selection, or model selection.",
                "Genes absent from an external platform are assigned 0 after training-based standardization (the training mean).",
                "GSE39582 HGNC symbols mapped to TCGA Ensembl IDs with /data/zhaoyh/SHAP/coad_degs_deseq2_new.csv; ambiguous symbols excluded.",
                "GSE96058 sequencing replicates were collapsed to one primary row per SCAN-B subject using the external-ID prefix in GEO metadata (3,409 rows to 3,273 subjects).",
                "GSE39582 non-tumoral mucosa samples were excluded using GEO source_name metadata (566 primary tumors retained).",
                "When present in the checkpoint, ReDeconv cell fractions are appended after the DEG bulk genes and external column names are matched after space/underscore normalization.",
                "TCGA samples without a ReDeconv row retain the original fold and receive an all-zero fraction vector; missing counts are stored in checkpoints and fold metrics.",
                "Mean cumulative/dynamic AUC from scikit-survival, up to 20 event times within train/evaluation follow-up overlap.",
                "Integrated Brier score from the same frozen-checkpoint evaluator used during bulk-baseline training (pycox EvalSurv).",
                "Not estimable: CPTAC KIRC has 24 usable patients, all events, and violates the training censoring/follow-up support required by IPCW tdAUC.",
            ],
        }
    )

    internal_fold.to_csv(cli.output_dir / "internal_fold_metrics.csv", index=False)
    internal_summary.to_csv(cli.output_dir / "internal_summary.csv", index=False)
    external_fold.to_csv(cli.output_dir / "external_fold_metrics.csv", index=False)
    external_summary.to_csv(cli.output_dir / "external_summary.csv", index=False)
    combined_summary.to_csv(cli.output_dir / "combined_summary.csv", index=False)
    cohort_qc.to_csv(cli.output_dir / "external_cohort_qc.csv", index=False)
    notes.to_csv(cli.output_dir / "method_notes.csv", index=False)
    workbook = cli.output_dir / "bulk_baseline_internal_external_metrics.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        combined_summary.to_excel(writer, sheet_name="combined_summary", index=False)
        internal_summary.to_excel(writer, sheet_name="internal_summary", index=False)
        internal_fold.to_excel(writer, sheet_name="internal_folds", index=False)
        external_summary.to_excel(writer, sheet_name="external_summary", index=False)
        external_fold.to_excel(writer, sheet_name="external_folds", index=False)
        cohort_qc.to_excel(writer, sheet_name="external_cohort_QC", index=False)
        notes.to_excel(writer, sheet_name="methods_and_notes", index=False)
    print(f"Wrote results to {cli.output_dir}", flush=True)


if __name__ == "__main__":
    main()
