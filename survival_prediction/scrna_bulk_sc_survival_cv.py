import os
import sys
from pathlib import Path

# Add DeSCENT scgep_generation to path for VAE imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_DESCENT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_DESCENT_ROOT / "scgep_generation"))

import gc
import json
import time
import math
import argparse
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

# Reuse existing components from the single-run script
sys.path.insert(0, str(_SCRIPT_DIR))
from scrna_bulk_sc_survival import (  # type: ignore
    BulkSCFusionSurvival,
    PatientBatchDataset,
    pad_collate_fn,
    load_survival_fold,
    extract_tcga_from_key,
    train_one_epoch,
    evaluate_cindex,
    plot_losses_from_history,
    load_vae,
)
from mil_survival_training import make_time_bins  # type: ignore


def _progress_write(message: str) -> None:
    """Log without breaking active tqdm bars."""
    tqdm.write(str(message))


def resolve_existing_deg_csv(deg_csv: str, cancer: Optional[str]) -> str:
    if not deg_csv or os.path.exists(deg_csv) or cancer is None:
        return deg_csv
    directory = os.path.dirname(deg_csv)
    prefix = cancer.lower()
    candidates = [
        os.path.join(directory, f"degs_tcga_{prefix}_1.5_mapped.csv"),
        os.path.join(directory, f"degs_tcga_{prefix}_1.5.csv"),
        os.path.join(directory, f"degs_tcga_{prefix}.csv"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            print(f"DEG file not found: {deg_csv}. Falling back to {candidate}.")
            return candidate
    return deg_csv


def decode_sc_to_full_gene_space(
    sc_npz_root: str,
    gene_list_csv: str,
    device: torch.device,
    vae_ckpt_path: str,
    vae_num_genes: int,
) -> Tuple[Dict[str, Dict], List[str]]:
    """Decode per-sample cell embeddings to full gene space (no DEG filter).
    Returns decoded_samples and gene_names_full.
    """
    sample_folders = [
        d
        for d in os.listdir(sc_npz_root)
        if os.path.isdir(os.path.join(sc_npz_root, d)) and d.startswith('cells_2048_TCGA-')
    ]
    samples_data: Dict[str, Dict] = {}
    for sample_folder in tqdm(sample_folders, total=len(sample_folders), desc='Loading SC data'):
        sample_path = os.path.join(sc_npz_root, sample_folder)
        npz_files = [os.path.join(sample_path, f) for f in os.listdir(sample_path) if f.endswith('.npz')]
        cell_gen_list: List[np.ndarray] = []
        type_ids: List[np.ndarray] = []
        type_names: List[str] = []
        for file in npz_files:
            npzfile = np.load(file, allow_pickle=True)
            if 'cell_gen' in npzfile:
                data_arr = npzfile['cell_gen']
                cell_gen_list.append(data_arr)
                base = os.path.basename(file)
                tname = os.path.splitext(base)[0]
                if tname not in type_names:
                    type_names.append(tname)
                t_id = type_names.index(tname)
                type_ids.append(np.full((data_arr.shape[0],), t_id, dtype=np.int64))
        if len(cell_gen_list) > 0:
            X_concat = np.concatenate(cell_gen_list, axis=0)
            t_concat = np.concatenate(type_ids, axis=0) if len(type_ids) > 0 else None
            samples_data[sample_folder] = {'X': X_concat, 'type_ids': t_concat, 'type_names': type_names}
    gene_order = pd.read_csv(gene_list_csv)
    gene_names_full = gene_order.iloc[:, 1].astype(str).tolist()
    vae = load_vae(device, num_genes=int(vae_num_genes), ckpt_path=vae_ckpt_path)
    decoded_samples: Dict[str, Dict] = {}
    for sample_name, sd in tqdm(samples_data.items(), total=len(samples_data), desc='Decoding SC data'):
        X = sd['X']
        type_ids_np = sd.get('type_ids', None)
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
            decoded = vae(X_tensor, return_decoded=True).detach().cpu().numpy()
        df = pd.DataFrame(decoded, columns=gene_names_full)
        decoded_samples[sample_name] = {
            'X_df': df,
            'type_ids': (type_ids_np.astype(np.int64) if type_ids_np is not None else None),
            'type_names': sd.get('type_names', []),
        }
    return decoded_samples, gene_names_full


def detect_sc_input_format(sc_npz_root: str) -> str:
    """Infer whether sc_npz_root stores scDiffusion latents or direct gene expression."""
    for dirpath, _, files in os.walk(sc_npz_root):
        for fname in files:
            if not fname.endswith('.npz'):
                continue
            path = os.path.join(dirpath, fname)
            try:
                npzfile = np.load(path, allow_pickle=True)
                keys = set(npzfile.files)
            except Exception:
                continue
            if 'cell_expr' in keys:
                return 'gene_expr'
            if 'cell_gen' in keys:
                return 'latent'
    raise ValueError(f'No .npz single-cell files found under {sc_npz_root}')


def load_sc_gene_expr_samples(sc_npz_root: str) -> Tuple[Dict[str, Dict], List[str]]:
    """Load reference-direct pseudo single-cell expression.

    Expected layout matches scDiffusion condgen:
      cells_2048_TCGA-.../<celltype>.npz

    Each npz should contain `cell_expr` (cells x genes) and `genes`.
    """
    sample_folders = [
        d
        for d in os.listdir(sc_npz_root)
        if os.path.isdir(os.path.join(sc_npz_root, d)) and d.startswith('cells_2048_TCGA-')
    ]
    samples_data: Dict[str, Dict] = {}
    gene_names: Optional[List[str]] = None

    for sample_folder in tqdm(sample_folders, total=len(sample_folders), desc='Loading SC gene expression'):
        sample_path = os.path.join(sc_npz_root, sample_folder)
        npz_files = [os.path.join(sample_path, f) for f in os.listdir(sample_path) if f.endswith('.npz')]
        cell_expr_list: List[np.ndarray] = []
        type_ids: List[np.ndarray] = []
        type_names: List[str] = []

        for file in npz_files:
            npzfile = np.load(file, allow_pickle=True)
            if 'cell_expr' not in npzfile:
                continue
            data_arr = np.asarray(npzfile['cell_expr'], dtype=np.float32)
            if data_arr.ndim != 2 or data_arr.shape[0] == 0:
                continue
            file_genes = [str(g) for g in npzfile['genes'].tolist()] if 'genes' in npzfile else None
            if file_genes is None:
                raise ValueError(f"reference-direct file missing genes array: {file}")
            if gene_names is None:
                gene_names = file_genes
            elif file_genes != gene_names:
                raise ValueError(f"Inconsistent gene order in reference-direct file: {file}")

            cell_expr_list.append(data_arr)
            base = os.path.basename(file)
            tname = str(npzfile['cell_type'].item()) if 'cell_type' in npzfile else os.path.splitext(base)[0]
            if tname not in type_names:
                type_names.append(tname)
            t_id = type_names.index(tname)
            type_ids.append(np.full((data_arr.shape[0],), t_id, dtype=np.int64))

        if len(cell_expr_list) > 0:
            X_concat = np.concatenate(cell_expr_list, axis=0)
            t_concat = np.concatenate(type_ids, axis=0) if len(type_ids) > 0 else None
            df = pd.DataFrame(X_concat, columns=gene_names)
            samples_data[sample_folder] = {
                'X_df': df,
                'type_ids': (t_concat.astype(np.int64) if t_concat is not None else None),
                'type_names': type_names,
            }

    if gene_names is None or len(samples_data) == 0:
        raise ValueError(f'No reference-direct cell_expr data found under {sc_npz_root}')
    return samples_data, gene_names


def decode_sc_to_deg_gene_space(
    sc_npz_root: str,
    gene_list_csv: str,
    deg_csv: str,
    device: torch.device,
    vae_ckpt_path: str,
    vae_num_genes: int,
) -> Tuple[Dict[str, Dict], List[str]]:
    """Decode per-sample cell embeddings to gene expression using pretrained VAE and
    immediately filter to DEG genes that are present in the model's gene list.

    Returns decoded_samples_filtered and final gene list (common genes).
    """
    # 1) Scan per-sample folders
    sample_folders = [
        d
        for d in os.listdir(sc_npz_root)
        if os.path.isdir(os.path.join(sc_npz_root, d)) and d.startswith('cells_2048_TCGA-')
    ]

    samples_data: Dict[str, Dict] = {}
    for sample_folder in tqdm(sample_folders, total=len(sample_folders), desc='Loading SC data'):
        sample_path = os.path.join(sc_npz_root, sample_folder)
        npz_files = [os.path.join(sample_path, f) for f in os.listdir(sample_path) if f.endswith('.npz')]

        cell_gen_list: List[np.ndarray] = []
        type_ids: List[np.ndarray] = []
        type_names: List[str] = []
        for file in npz_files:
            npzfile = np.load(file, allow_pickle=True)
            if 'cell_gen' in npzfile:
                data_arr = npzfile['cell_gen']
                cell_gen_list.append(data_arr)
                base = os.path.basename(file)
                tname = os.path.splitext(base)[0]
                if tname not in type_names:
                    type_names.append(tname)
                t_id = type_names.index(tname)
                type_ids.append(np.full((data_arr.shape[0],), t_id, dtype=np.int64))

        if len(cell_gen_list) > 0:
            X_concat = np.concatenate(cell_gen_list, axis=0)
            t_concat = np.concatenate(type_ids, axis=0) if len(type_ids) > 0 else None
            samples_data[sample_folder] = {
                'X': X_concat,
                'type_ids': t_concat,
                'type_names': type_names,
            }

    # 2) Load gene names and DEG list
    gene_order = pd.read_csv(gene_list_csv)
    gene_names_full = gene_order.iloc[:, 1].astype(str).tolist()
    deg_df = pd.read_csv(deg_csv, index_col=0)
    deg_genes = deg_df.index.astype(str).tolist()
    common_genes = [g for g in deg_genes if g in gene_names_full]
    if len(common_genes) == 0:
        raise ValueError('No overlapping DEG genes found in the model\'s gene list.')

    # 3) Decode each sample with VAE and filter to common_genes
    decoded_samples_filtered: Dict[str, Dict] = {}
    vae = load_vae(device, num_genes=int(vae_num_genes), ckpt_path=vae_ckpt_path)
    for sample_name, sd in tqdm(samples_data.items(), total=len(samples_data), desc='Decoding SC data'):
        X = sd['X']
        type_ids_np = sd.get('type_ids', None)
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
            decoded = vae(X_tensor, return_decoded=True).detach().cpu().numpy()
        df = pd.DataFrame(decoded, columns=gene_names_full)
        # Filter to DEGs present in model gene list
        df_f = df[common_genes]
        # Deduplicate ENSG columns by keeping the one with the largest mean across cells
        if df_f.columns.duplicated().any():
            cols_index = pd.Index(df_f.columns)
            dup_names = cols_index[cols_index.duplicated()].unique().tolist()
            # Map gene name to all column positions
            name_to_positions: Dict[str, List[int]] = {}
            for j, nm in enumerate(cols_index):
                name_to_positions.setdefault(str(nm), []).append(j)
            keep_positions: List[int] = []
            for nm, pos_list in name_to_positions.items():
                if len(pos_list) == 1:
                    keep_positions.append(pos_list[0])
                    continue
                # Compute mean per duplicate column, choose the best
                sub_means = df_f.iloc[:, pos_list].mean(axis=0)
                best_rel_idx = int(np.argmax(sub_means.values))
                best_abs_pos = pos_list[best_rel_idx]
                keep_positions.append(best_abs_pos)
            keep_positions = sorted(keep_positions)
            df_f = df_f.iloc[:, keep_positions]
        decoded_samples_filtered[sample_name] = {
            'X_df': df_f,
            'type_ids': (type_ids_np.astype(np.int64) if type_ids_np is not None else None),
            'type_names': sd.get('type_names', []),
        }

    # Also return a de-duplicated final gene list consistent with df_f columns
    final_gene_cols = (
        decoded_samples_filtered[next(iter(decoded_samples_filtered))]['X_df'].columns.tolist()
        if len(decoded_samples_filtered) > 0 else list(dict.fromkeys(common_genes))
    )
    return decoded_samples_filtered, final_gene_cols


def load_bulk_filtered(bulk_dir: str, final_gene_cols: List[str]) -> pd.DataFrame:
    """Load and merge bulk CSVs, filter to final_gene_cols and return a DataFrame."""
    bulk_files = [os.path.join(bulk_dir, f) for f in os.listdir(bulk_dir) if f.endswith('.csv')]
    bulk_frames: List[pd.DataFrame] = []
    for fp in tqdm(bulk_files, total=len(bulk_files), desc='Loading bulk data'):
        try:
            df = pd.read_csv(fp, index_col=0)
            bulk_frames.append(df)
        except Exception:
            continue
    if len(bulk_frames) == 0:
        raise ValueError(f'No bulk CSVs found under {bulk_dir}')
    bulk_all = pd.concat(bulk_frames, axis=0, sort=False)
    bulk_all = bulk_all[~bulk_all.index.duplicated(keep='first')]
    final_cols = [g for g in final_gene_cols if g in bulk_all.columns]
    if len(final_cols) == 0:
        raise ValueError('No overlapping bulk genes found between DEG list and bulk CSV columns.')
    bulk_all = bulk_all[final_cols].astype(float)
    return bulk_all


def load_bulk_all(bulk_dir: str) -> pd.DataFrame:
    """Load and merge all bulk CSVs without a gene filter."""
    bulk_files = [os.path.join(bulk_dir, f) for f in os.listdir(bulk_dir) if f.endswith('.csv')]
    bulk_frames: List[pd.DataFrame] = []
    for fp in tqdm(bulk_files, total=len(bulk_files), desc='Loading bulk data'):
        try:
            bulk_frames.append(pd.read_csv(fp, index_col=0))
        except Exception:
            continue
    if len(bulk_frames) == 0:
        raise ValueError(f'No bulk CSVs found under {bulk_dir}')
    bulk_all = pd.concat(bulk_frames, axis=0, sort=False)
    bulk_all = bulk_all[~bulk_all.index.duplicated(keep='first')]
    bulk_all = bulk_all.apply(pd.to_numeric, errors='coerce')
    bulk_all = bulk_all.dropna(axis=1, how='all').fillna(0.0)
    return bulk_all.astype(float)


def load_celltype_features(celltypes_csv: str) -> pd.DataFrame:
    """Load cell-type fractions as numeric patient-level features."""
    if not celltypes_csv or not os.path.exists(celltypes_csv):
        raise ValueError(f'Celltype fraction file not found: {celltypes_csv}')
    df = pd.read_csv(celltypes_csv, index_col=0)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    df = df[~df.index.duplicated(keep='first')]
    df.columns = [f'celltype:{c}' for c in df.columns.astype(str)]
    return df.astype(float)


def align_aux_features(
    patient_ids: List[str],
    aux_df: pd.DataFrame,
    id_map: Dict[str, str],
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """Align auxiliary patient features to internal patient IDs."""
    rows = []
    for pid in patient_ids:
        candidates = [pid]
        mapped = id_map.get(pid)
        if mapped is not None:
            candidates.append(mapped)
        tcga = extract_tcga_from_key(pid)
        if tcga is not None:
            candidates.append(tcga)
        match = None
        for key in candidates:
            if key in aux_df.index:
                match = key
                break
            starts = [idx for idx in aux_df.index if str(idx).startswith(str(key))]
            if starts:
                match = starts[0]
                break
        if match is None:
            rows.append(pd.Series(fill_value, index=aux_df.columns, dtype=float))
        else:
            rows.append(aux_df.loc[match].astype(float))
    out = pd.DataFrame(rows, index=patient_ids, columns=aux_df.columns)
    return out.fillna(fill_value).astype(float)


def build_pseudo_sc_samples(feature_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Represent patient-level features as a one-token bag for no-sc ablations."""
    samples: Dict[str, Dict[str, Any]] = {}
    for pid, row in feature_df.iterrows():
        samples[str(pid)] = {'X_df': pd.DataFrame([row.values], columns=feature_df.columns)}
    return samples


def map_ids_to_bulk(indices: List[str], bulk_all: pd.DataFrame) -> Dict[str, str]:
    """Map SC sample keys to bulk indices using TCGA barcode containment or exact match."""
    mapping: Dict[str, str] = {}
    for pid in indices:
        tcga = extract_tcga_from_key(pid)
        if tcga is not None:
            if tcga in bulk_all.index:
                mapping[pid] = tcga
                continue
            candidates = [idx for idx in bulk_all.index if str(idx).startswith(tcga)]
            if len(candidates) > 0:
                mapping[pid] = candidates[0]
                continue
        if pid in bulk_all.index:
            mapping[pid] = pid
    return mapping


def collect_targets_for_sc_keys(
    surv_df: pd.DataFrame,
    sc_keys: List[str],
) -> Tuple[List[str], Dict[str, float], Dict[str, int]]:
    """Collect survival targets for SC sample keys based on containment of TCGA barcode."""
    ids: List[str] = []
    durs: Dict[str, float] = {}
    evs: Dict[str, int] = {}
    for _, row in surv_df.iterrows():
        barcode = str(row['tcga_barcode'])
        matched_keys = [k for k in sc_keys if barcode in k]
        if len(matched_keys) == 0:
            continue
        pid = matched_keys[0]
        ids.append(pid)
        durs[pid] = float(row['duration'])
        evs[pid] = int(row['event'])
    return ids, durs, evs


def split_train_early_val_ids(
    train_ids: List[str],
    train_evs: Dict[str, int],
    val_fraction: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """Split the training fold into fit and early-stop validation IDs."""
    if not 0.0 < float(val_fraction) < 1.0:
        raise ValueError(f"early_stop_fraction must be between 0 and 1, got {val_fraction}")
    if len(train_ids) < 2:
        raise ValueError("Need at least two training samples to create an early-stop validation split.")

    rng = np.random.default_rng(seed)
    by_event: Dict[int, List[str]] = {}
    for pid in train_ids:
        by_event.setdefault(int(train_evs[pid]), []).append(pid)

    early_val_ids: List[str] = []
    fit_ids: List[str] = []
    for group_ids in by_event.values():
        shuffled = list(group_ids)
        rng.shuffle(shuffled)
        if len(shuffled) >= 2:
            n_val = int(round(len(shuffled) * float(val_fraction)))
            n_val = min(max(1, n_val), len(shuffled) - 1)
        else:
            n_val = 0
        early_val_ids.extend(shuffled[:n_val])
        fit_ids.extend(shuffled[n_val:])

    if len(early_val_ids) == 0:
        shuffled = list(train_ids)
        rng.shuffle(shuffled)
        n_val = min(max(1, int(round(len(shuffled) * float(val_fraction)))), len(shuffled) - 1)
        early_val_ids = shuffled[:n_val]
        fit_ids = shuffled[n_val:]

    fit_set = set(fit_ids)
    early_val_set = set(early_val_ids)
    return (
        [pid for pid in train_ids if pid in fit_set],
        [pid for pid in train_ids if pid in early_val_set],
    )


@torch.no_grad()
def collect_model_predictions(
    model: BulkSCFusionSurvival,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Collect risk scores, optional logits, durations, and events from a loader."""
    model.eval()
    risks: List[np.ndarray] = []
    logits_list: List[np.ndarray] = []
    durations: List[np.ndarray] = []
    events: List[np.ndarray] = []
    for batch in loader:
        x_bulk = batch['x_bulk'].to(device)
        x_sc = batch['x_sc'].to(device)
        mask = batch['mask'].to(device)
        risk, _, _, _, logits = model.forward_outputs(x_bulk, x_sc, mask, return_logits=True)
        risks.append(risk.detach().cpu().numpy().reshape(-1))
        if logits is not None:
            logits_list.append(logits.detach().cpu().numpy())
        durations.append(batch['duration'].cpu().numpy().reshape(-1))
        events.append(batch['event'].cpu().numpy().reshape(-1))
    return {
        'risk': np.concatenate(risks, axis=0) if risks else np.array([], dtype=float),
        'logits': np.concatenate(logits_list, axis=0) if logits_list else np.empty((0, 0), dtype=float),
        'duration': np.concatenate(durations, axis=0) if durations else np.array([], dtype=float),
        'event': np.concatenate(events, axis=0).astype(int) if events else np.array([], dtype=int),
    }


def harrell_cindex_from_arrays(risk: np.ndarray, durations: np.ndarray, events: np.ndarray) -> float:
    """Compute the same held-out Harrell C-index convention used by evaluate_cindex."""
    risk = np.asarray(risk, dtype=np.float64)
    durations = np.asarray(durations, dtype=np.float64)
    events = np.asarray(events, dtype=np.int32)
    order = np.argsort(durations)
    durations = durations[order]
    events = events[order]
    risk = risk[order]
    num = 0.0
    den = 0
    for i in range(len(durations)):
        if events[i] == 0:
            continue
        for j in range(i + 1, len(durations)):
            if durations[j] > durations[i]:
                den += 1
                if risk[i] > risk[j]:
                    num += 1.0
                elif risk[i] == risk[j]:
                    num += 0.5
    return float(num / den) if den > 0 else float('nan')


def build_survival_dataframe(
    loss_fn: str,
    train_pred: Dict[str, np.ndarray],
    pred: Dict[str, np.ndarray],
    bin_edges: Optional[np.ndarray],
) -> Optional[pd.DataFrame]:
    """Build survival curves for pycox EvalSurv from Cox or discrete-time outputs."""
    loss_fn = str(loss_fn).lower()
    if pred['risk'].size == 0:
        return None
    if loss_fn == 'cox':
        train_durations = np.asarray(train_pred['duration'], dtype=np.float64)
        train_events = np.asarray(train_pred['event'], dtype=np.int64)
        train_risk = np.asarray(train_pred['risk'], dtype=np.float64)
        event_mask = train_events > 0
        event_times = np.unique(train_durations[event_mask])
        event_times = event_times[np.isfinite(event_times)]
        event_times.sort()
        if event_times.size == 0:
            return None
        exp_train = np.exp(np.clip(train_risk, -50.0, 50.0))
        cum_hazards = []
        cumulative = 0.0
        for t in event_times:
            at_risk = exp_train[train_durations >= t]
            risk_sum = at_risk.sum()
            if risk_sum <= 0:
                continue
            d_i = train_events[(train_durations == t) & event_mask].sum()
            if d_i <= 0:
                continue
            cumulative += float(d_i) / max(float(risk_sum), 1e-12)
            cum_hazards.append(cumulative)
        if len(cum_hazards) == 0:
            return None
        pred_exp = np.exp(np.clip(np.asarray(pred['risk'], dtype=np.float64), -50.0, 50.0))
        surv_matrix = np.exp(-np.outer(pred_exp, np.asarray(cum_hazards, dtype=np.float64)))
        surv_matrix = np.hstack([np.ones((pred_exp.shape[0], 1)), surv_matrix])
        timeline = np.concatenate(([0.0], event_times.astype(np.float64)))
        return pd.DataFrame(surv_matrix.T, index=timeline)

    if bin_edges is None or pred['logits'].size == 0:
        return None
    logits = np.asarray(pred['logits'], dtype=np.float64)
    if loss_fn == 'deephit':
        logits = logits - np.max(logits, axis=1, keepdims=True)
        probs = np.exp(logits)
        probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
        surv_matrix = 1.0 - np.cumsum(probs, axis=1)
    else:
        hazards = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
        hazards = np.clip(hazards, 1e-6, 1.0 - 1e-6)
        surv_matrix = np.cumprod(1.0 - hazards, axis=1)
    surv_matrix = np.clip(surv_matrix, 0.0, 1.0)
    surv_matrix = np.hstack([np.ones((surv_matrix.shape[0], 1)), surv_matrix])
    timeline = np.asarray(bin_edges, dtype=np.float64)
    if timeline.shape[0] != surv_matrix.shape[1]:
        return None
    return pd.DataFrame(surv_matrix.T, index=timeline)


def compute_integrated_brier_score(
    surv_df: Optional[pd.DataFrame],
    durations: np.ndarray,
    events: np.ndarray,
) -> float:
    """Compute IBS with pycox EvalSurv; return NaN when the fold is not estimable."""
    if surv_df is None or len(surv_df.index) < 2:
        return float('nan')
    durations = np.asarray(durations, dtype=np.float64)
    events = np.asarray(events, dtype=np.int64)
    times = surv_df.index.values.astype(np.float64)
    positive = times[times > 0]
    if positive.size == 0:
        return float('nan')
    lower = float(positive[0])
    upper = float(times[-1])
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return float('nan')
    grid = times[(times >= lower) & (times <= upper)]
    if grid.size < 2:
        return float('nan')
    try:
        from pycox.evaluation import EvalSurv
        eval_surv = EvalSurv(surv_df, durations, events, censor_surv='km')
        bs = eval_surv.brier_score(grid)
        return float(np.trapz(bs.values.astype(np.float64), x=grid) / (grid[-1] - grid[0]))
    except Exception:
        return float('nan')


def compute_td_auc(
    train_pred: Dict[str, np.ndarray],
    pred: Dict[str, np.ndarray],
    num_times: int = 20,
) -> float:
    """Mean cumulative/dynamic tdAUC using scikit-survival when available."""
    try:
        from sksurv.metrics import cumulative_dynamic_auc
    except Exception:
        return float('nan')
    train_d = np.asarray(train_pred['duration'], dtype=np.float64)
    train_e = np.asarray(train_pred['event'], dtype=bool)
    test_d = np.asarray(pred['duration'], dtype=np.float64)
    test_e = np.asarray(pred['event'], dtype=bool)
    risk = np.asarray(pred['risk'], dtype=np.float64)
    if train_d.size == 0 or test_d.size == 0 or not np.any(test_e):
        return float('nan')
    event_times = np.unique(test_d[test_e])
    lower = max(float(np.min(test_d)), float(np.min(train_d)))
    upper = min(float(np.max(test_d)), float(np.max(train_d)))
    times = event_times[(event_times > lower) & (event_times < upper)]
    if times.size > int(num_times):
        qs = np.linspace(0.05, 0.95, int(num_times))
        times = np.unique(np.quantile(times, qs))
    if times.size == 0:
        return float('nan')
    y_train = np.array(list(zip(train_e, train_d)), dtype=[('event', '?'), ('time', '<f8')])
    y_test = np.array(list(zip(test_e, test_d)), dtype=[('event', '?'), ('time', '<f8')])
    try:
        _, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk, times)
        return float(mean_auc)
    except Exception:
        return float('nan')


def evaluate_survival_metrics(
    model: BulkSCFusionSurvival,
    train_loader: torch.utils.data.DataLoader,
    eval_loader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_fn: str,
    bin_edges: Optional[np.ndarray],
) -> Dict[str, float]:
    train_pred = collect_model_predictions(model, train_loader, device)
    eval_pred = collect_model_predictions(model, eval_loader, device)
    surv_df = build_survival_dataframe(loss_fn, train_pred, eval_pred, bin_edges)
    return {
        'c_index': harrell_cindex_from_arrays(eval_pred['risk'], eval_pred['duration'], eval_pred['event']),
        'td_auc': compute_td_auc(train_pred, eval_pred),
        'integrated_brier_score': compute_integrated_brier_score(surv_df, eval_pred['duration'], eval_pred['event']),
    }


def run_fold(
    fold_num: int,
    decoded_samples_filtered: Dict[str, Dict],
    bulk_all: pd.DataFrame,
    final_gene_cols: List[str],
    args: argparse.Namespace,
    device: torch.device,
    results_dir: str,
    deg_dir: Optional[str] = None,
    celltype_all: Optional[pd.DataFrame] = None,
    quiet: bool = False,
    epoch_progress_position: int = 1,
) -> Dict[str, float]:
    """Run one outer CV fold.

    The outer fold's val_data_{k}.csv is treated as held-out test data.
    Best epoch selection uses an early-stop validation split carved only from
    train_data_{k}.csv.
    """
    # 0) Load survival labels first so we know which samples we need (avoids copying all 1111 samples)
    train_surv, test_surv = load_survival_fold(fold_num, args.surv_label_dir)
    sc_keys = list(decoded_samples_filtered.keys())
    train_ids, train_durs, train_evs = collect_targets_for_sc_keys(train_surv, sc_keys)
    test_ids, test_durs, test_evs = collect_targets_for_sc_keys(test_surv, sc_keys)
    train_ids = [pid for pid in train_ids if pid in decoded_samples_filtered]
    test_ids = [pid for pid in test_ids if pid in decoded_samples_filtered]

    # 1) Per-fold DEG: filter to common genes, then only copy samples needed for this fold
    if deg_dir is not None:
        deg_path = os.path.join(deg_dir, f'degs_fold{fold_num}.csv')
        if os.path.exists(deg_path):
            deg_df = pd.read_csv(deg_path, index_col=0)
            deg_genes_raw = deg_df.index.astype(str).tolist()
            if 'symbol' in deg_df.columns:
                deg_genes_raw = deg_df['symbol'].astype(str).tolist()
            common_genes = [g for g in deg_genes_raw if g in bulk_all.columns]
            gene_list_path = getattr(args, 'gene_list_path', '')
            if len(common_genes) == 0 and gene_list_path and os.path.exists(gene_list_path):
                gl_df = pd.read_csv(gene_list_path)
                if len(gl_df.columns) >= 3:
                    ensg_to_symbol = dict(zip(gl_df.iloc[:, 1].astype(str), gl_df.iloc[:, 2].astype(str)))
                    def ensg_to_sym(g):
                        s = ensg_to_symbol.get(g)
                        if s is None and '.' in g:
                            s = ensg_to_symbol.get(g.split('.')[0], g)
                        return s if s is not None else g
                    deg_symbols = [ensg_to_sym(g) for g in deg_genes_raw]
                    common_genes = [g for g in deg_symbols if g in bulk_all.columns]
            if len(common_genes) > 0:
                if len(decoded_samples_filtered) > 0:
                    first_key = next(iter(decoded_samples_filtered))
                    sc_gene_set = set(decoded_samples_filtered[first_key]['X_df'].columns.astype(str))
                    common_genes = [g for g in common_genes if g in sc_gene_set]
            if len(common_genes) > 0:
                final_gene_cols = [c for c in common_genes if c in bulk_all.columns]
                bulk_all = bulk_all[final_gene_cols].copy()
                needed_ids = set(train_ids) | set(test_ids)
                decoded_samples_fold = {}
                for k in needed_ids:
                    if k not in decoded_samples_filtered:
                        continue
                    df = decoded_samples_filtered[k]['X_df']
                    cols = [c for c in final_gene_cols if c in df.columns]
                    decoded_samples_fold[k] = {
                        **decoded_samples_filtered[k],
                        'X_df': df[cols].copy(),
                    }
                decoded_samples_filtered = decoded_samples_fold
                _progress_write(
                    f"  Fold {fold_num}: DEG filter -> {len(final_gene_cols)} genes, {len(decoded_samples_fold)} samples"
                )
            else:
                _progress_write(
                    f"  WARNING: Fold {fold_num} DEG filter produced 0 common genes (check ENSG/symbol mapping)"
                )
    else:
        # No deg_dir: still restrict to needed samples to save RAM
        needed_ids = set(train_ids) | set(test_ids)
        decoded_samples_filtered = {k: v for k, v in decoded_samples_filtered.items() if k in needed_ids}

    # 2) Map SC ids to bulk indices and filter
    tr_map = map_ids_to_bulk(train_ids, bulk_all)
    te_map = map_ids_to_bulk(test_ids, bulk_all)
    train_ids = [pid for pid in train_ids if pid in tr_map and tr_map[pid] in bulk_all.index]
    test_ids = [pid for pid in test_ids if pid in te_map and te_map[pid] in bulk_all.index]

    fit_ids, early_val_ids = split_train_early_val_ids(
        train_ids,
        train_evs,
        val_fraction=float(args.early_stop_fraction),
        seed=int(args.seed) + int(fold_num),
    )

    # 3) Reindex bulk matrices to SC ids
    bulk_fit = bulk_all.loc[[tr_map[pid] for pid in fit_ids]].copy()
    bulk_fit.index = fit_ids
    bulk_early_val = bulk_all.loc[[tr_map[pid] for pid in early_val_ids]].copy()
    bulk_early_val.index = early_val_ids
    bulk_test = bulk_all.loc[[te_map[pid] for pid in test_ids]].copy()
    bulk_test.index = test_ids

    if args.input_mode == 'bulk_celltype':
        if celltype_all is None:
            raise ValueError("input_mode=bulk_celltype requires --celltypes_csv or a config 'celltypes' entry.")
        bulk_fit = pd.concat([bulk_fit, align_aux_features(fit_ids, celltype_all, tr_map)], axis=1)
        bulk_early_val = pd.concat([bulk_early_val, align_aux_features(early_val_ids, celltype_all, tr_map)], axis=1)
        bulk_test = pd.concat([bulk_test, align_aux_features(test_ids, celltype_all, te_map)], axis=1)
        final_gene_cols = bulk_fit.columns.astype(str).tolist()
    elif args.input_mode == 'bulk':
        final_gene_cols = bulk_fit.columns.astype(str).tolist()

    if args.input_mode in ('bulk', 'bulk_celltype'):
        pseudo_features = pd.concat([bulk_fit, bulk_early_val, bulk_test], axis=0)
        decoded_samples_filtered = build_pseudo_sc_samples(pseudo_features)

    # 6) Narrow SC dataframes to final_gene_cols (safety)
    for k in decoded_samples_filtered.keys():
        df = decoded_samples_filtered[k]['X_df']
        if list(df.columns) != final_gene_cols:
            decoded_samples_filtered[k]['X_df'] = df[final_gene_cols]

    # 7) Build datasets and loaders
    train_ds = PatientBatchDataset(
        fit_ids,
        decoded_samples_filtered,
        bulk_fit,
        {pid: train_durs[pid] for pid in fit_ids},
        {pid: train_evs[pid] for pid in fit_ids},
        max_cells=args.max_cells,
        seed=args.seed,
    )
    early_val_ds = PatientBatchDataset(
        early_val_ids,
        decoded_samples_filtered,
        bulk_early_val,
        {pid: train_durs[pid] for pid in early_val_ids},
        {pid: train_evs[pid] for pid in early_val_ids},
        max_cells=args.max_cells,
        seed=args.seed,
    )
    test_ds = PatientBatchDataset(
        test_ids,
        decoded_samples_filtered,
        bulk_test,
        test_durs,
        test_evs,
        max_cells=args.max_cells,
        seed=args.seed,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=pad_collate_fn, drop_last=False
    )
    early_val_loader = torch.utils.data.DataLoader(
        early_val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=pad_collate_fn, drop_last=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=pad_collate_fn, drop_last=False
    )

    # 8) Build model
    num_genes = len(final_gene_cols)
    # Parse hidden dims from CLI strings
    def _parse_dims(s: str) -> Optional[List[int]]:
        s = str(s).strip()
        if s == '' or s.lower() == 'auto':
            return None
        vals = [v for v in s.split(',') if v.strip() != '']
        if len(vals) == 1 and vals[0].strip() == '0':
            return []
        return [int(v) for v in vals]

    # Determine prediction head structure
    pred_head_hidden = _parse_dims(args.pred_head_hidden)
    if bool(args.direct_cox_from_fusion):
        # No hidden layers: direct Linear from fusion/head input to risk
        pred_head_hidden = []

    model = BulkSCFusionSurvival(
        num_genes=num_genes,
        embed_dim=args.embed_dim,
        cell_encoder_hidden=[512, 256],
        num_attention_heads=args.num_heads,
        num_attention_layers=args.num_layers,
        dropout=args.dropout,
        bulk_encoder_type=args.bulk_encoder,
        cell_encoder_type=args.cell_encoder,
        fusion_type=args.fusion_type,
        pooling=args.pooling,
        risk_head=args.risk_head,
        output_mode=args.loss_fn,
        num_time_bins=args.num_time_bins,
        concat_sc_with_fusion=bool(args.concat_sc_with_fusion),
        residual_hfuse_to_bulk=bool(args.residual_hfuse_to_bulk),
        bulk_mlp_hidden=_parse_dims(args.bulk_mlp_hidden),
        bulk_mlp_dropout=float(args.bulk_mlp_dropout),
        fusion_mlp_hidden=_parse_dims(args.fusion_mlp_hidden),
        pred_head_hidden=pred_head_hidden,
        use_sc=(args.input_mode == 'bulk_scgep'),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Cosine schedule with 5% warmup
    total_steps = max(1, args.epochs * max(1, math.ceil(len(train_loader))))
    warmup_steps = max(1, int(0.05 * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss weights
    weights = {'cox': 1.0, 'nce': args.alpha_nce, 'match': args.beta_match, 'mask': args.gamma_mask}
    # Apply ablation presets
    if args.ablation == 'no_align':
        weights['nce'] = 0.0
        weights['match'] = 0.0
    elif args.ablation == 'nce_only':
        weights['match'] = 0.0
    elif args.ablation == 'match_only':
        weights['nce'] = 0.0
    elif args.ablation == 'no_mask':
        weights['mask'] = 0.0
    elif args.ablation == 'all':
        weights['nce'] = 0.0
        weights['match'] = 0.0
        weights['mask'] = 0.0
    if args.input_mode in ('bulk', 'bulk_celltype'):
        weights['nce'] = 0.0
        weights['match'] = 0.0
        weights['mask'] = 0.0

    # Prepare bin edges for discrete-time losses
    bin_edges_np = None
    if args.loss_fn in ('deephit', 'mtlr'):
        durs_arr = np.array([train_durs[pid] for pid in fit_ids], dtype=np.float32)
        bin_edges_np = make_time_bins(durs_arr, num_bins=args.num_time_bins, strategy=args.time_bins_strategy)

    # Train
    fold_dir = os.path.join(results_dir, f'fold{fold_num}')
    os.makedirs(fold_dir, exist_ok=True)
    best_early_val_cindex = -1.0
    best_epoch = -1
    best_path = os.path.join(fold_dir, 'model.pt')
    saved_best = False
    history: List[Dict[str, float]] = []

    epoch_desc = f'Fold {fold_num} epochs'
    epoch_progress = tqdm(
        range(1, args.epochs + 1),
        total=args.epochs,
        desc=epoch_desc,
        position=epoch_progress_position,
        leave=False,
        dynamic_ncols=True,
        disable=False,
    )
    try:
        for epoch in epoch_progress:
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                weights,
                gene_idx_tensor=None,
                loss_type=args.loss_fn,
                bin_edges=bin_edges_np,
                scheduler=scheduler,
            )

            early_val_c = evaluate_cindex(model, early_val_loader, device)
            log_row = {
                'epoch': epoch,
                **train_metrics,
                'early_val_c_index': float(early_val_c) if np.isfinite(early_val_c) else float('nan'),
                # Backward-compatible alias; this is now the validation split from train_data_k.
                'val_c_index': float(early_val_c) if np.isfinite(early_val_c) else float('nan'),
                'lr': float(optimizer.param_groups[0]['lr']),
            }
            history.append(log_row)

            val_c_display = log_row['early_val_c_index']
            epoch_progress.set_postfix({
                'train_loss': f"{train_metrics['loss']:.4f}",
                'early_val_c': (f"{val_c_display:.4f}" if np.isfinite(val_c_display) else 'nan'),
                'lr': f"{log_row['lr']:.2e}",
            })
            if not quiet:
                _progress_write(json.dumps({'fold': fold_num, **log_row}))

            if np.isfinite(early_val_c) and float(early_val_c) > best_early_val_cindex:
                best_early_val_cindex = float(early_val_c)
                best_epoch = int(epoch)
                torch.save(
                    {
                        'model': model.state_dict(),
                        'epoch': epoch,
                        'early_val_c_index': best_early_val_cindex,
                        # Backward-compatible alias; this is not the outer held-out test C-index.
                        'val_c_index': best_early_val_cindex,
                        'args': vars(args),
                        'final_gene_cols': final_gene_cols,
                        'split_counts': {
                            'train_fit': len(fit_ids),
                            'early_val': len(early_val_ids),
                            'test': len(test_ids),
                        },
                    },
                    best_path,
                )
                saved_best = True
    finally:
        epoch_progress.close()

    if not saved_best:
        best_early_val_cindex = float('nan')
        best_epoch = int(args.epochs)
        torch.save(
            {
                'model': model.state_dict(),
                'epoch': best_epoch,
                'early_val_c_index': float('nan'),
                'val_c_index': float('nan'),
                'args': vars(args),
                'final_gene_cols': final_gene_cols,
                'split_counts': {
                    'train_fit': len(fit_ids),
                    'early_val': len(early_val_ids),
                    'test': len(test_ids),
                },
            },
            best_path,
        )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    early_val_metrics = evaluate_survival_metrics(
        model, train_loader, early_val_loader, device, args.loss_fn, bin_edges_np
    )
    test_metrics = evaluate_survival_metrics(
        model, train_loader, test_loader, device, args.loss_fn, bin_edges_np
    )
    test_cindex = test_metrics['c_index']

    # Persist history and plot losses
    with open(os.path.join(fold_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    try:
        plot_losses_from_history(history, save_path=os.path.join(fold_dir, 'losses.png'))
    except Exception:
        pass

    # Force GC to free fold-specific data (model, loaders, decoded_samples_fold) before next fold
    del model, optimizer, scheduler, train_loader, early_val_loader, test_loader, train_ds, early_val_ds, test_ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'fold': float(fold_num),
        'best_epoch': float(best_epoch),
        'best_early_val_c_index': float(best_early_val_cindex),
        'early_val_c_index': float(early_val_metrics['c_index']),
        'early_val_td_auc': float(early_val_metrics['td_auc']),
        'early_val_integrated_brier_score': float(early_val_metrics['integrated_brier_score']),
        'test_c_index': float(test_cindex) if np.isfinite(test_cindex) else float('nan'),
        'test_td_auc': float(test_metrics['td_auc']),
        'test_integrated_brier_score': float(test_metrics['integrated_brier_score']),
        'feature_count': float(len(final_gene_cols)),
        'train_fit_samples': float(len(fit_ids)),
        'early_val_samples': float(len(early_val_ids)),
        'test_samples': float(len(test_ids)),
    }


def main(argv: Optional[List[str]] = None):
    _default_config = str(_DESCENT_ROOT / "config" / "path_local.json")
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=_default_config, help='Path to config json')
    parser.add_argument('--cancer', type=str, default=None, help='Cancer type; when set, load paths from config')
    parser.add_argument('--sc_npz_root', type=str, default=None)
    parser.add_argument('--gene_list_csv', type=str, default=None)
    parser.add_argument('--deg_csv', type=str, default=None)
    parser.add_argument('--deg_dir', type=str, default=None, help='Dir with degs_fold{k}.csv for per-fold DEG (overrides deg_csv)')
    parser.add_argument('--bulk_dir', type=str, default=None)
    parser.add_argument('--celltypes_csv', type=str, default=None, help='Cell-type fraction CSV for input_mode=bulk_celltype')
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--surv_label_dir', type=str, default=None)
    parser.add_argument('--vae_ckpt_path', type=str, default=None)
    parser.add_argument('--vae_num_genes', type=int, default=28952)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--quiet', action='store_true', help='Suppress per-epoch JSON logs while keeping fold/epoch tqdm progress bars.')
    parser.add_argument('--input_mode', type=str, default='bulk_scgep', choices=['bulk_scgep', 'bulk', 'bulk_celltype'], help='Model inputs: full bulk+scGEP, bulk only, or bulk plus cell-type fractions.')
    parser.add_argument('--sc_input_format', type=str, default='auto', choices=['auto', 'latent', 'gene_expr'], help='SC input format for input_mode=bulk_scgep: scDiffusion latent npz or direct gene-expression npz.')
    parser.add_argument('--max_cells', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--alpha_nce', type=float, default=1.0)
    parser.add_argument('--beta_match', type=float, default=0.5)
    parser.add_argument('--gamma_mask', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--bulk_encoder', type=str, default='vae_mlp', choices=['vae_mlp', 'light_mlp'])
    parser.add_argument('--cell_encoder', type=str, default='mil', choices=['mil', 'symmetric'])
    # Ablation controls
    parser.add_argument('--risk_head', type=str, default='mlp', choices=['mlp', 'cosine'])
    parser.add_argument('--fusion_type', type=str, default='cross_attn', choices=['cross_attn', 'concat_mlp'])
    parser.add_argument('--pooling', type=str, default='attn', choices=['attn', 'mean'])
    parser.add_argument('--ablation', type=str, default='none', choices=['none', 'no_align', 'nce_only', 'match_only', 'no_mask', 'all'])
    # Survival loss controls
    parser.add_argument('--loss_fn', type=str, default='cox', choices=['cox', 'deephit', 'mtlr'])
    parser.add_argument('--num_time_bins', type=int, default=60)
    parser.add_argument('--time_bins_strategy', type=str, default='quantile', choices=['quantile', 'uniform'])
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--early_stop_fraction', type=float, default=0.2, help='Fraction of each training fold used for best-epoch selection')
    # Fusion head input augmentation
    parser.add_argument('--concat_sc_with_fusion', action='store_true', help='Concat pooled cell vector with fusion vector before prediction heads')
    # Residual addition of h_fuse into h_bulk for alignment/cosine-risk
    parser.add_argument('--residual_hfuse_to_bulk', action='store_true', help='Add h_fuse as residual to h_bulk for alignment and cosine head')
    # New: configurable MLP structures
    parser.add_argument('--bulk_mlp_hidden', type=str, default='512,256', help='Comma-separated hidden dims for light_mlp bulk encoder (e.g., 1024,512). Empty uses defaults')
    parser.add_argument('--bulk_mlp_dropout', type=float, default=0.2, help='Dropout for light_mlp bulk encoder hidden layers')
    parser.add_argument('--fusion_mlp_hidden', type=str, default='auto', help="Hidden dims for concat_mlp fusion (input is 2*embed_dim). Use 'auto' for original [2d→d]")
    parser.add_argument('--pred_head_hidden', type=str, default='auto', help="Hidden dims for risk head MLP. Use 'auto' for original [head_in_dim//2]")
    parser.add_argument('--direct_cox_from_fusion', action='store_true', help='Compute Cox risk directly from fusion/head input via a single Linear (no hidden MLP)')
    args = parser.parse_args(argv)

    # Load paths from config if --cancer is set
    if args.cancer is not None:
        import json
        deg_from_config = args.deg_csv is None
        with open(args.config) as f:
            cfg = json.load(f)
        c = cfg[args.cancer.upper()]
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(args.config)))
        def resolve(p):
            if not p:
                return p
            return os.path.normpath(os.path.join(project_root, p)) if not os.path.isabs(p) else p
        args.sc_npz_root = args.sc_npz_root or resolve(c['sc_npz'])
        args.gene_list_csv = args.gene_list_csv or resolve(c['gene_list'])
        config_deg_dir = resolve(c.get('deg_dir', '')) if c.get('deg_dir') else ''
        if args.deg_dir is None and config_deg_dir:
            args.deg_dir = config_deg_dir
        if args.deg_dir is not None:
            args.deg_csv = ''
        else:
            args.deg_csv = args.deg_csv or resolve(c.get('deg', '')) or ""
            if deg_from_config:
                args.deg_csv = resolve_existing_deg_csv(args.deg_csv, args.cancer)
        args.bulk_dir = args.bulk_dir or resolve(c['bulk'])
        args.celltypes_csv = args.celltypes_csv or resolve(c.get('celltypes', ''))
        args.surv_label_dir = args.surv_label_dir or resolve(c['surv_label'])
        args.vae_ckpt_path = args.vae_ckpt_path or resolve(c['VAE'])
        args.gene_list_path = resolve(c.get('gene_list_path', '')) if c.get('gene_list_path') else ''
        args.results_dir = args.results_dir or str(_DESCENT_ROOT / "output" / "survival_cv" / args.cancer)
        # Infer vae_num_genes from gene_list (required for BRCA etc. with different gene counts)
        gl_path = args.gene_list_csv or c.get('gene_list')
        if gl_path and os.path.exists(gl_path):
            gl_df = pd.read_csv(gl_path)
            args.vae_num_genes = len(gl_df)
    else:
        # Fallback defaults when no config
        args.sc_npz_root = args.sc_npz_root or str(_DESCENT_ROOT / "output" / "cell_embs_2048")
        args.gene_list_csv = args.gene_list_csv or str(_DESCENT_ROOT / "scgep_generation" / "VAE" / "28952genes.csv")
        args.deg_csv = args.deg_csv or ""
        args.bulk_dir = args.bulk_dir or str(_DESCENT_ROOT / "data" / "bulk")
        args.celltypes_csv = args.celltypes_csv or ""
        args.surv_label_dir = args.surv_label_dir or str(_DESCENT_ROOT / "data" / "surv_label")
        args.vae_ckpt_path = args.vae_ckpt_path or ""
        args.results_dir = args.results_dir or str(_DESCENT_ROOT / "output" / "survival_cv")

    print(args)
    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    use_per_fold_deg = args.deg_dir is not None
    celltype_all = load_celltype_features(args.celltypes_csv) if args.input_mode == 'bulk_celltype' else None
    if args.input_mode == 'bulk_scgep':
        sc_input_format = args.sc_input_format
        if sc_input_format == 'auto':
            sc_input_format = detect_sc_input_format(args.sc_npz_root)
        print(f'Using single-cell input format: {sc_input_format}')
        if use_per_fold_deg:
            # Per-fold DEG filtering happens in run_fold.
            if sc_input_format == 'gene_expr':
                print('Loading reference-direct gene expression (per-fold DEG)...')
                decoded_samples_filtered, gene_names_full = load_sc_gene_expr_samples(args.sc_npz_root)
            else:
                print('Decoding single-cell embeddings to full gene space (per-fold DEG)...')
                decoded_samples_filtered, gene_names_full = decode_sc_to_full_gene_space(
                    sc_npz_root=args.sc_npz_root,
                    gene_list_csv=args.gene_list_csv,
                    device=device,
                    vae_ckpt_path=args.vae_ckpt_path,
                    vae_num_genes=args.vae_num_genes,
                )
            final_gene_cols = gene_names_full
            bulk_all = load_bulk_filtered(args.bulk_dir, final_gene_cols)
        else:
            # Single DEG file.
            if not args.deg_csv:
                raise ValueError("Set --deg_dir or provide a config entry with 'deg'.")
            if sc_input_format == 'gene_expr':
                print('Loading reference-direct gene expression and filtering through bulk/DEG intersection...')
                decoded_samples_filtered, gene_names_full = load_sc_gene_expr_samples(args.sc_npz_root)
                deg_df = pd.read_csv(args.deg_csv, index_col=0)
                candidate_lists = []
                if 'symbol' in deg_df.columns:
                    candidate_lists.append(deg_df['symbol'].astype(str).tolist())
                candidate_lists.append(deg_df.index.astype(str).tolist())
                bulk_all_unfiltered = load_bulk_all(args.bulk_dir)
                sc_gene_set = set(gene_names_full)
                final_gene_cols = []
                for candidates in candidate_lists:
                    final_gene_cols = [g for g in candidates if g in sc_gene_set and g in bulk_all_unfiltered.columns]
                    if len(final_gene_cols) > 0:
                        break
                if len(final_gene_cols) == 0:
                    raise ValueError(f"No overlapping genes among {args.deg_csv}, reference-direct scGEP, and bulk columns.")
                for k in decoded_samples_filtered.keys():
                    decoded_samples_filtered[k]['X_df'] = decoded_samples_filtered[k]['X_df'][final_gene_cols].copy()
                bulk_all = bulk_all_unfiltered[final_gene_cols].copy()
            else:
                print('Decoding single-cell embeddings to gene space and filtering to DEGs...')
                decoded_samples_filtered, final_gene_cols = decode_sc_to_deg_gene_space(
                    sc_npz_root=args.sc_npz_root,
                    gene_list_csv=args.gene_list_csv,
                    deg_csv=args.deg_csv,
                    device=device,
                    vae_ckpt_path=args.vae_ckpt_path,
                    vae_num_genes=args.vae_num_genes,
                )
                bulk_all = load_bulk_filtered(args.bulk_dir, final_gene_cols)
    else:
        print(f'Loading bulk expression for input_mode={args.input_mode}...')
        if use_per_fold_deg:
            bulk_all = load_bulk_all(args.bulk_dir)
            final_gene_cols = bulk_all.columns.astype(str).tolist()
        else:
            if not args.deg_csv:
                raise ValueError("Set --deg_dir or provide a config entry with 'deg'.")
            deg_df = pd.read_csv(args.deg_csv, index_col=0)
            deg_genes = deg_df['symbol'].astype(str).tolist() if 'symbol' in deg_df.columns else deg_df.index.astype(str).tolist()
            bulk_all_unfiltered = load_bulk_all(args.bulk_dir)
            final_gene_cols = [g for g in deg_genes if g in bulk_all_unfiltered.columns]
            if len(final_gene_cols) == 0:
                raise ValueError(f"No overlapping genes between {args.deg_csv} and bulk columns.")
            bulk_all = bulk_all_unfiltered[final_gene_cols].copy()
        decoded_samples_filtered = build_pseudo_sc_samples(bulk_all)

    print('Loading bulk expression tables and aligning gene columns...')

    # Run folds
    fold_metrics: List[Dict[str, float]] = []
    fold_progress = tqdm(
        range(1, args.num_folds + 1),
        total=args.num_folds,
        desc='Running folds',
        position=0,
        leave=True,
        dynamic_ncols=True,
    )
    try:
        for fold in fold_progress:
            metrics = run_fold(
                fold_num=fold,
                decoded_samples_filtered=decoded_samples_filtered,
                bulk_all=bulk_all,
                final_gene_cols=final_gene_cols,
                args=args,
                device=device,
                results_dir=args.results_dir,
                deg_dir=args.deg_dir,
                celltype_all=celltype_all,
                quiet=args.quiet,
                epoch_progress_position=1,
            )
            fold_metrics.append(metrics)
            test_cindex = metrics['test_c_index']
            fold_progress.set_postfix({
                'test_c': (f"{test_cindex:.4f}" if np.isfinite(test_cindex) else 'nan'),
            })
    finally:
        fold_progress.close()

    # Aggregate
    valid_test_cis = [m['test_c_index'] for m in fold_metrics if np.isfinite(m['test_c_index'])]
    mean_test_ci = float(np.mean(valid_test_cis)) if len(valid_test_cis) > 0 else float('nan')
    std_test_ci = float(np.std(valid_test_cis, ddof=0)) if len(valid_test_cis) > 0 else float('nan')
    valid_test_td_auc = [m['test_td_auc'] for m in fold_metrics if np.isfinite(m.get('test_td_auc', float('nan')))]
    mean_test_td_auc = float(np.mean(valid_test_td_auc)) if len(valid_test_td_auc) > 0 else float('nan')
    std_test_td_auc = float(np.std(valid_test_td_auc, ddof=0)) if len(valid_test_td_auc) > 0 else float('nan')
    valid_test_ibs = [
        m['test_integrated_brier_score']
        for m in fold_metrics
        if np.isfinite(m.get('test_integrated_brier_score', float('nan')))
    ]
    mean_test_ibs = float(np.mean(valid_test_ibs)) if len(valid_test_ibs) > 0 else float('nan')
    std_test_ibs = float(np.std(valid_test_ibs, ddof=0)) if len(valid_test_ibs) > 0 else float('nan')
    valid_early_cis = [
        m['best_early_val_c_index']
        for m in fold_metrics
        if np.isfinite(m['best_early_val_c_index'])
    ]
    mean_early_ci = float(np.mean(valid_early_cis)) if len(valid_early_cis) > 0 else float('nan')
    std_early_ci = float(np.std(valid_early_cis, ddof=0)) if len(valid_early_cis) > 0 else float('nan')

    summary = {
        'mean_test_c_index': mean_test_ci,
        'std_test_c_index': std_test_ci,
        'mean_test_td_auc': mean_test_td_auc,
        'std_test_td_auc': std_test_td_auc,
        'mean_test_integrated_brier_score': mean_test_ibs,
        'std_test_integrated_brier_score': std_test_ibs,
        'mean_best_early_val_c_index': mean_early_ci,
        'std_best_early_val_c_index': std_early_ci,
        'folds': fold_metrics,
        'args': vars(args),
        'final_gene_count': len(final_gene_cols),
        'feature_count': int(max([m.get('feature_count', len(final_gene_cols)) for m in fold_metrics], default=len(final_gene_cols))),
    }

    with open(os.path.join(args.results_dir, 'cv_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Append a one-line CSV row for quick comparison across runs
    csv_path = os.path.join(args.results_dir, 'cv_results.csv')
    csv_row = pd.DataFrame([
        {
            'timestamp': time.time(),
            'mean_test_c_index': mean_test_ci,
            'std_test_c_index': std_test_ci,
            'mean_test_td_auc': mean_test_td_auc,
            'std_test_td_auc': std_test_td_auc,
            'mean_test_integrated_brier_score': mean_test_ibs,
            'std_test_integrated_brier_score': std_test_ibs,
            'mean_best_early_val_c_index': mean_early_ci,
            'std_best_early_val_c_index': std_early_ci,
            'num_folds': int(args.num_folds),
            'epochs': int(args.epochs),
            'early_stop_fraction': float(args.early_stop_fraction),
            'batch_size': int(args.batch_size),
            'embed_dim': int(args.embed_dim),
            'num_heads': int(args.num_heads),
            'num_layers': int(args.num_layers),
            'dropout': float(args.dropout),
            'lr': float(args.lr),
            'weight_decay': float(args.weight_decay),
            'alpha_nce': float(args.alpha_nce),
            'beta_match': float(args.beta_match),
            'gamma_mask': float(args.gamma_mask),
            'input_mode': str(args.input_mode),
            'loss_fn': str(args.loss_fn),
            'bulk_encoder': str(args.bulk_encoder),
            'bulk_mlp_hidden': str(args.bulk_mlp_hidden),
            'bulk_mlp_dropout': float(args.bulk_mlp_dropout),
            'fusion_type': str(args.fusion_type),
            'fusion_mlp_hidden': str(args.fusion_mlp_hidden),
            'risk_head': str(args.risk_head),
            'pred_head_hidden': str(args.pred_head_hidden),
            'direct_cox_from_fusion': bool(args.direct_cox_from_fusion),
            'feature_count': int(summary['feature_count']),
        }
    ])
    header_needed = not os.path.exists(csv_path)
    csv_row.to_csv(csv_path, mode='a', index=False, header=header_needed)

    print(
        f"\n5-fold CV finished. Mean held-out test C-index: {mean_test_ci:.4f} ± {std_test_ci:.4f}; "
        f"tdAUC: {mean_test_td_auc:.4f} ± {std_test_td_auc:.4f}; "
        f"IBS: {mean_test_ibs:.4f} ± {std_test_ibs:.4f}"
    )


if __name__ == '__main__':
    main()
