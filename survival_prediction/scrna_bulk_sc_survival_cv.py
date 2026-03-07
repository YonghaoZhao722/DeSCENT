import os
import sys
from pathlib import Path

# Add DeSCENT scgep_generation to path for VAE imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_DESCENT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_DESCENT_ROOT / "scgep_generation"))

import json
import time
import math
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

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


def run_fold(
    fold_num: int,
    decoded_samples_filtered: Dict[str, Dict],
    bulk_all: pd.DataFrame,
    final_gene_cols: List[str],
    args: argparse.Namespace,
    device: torch.device,
    results_dir: str,
    deg_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Run training and validation for one fold. Returns summary metrics for the fold."""
    # 0) Per-fold DEG: load fold-specific DEG and filter to common genes (use copies to avoid mutating shared data)
    if deg_dir is not None:
        deg_path = os.path.join(deg_dir, f'degs_fold{fold_num}.csv')
        if os.path.exists(deg_path):
            deg_df = pd.read_csv(deg_path, index_col=0)
            deg_genes_raw = deg_df.index.astype(str).tolist()
            if 'symbol' in deg_df.columns:
                deg_genes_raw = deg_df['symbol'].astype(str).tolist()
            common_genes = [g for g in deg_genes_raw if g in bulk_all.columns]
            # If DEG uses ENSG but bulk uses symbols, map ENSG->symbol via gene_list_path
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
                final_gene_cols = [c for c in common_genes if c in bulk_all.columns]
                bulk_all = bulk_all[final_gene_cols].copy()
                decoded_samples_fold = {}
                for k in decoded_samples_filtered:
                    df = decoded_samples_filtered[k]['X_df']
                    cols = [c for c in final_gene_cols if c in df.columns]
                    decoded_samples_fold[k] = {
                        **decoded_samples_filtered[k],
                        'X_df': df[cols].copy(),
                    }
                decoded_samples_filtered = decoded_samples_fold
                print(f"  Fold {fold_num}: DEG filter -> {len(final_gene_cols)} genes")
            else:
                print(f"  WARNING: Fold {fold_num} DEG filter produced 0 common genes (check ENSG/symbol mapping)")

    # 1) Load survival labels for this fold
    train_surv, val_surv = load_survival_fold(fold_num, args.surv_label_dir)

    # 2) Collect survival targets for SC samples
    sc_keys = list(decoded_samples_filtered.keys())
    train_ids, train_durs, train_evs = collect_targets_for_sc_keys(train_surv, sc_keys)
    val_ids, val_durs, val_evs = collect_targets_for_sc_keys(val_surv, sc_keys)

    # 3) Keep only ids that have SC data
    train_ids = [pid for pid in train_ids if pid in decoded_samples_filtered]
    val_ids = [pid for pid in val_ids if pid in decoded_samples_filtered]

    # 4) Map SC ids to bulk indices and filter
    tr_map = map_ids_to_bulk(train_ids, bulk_all)
    vl_map = map_ids_to_bulk(val_ids, bulk_all)
    train_ids = [pid for pid in train_ids if pid in tr_map and tr_map[pid] in bulk_all.index]
    val_ids = [pid for pid in val_ids if pid in vl_map and vl_map[pid] in bulk_all.index]

    # 5) Reindex bulk matrices to SC ids
    bulk_train = bulk_all.loc[[tr_map[pid] for pid in train_ids]].copy()
    bulk_train.index = train_ids
    bulk_val = bulk_all.loc[[vl_map[pid] for pid in val_ids]].copy()
    bulk_val.index = val_ids

    # 6) Narrow SC dataframes to final_gene_cols (safety)
    for k in decoded_samples_filtered.keys():
        df = decoded_samples_filtered[k]['X_df']
        if list(df.columns) != final_gene_cols:
            decoded_samples_filtered[k]['X_df'] = df[final_gene_cols]

    # 7) Build datasets and loaders
    train_ds = PatientBatchDataset(
        train_ids,
        decoded_samples_filtered,
        bulk_train,
        train_durs,
        train_evs,
        max_cells=args.max_cells,
        seed=args.seed,
    )
    val_ds = PatientBatchDataset(
        val_ids,
        decoded_samples_filtered,
        bulk_val,
        val_durs,
        val_evs,
        max_cells=args.max_cells,
        seed=args.seed,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=pad_collate_fn, drop_last=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=pad_collate_fn, drop_last=False
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

    # Prepare bin edges for discrete-time losses
    bin_edges_np = None
    if args.loss_fn in ('deephit', 'mtlr'):
        durs_arr = np.array([train_durs[pid] for pid in train_ids], dtype=np.float32)
        bin_edges_np = make_time_bins(durs_arr, num_bins=args.num_time_bins, strategy=args.time_bins_strategy)

    # Train
    fold_dir = os.path.join(results_dir, f'fold{fold_num}')
    os.makedirs(fold_dir, exist_ok=True)
    best_val_cindex = -1.0
    best_path = os.path.join(fold_dir, 'model.pt')
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
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

        val_c = evaluate_cindex(model, val_loader, device)
        log_row = {
            'epoch': epoch,
            **train_metrics,
            'val_c_index': float(val_c) if np.isfinite(val_c) else float('nan'),
            'lr': float(optimizer.param_groups[0]['lr']),
        }
        history.append(log_row)
        print(json.dumps({'fold': fold_num, **log_row}))

        if np.isfinite(val_c) and float(val_c) > best_val_cindex:
            best_val_cindex = float(val_c)
            torch.save(
                {
                    'model': model.state_dict(),
                    'epoch': epoch,
                    'val_c_index': best_val_cindex,
                    'args': vars(args),
                    'final_gene_cols': final_gene_cols,
                },
                best_path,
            )

    # Persist history and plot losses
    with open(os.path.join(fold_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    try:
        plot_losses_from_history(history, save_path=os.path.join(fold_dir, 'losses.png'))
    except Exception:
        pass

    return {
        'fold': float(fold_num),
        'best_val_c_index': float(best_val_cindex),
    }


def main():
    _default_config = str(_DESCENT_ROOT / "config" / "path.json")
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=_default_config, help='Path to path.json')
    parser.add_argument('--cancer', type=str, default=None, help='Cancer type; when set, load paths from config')
    parser.add_argument('--sc_npz_root', type=str, default=None)
    parser.add_argument('--gene_list_csv', type=str, default=None)
    parser.add_argument('--deg_csv', type=str, default=None)
    parser.add_argument('--deg_dir', type=str, default=None, help='Dir with degs_fold{k}.csv for per-fold DEG (overrides deg_csv)')
    parser.add_argument('--bulk_dir', type=str, default=None)
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--surv_label_dir', type=str, default=None)
    parser.add_argument('--vae_ckpt_path', type=str, default=None)
    parser.add_argument('--vae_num_genes', type=int, default=28952)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=16)
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
    args = parser.parse_args()

    # Load paths from config if --cancer is set
    if args.cancer is not None:
        import json
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
        # When deg_dir is set, use per-fold DEG only; do not load global deg_csv from config
        if args.deg_dir is None:
            args.deg_csv = args.deg_csv or resolve(c.get('deg', ''))
        else:
            args.deg_csv = ''
        args.bulk_dir = args.bulk_dir or resolve(c['bulk'])
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
        args.surv_label_dir = args.surv_label_dir or str(_DESCENT_ROOT / "data" / "surv_label")
        args.vae_ckpt_path = args.vae_ckpt_path or ""
        args.results_dir = args.results_dir or str(_DESCENT_ROOT / "output" / "survival_cv")

    print(args)
    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    use_per_fold_deg = args.deg_dir is not None
    if use_per_fold_deg:
        # Decode to full gene space; per-fold DEG filtering happens in run_fold
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
        # Single DEG file (original behavior)
        if not args.deg_csv:
            raise ValueError("--deg_csv or --deg_dir required when not using --cancer with config")
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

    print('Loading bulk expression tables and aligning gene columns...')

    # Run folds
    fold_metrics: List[Dict[str, float]] = []
    for fold in tqdm(range(1, args.num_folds + 1), total=args.num_folds, desc='Running folds'):
        metrics = run_fold(
            fold_num=fold,
            decoded_samples_filtered=decoded_samples_filtered,
            bulk_all=bulk_all,
            final_gene_cols=final_gene_cols,
            args=args,
            device=device,
            results_dir=args.results_dir,
            deg_dir=args.deg_dir,
        )
        fold_metrics.append(metrics)

    # Aggregate
    valid_cis = [m['best_val_c_index'] for m in fold_metrics if np.isfinite(m['best_val_c_index'])]
    mean_best_ci = float(np.mean(valid_cis)) if len(valid_cis) > 0 else float('nan')
    std_best_ci = float(np.std(valid_cis, ddof=0)) if len(valid_cis) > 0 else float('nan')

    summary = {
        'mean_best_val_c_index': mean_best_ci,
        'std_best_val_c_index': std_best_ci,
        'folds': fold_metrics,
        'args': vars(args),
        'final_gene_count': len(final_gene_cols),
    }

    with open(os.path.join(args.results_dir, 'cv_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Append a one-line CSV row for quick comparison across runs
    csv_path = os.path.join(args.results_dir, 'cv_results.csv')
    csv_row = pd.DataFrame([
        {
            'timestamp': time.time(),
            'mean_best_val_c_index': mean_best_ci,
            'std_best_val_c_index': std_best_ci,
            'num_folds': int(args.num_folds),
            'epochs': int(args.epochs),
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
            'bulk_encoder': str(args.bulk_encoder),
            'bulk_mlp_hidden': str(args.bulk_mlp_hidden),
            'bulk_mlp_dropout': float(args.bulk_mlp_dropout),
            'fusion_type': str(args.fusion_type),
            'fusion_mlp_hidden': str(args.fusion_mlp_hidden),
            'risk_head': str(args.risk_head),
            'pred_head_hidden': str(args.pred_head_hidden),
            'direct_cox_from_fusion': bool(args.direct_cox_from_fusion),
        }
    ])
    header_needed = not os.path.exists(csv_path)
    csv_row.to_csv(csv_path, mode='a', index=False, header=header_needed)

    print(f"\n5-fold CV finished. Mean val C-index: {mean_best_ci:.4f} ± {std_best_ci:.4f}")


if __name__ == '__main__':
    main()


