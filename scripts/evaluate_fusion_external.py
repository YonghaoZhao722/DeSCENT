"""External validation of frozen DeSCENT fusion checkpoints.

Mirrors the protocol already used for the bulk-only baseline
(scripts/evaluate_bulk_mlp_external.py): every TCGA outer-fold checkpoint is evaluated,
unchanged, on the full prepared external cohort. External outcomes are never used for
fitting, preprocessing statistics, epoch selection, or model selection. Bulk features are
standardised with the checkpoint's own training mean/scale; genes absent from the external
platform receive z-score 0, i.e. the training mean.

Unlike the bulk baseline, the fusion model also consumes scGEP, so the external cohort must
have its own generated single-cell profiles. Only BRCA (GSE96058) currently does.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SURV = ROOT / "survival_prediction"
if str(SURV) not in sys.path:
    sys.path.insert(0, str(SURV))

import scrna_bulk_sc_survival_cv as cv
from scrna_bulk_sc_survival import BulkSCFusionSurvival, PatientBatchDataset, pad_collate_fn
from mil_survival_training import make_time_bins

EXTERNAL_COHORT = {"BRCA": "GSE96058"}

# TCGA OS.time is in DAYS (BRCA median 847, max 8605). The GEO external cohorts report
# OS.time in MONTHS (GSE96058 median 52.4, max 81.3; GSE84437 values like 23.0, 24.0).
# The model's baseline hazard and discrete-time bin edges live on the training (day)
# axis, so external durations MUST be converted before any time-dependent metric.
# C-index is rank-based and unaffected; IBS and tdAUC are not. Evaluating months
# against a day-scale curve is what drives the bulk baseline's external IBS to
# 0.054-0.774 while its internal IBS is a sane 0.169-0.301.
DAYS_PER_MONTH = 365.25 / 12.0
EXTERNAL_TIME_UNIT = {"BRCA": "months"}


def list_sample_folders(root: str, prefix: str) -> List[str]:
    return [d for d in sorted(os.listdir(root))
            if d.startswith(prefix) and os.path.isdir(os.path.join(root, d))]


def load_gene_positions(gene_list_csv: str, selected: Sequence[str]):
    full = pd.read_csv(gene_list_csv).iloc[:, 1].astype(str).tolist()
    pos_by_gene = {g: i for i, g in enumerate(full)}
    names = [g for g in selected if g in pos_by_gene]
    positions = [pos_by_gene[g] for g in names]
    if not positions:
        raise ValueError('selected genes do not overlap the VAE gene space')
    return positions, names


def decode_chunk(root: str, folders: Sequence[str], vae, device, positions, names) -> Dict[str, Dict]:
    """Decode one chunk of samples to the checkpoint's gene space.

    Decoding the whole cohort at once is not viable: ~2045 cells x ~3000 genes per sample
    is ~25 MB, so 3409 external samples would need ~85 GB (an earlier attempt reached
    230 GB RSS before being killed). Chunking bounds it to a few GB.
    """
    out: Dict[str, Dict] = {}
    for folder in folders:
        path = os.path.join(root, folder)
        cell_gen, type_ids, type_names = [], [], []
        for f in sorted(os.listdir(path)):
            if not f.endswith('.npz'):
                continue
            npz = np.load(os.path.join(path, f), allow_pickle=True)
            if 'cell_gen' not in npz:
                continue
            arr = npz['cell_gen']
            cell_gen.append(arr)
            tname = os.path.splitext(f)[0]
            if tname not in type_names:
                type_names.append(tname)
            type_ids.append(np.full((arr.shape[0],), type_names.index(tname), dtype=np.int64))
        if not cell_gen:
            continue
        X = np.concatenate(cell_gen, axis=0)
        with torch.no_grad():
            dec = vae(torch.tensor(X, dtype=torch.float32, device=device),
                      return_decoded=True).detach().cpu().numpy()
        out[folder] = {'X_df': pd.DataFrame(dec[:, positions].astype(np.float32), columns=names),
                       'type_ids': np.concatenate(type_ids, axis=0) if type_ids else None,
                       'type_names': type_names}
    return out


def predict_streaming(model, root, folders, bulk_std, durs, evs, vae, device, positions,
                      names, max_cells, seed, batch_size, chunk=120) -> Dict[str, np.ndarray]:
    """Decode -> forward -> keep only predictions, one chunk at a time."""
    import gc
    parts = []
    for i in range(0, len(folders), chunk):
        sub = folders[i:i + chunk]
        decoded = decode_chunk(root, sub, vae, device, positions, names)
        ids = [k for k in sub if k in decoded and k in durs]
        if not ids:
            del decoded; gc.collect(); continue
        loader = make_loader(ids, decoded, bulk_std.loc[ids],
                             {k: durs[k] for k in ids}, {k: evs[k] for k in ids},
                             max_cells, seed, batch_size)
        parts.append(cv.collect_model_predictions(model, loader, device))
        del decoded, loader
        gc.collect()
    if not parts:
        raise ValueError('no samples produced predictions')
    keys = ['risk', 'duration', 'event']
    merged = {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}
    logits = [p['logits'] for p in parts if p['logits'].size]
    merged['logits'] = np.concatenate(logits, axis=0) if logits else np.empty((0, 0))
    return merged


def standardize(frame: pd.DataFrame, genes: Sequence[str], mean: np.ndarray,
                scale: np.ndarray, transform: str) -> pd.DataFrame:
    """Align to the checkpoint's gene order and apply its training statistics.
    Genes missing from the external platform become 0 after standardisation."""
    out = pd.DataFrame(0.0, index=frame.index, columns=list(genes), dtype=np.float32)
    shared = [g for g in genes if g in frame.columns]
    vals = frame[shared].to_numpy(dtype=np.float32)
    if transform == 'log1p_zscore':
        vals = np.log1p(np.clip(vals, 0.0, None))
    elif transform not in ('zscore',):
        raise ValueError(f'unsupported transform for external eval: {transform}')
    idx = np.array([list(genes).index(g) for g in shared], dtype=int)
    out.loc[:, shared] = (vals - mean[idx]) / scale[idx]
    return out


def build_model(ckpt: Dict, device) -> BulkSCFusionSurvival:
    a = argparse.Namespace(**ckpt['args'])
    def dims(s):
        s = str(s).strip()
        if s == '' or s.lower() == 'auto': return None
        v = [x for x in s.split(',') if x.strip() != '']
        return [] if (len(v) == 1 and v[0].strip() == '0') else [int(x) for x in v]
    ph = dims(a.pred_head_hidden)
    if bool(a.direct_cox_from_fusion): ph = []
    model = BulkSCFusionSurvival(
        num_genes=len(ckpt['final_gene_cols']), embed_dim=a.embed_dim,
        cell_encoder_hidden=[512, 256], num_attention_heads=a.num_heads,
        num_attention_layers=a.num_layers, dropout=a.dropout,
        bulk_encoder_type=a.bulk_encoder, cell_encoder_type=a.cell_encoder,
        fusion_type=a.fusion_type, pooling=a.pooling, risk_head=a.risk_head,
        output_mode=a.loss_fn, num_time_bins=a.num_time_bins,
        concat_sc_with_fusion=bool(a.concat_sc_with_fusion),
        residual_hfuse_to_bulk=bool(a.residual_hfuse_to_bulk),
        balance_residual=bool(a.balance_residual),
        sc_gate_init=float(a.sc_gate_init), sc_scale=float(a.sc_scale),
        bulk_mlp_hidden=dims(a.bulk_mlp_hidden), bulk_mlp_dropout=float(a.bulk_mlp_dropout),
        fusion_mlp_hidden=dims(a.fusion_mlp_hidden), pred_head_hidden=ph,
        use_sc=(a.input_mode == 'bulk_scgep'),
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


def make_loader(ids, decoded, bulk, durs, evs, max_cells, seed, batch_size):
    ds = PatientBatchDataset(list(ids), decoded, bulk, durs, evs,
                             max_cells=max_cells, seed=seed, resample_cells=False)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                       num_workers=0, collate_fn=pad_collate_fn)


def allowed_external_samples(cancer: str):
    """Same cohort filter as the bulk-baseline evaluator, so the two are comparable.
    GSE96058 holds 3,273 subjects plus 136 sequencing replicates; keep one row per subject."""
    if cancer != 'BRCA':
        return None
    import gzip
    selected: Dict[str, str] = {}
    for path in sorted(Path('/data/zhaoyh/SHAP/data/BRCA_GEO').glob('GSE96058-*series_matrix.txt.gz')):
        titles = external_ids = None
        with gzip.open(path, 'rt', errors='replace') as fh:
            for line in fh:
                if line.startswith('!Sample_title'):
                    titles = [i.strip('"\n') for i in line.split('\t')[1:]]
                elif line.startswith('!Sample_characteristics_ch1') and 'scan-b external id:' in line.lower():
                    external_ids = [i.strip('"\n').split(': ', 1)[1] for i in line.split('\t')[1:]]
                if titles is not None and external_ids is not None:
                    break
        if titles is None or external_ids is None or len(titles) != len(external_ids):
            raise ValueError(f'could not parse BRCA replicate metadata from {path}')
        for title, ext in zip(titles, external_ids):
            selected.setdefault(ext.split('.l', 1)[0], title)
    if len(selected) != 3273:
        raise ValueError(f'expected 3,273 unique GSE96058 subjects, found {len(selected)}')
    return set(selected.values())


def load_external(cancer: str) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, int]]:
    """Prepared fold-1 train+val recombined into the full external cohort, with outcomes."""
    bulk_dir = ROOT / 'data' / cancer / 'bulk_geo'
    surv_dir = Path('/data/zhaoyh/SHAP/data/BRCA_GEO') if cancer == 'BRCA' \
        else Path('/data/zhaoyh/SHAP/data') / cancer
    bulk = pd.concat([pd.read_csv(bulk_dir / f'{s}_data_1.csv', index_col=0)
                      for s in ('train', 'val')], axis=0)
    bulk.index = bulk.index.astype(str)
    if bulk.index.duplicated().any():
        bulk = bulk[~bulk.index.duplicated(keep='first')]
    tr, va = cv.load_survival_fold(1, str(surv_dir))
    surv = pd.concat([tr, va], ignore_index=True)
    allowed = allowed_external_samples(cancer)
    durs, evs = {}, {}
    for _, r in surv.iterrows():
        sid = str(r['tcga_barcode'])
        d = pd.to_numeric(r['duration'], errors='coerce')
        e = pd.to_numeric(r['event'], errors='coerce')
        if sid not in bulk.index or not np.isfinite(d) or not np.isfinite(e):
            continue
        if allowed is not None and sid not in allowed:
            continue
        durs[sid], evs[sid] = float(d), int(e)
    if EXTERNAL_TIME_UNIT.get(cancer) == 'months':
        durs = {k: v * DAYS_PER_MONTH for k, v in durs.items()}
    return bulk, durs, evs


def internal_fit_partition(cancer: str, fold: int, a, sc_keys: Sequence[str]):
    """Reproduce exactly what training used: barcode-containment matching over the fold's
    survival table (cv.collect_targets_for_sc_keys), the merged bulk matrix, then the
    seeded fit / early-stop split. Iterating sc folders instead gives a different set."""
    train_surv, _ = cv.load_survival_fold(fold, a.surv_label_dir)
    train_ids, train_durs, train_evs = cv.collect_targets_for_sc_keys(train_surv, list(sc_keys))
    bulk_all = cv.load_bulk_all(a.bulk_dir)
    tr_map = cv.map_ids_to_bulk(train_ids, bulk_all)
    train_ids = [p for p in train_ids if p in tr_map and tr_map[p] in bulk_all.index]
    fit_ids, early_val_ids = cv.split_train_early_val_ids(
        train_ids, train_evs, val_fraction=float(a.early_stop_fraction),
        seed=int(a.seed) + int(fold))
    return fit_ids, early_val_ids, train_durs, train_evs, tr_map, bulk_all


def evaluate_arm(cancer: str, arm_dir: Path, device, batch_size: int = 32,
                 chunk: int = 120, prediction_dir: Path | None = None) -> List[Dict]:
    """One frozen checkpoint per outer fold, each scored on the full external cohort."""
    import gc
    rows = []
    ext_root = str(ROOT / 'output/scgep_condgen/BRCA_GEO/redeconv')
    ext_bulk_raw, ext_durs_raw, ext_evs_raw = load_external(cancer)
    for fold in range(1, 6):
        ck_path = arm_dir / f'fold{fold}_run' / f'fold{fold}' / 'model.pt'
        if not ck_path.exists():
            rows.append({'fold': fold, 'error': 'checkpoint missing'}); continue
        ck = torch.load(ck_path, map_location='cpu', weights_only=False)
        a = argparse.Namespace(**ck['args'])
        genes = list(map(str, ck['final_gene_cols']))
        mean = np.asarray(ck['bulk_feature_mean'], dtype=np.float32)
        scale = np.asarray(ck['bulk_feature_scale'], dtype=np.float32)
        positions, names = load_gene_positions(a.gene_list_csv, genes)
        vae = cv.load_vae(device, num_genes=int(a.vae_num_genes), ckpt_path=a.vae_ckpt_path)
        model = build_model(ck, device)

        # --- internal reference distribution: this fold's fit partition ---
        int_folders = list_sample_folders(a.sc_npz_root, 'cells_2048_TCGA-')
        fit_ids, early_val_ids, tr_durs, tr_evs, tr_map, bulk_all = internal_fit_partition(
            cancer, fold, a, int_folders)
        sc_counts = ck.get('split_counts') or {}
        if sc_counts and (len(fit_ids) != sc_counts.get('train_fit') or
                          len(early_val_ids) != sc_counts.get('early_val')):
            raise ValueError(
                f'fold {fold}: reconstructed split {len(fit_ids)}/{len(early_val_ids)} does not '
                f'match the checkpoint ({sc_counts.get("train_fit")}/{sc_counts.get("early_val")})')
        bin_edges = None
        if str(a.loss_fn).lower() in ('deephit', 'mtlr'):
            bin_edges = make_time_bins(
                np.array([tr_durs[p] for p in fit_ids], dtype=np.float32),
                num_bins=int(a.num_time_bins), strategy=str(a.time_bins_strategy))
        ib = bulk_all.loc[[tr_map[p] for p in fit_ids]].copy(); ib.index = fit_ids
        ib = standardize(ib, names, mean, scale, a.bulk_transform)
        train_pred = predict_streaming(
            model, a.sc_npz_root, fit_ids, ib,
            {p: tr_durs[p] for p in fit_ids}, {p: tr_evs[p] for p in fit_ids},
            vae, device, positions, names, a.max_cells, a.seed, batch_size, chunk)

        # --- external cohort ---
        ext_folders = list_sample_folders(ext_root, 'cells_2048_F')
        ext_ids = [k for k in ext_folders if k.replace('cells_2048_', '') in ext_durs_raw
                   and k.replace('cells_2048_', '') in ext_bulk_raw.index]
        eb = ext_bulk_raw.loc[[k.replace('cells_2048_', '') for k in ext_ids]].copy()
        eb.index = ext_ids
        eb = standardize(eb, names, mean, scale, a.bulk_transform)
        eval_pred = predict_streaming(
            model, ext_root, ext_ids, eb,
            {k: ext_durs_raw[k.replace('cells_2048_', '')] for k in ext_ids},
            {k: ext_evs_raw[k.replace('cells_2048_', '')] for k in ext_ids},
            vae, device, positions, names, a.max_cells, a.seed, batch_size, chunk)

        surv_df = cv.build_survival_dataframe(a.loss_fn, train_pred, eval_pred, bin_edges)
        # The unrestricted IBS integrates over the TCGA event grid, far past the external
        # cohort's last follow-up, where the IPCW weights diverge; restrict it and report
        # the cohort's own marginal-KM null alongside so the number can be read at all.
        horizon = float(np.max(eval_pred['duration'])) if eval_pred['duration'].size else float('nan')
        null_ibs, ipa = cv.compute_ipa(surv_df, eval_pred['duration'], eval_pred['event'], horizon)
        rows.append({
            'fold': fold, 'n_external': int(eval_pred['risk'].size),
            'n_reference': int(train_pred['risk'].size), 'n_genes': len(names),
            'c_index': cv.harrell_cindex_from_arrays(eval_pred['risk'], eval_pred['duration'], eval_pred['event']),
            'td_auc': cv.compute_td_auc(train_pred, eval_pred),
            'integrated_brier_score': cv.compute_integrated_brier_score(surv_df, eval_pred['duration'], eval_pred['event']),
            'ibs_restricted': cv.compute_integrated_brier_score(surv_df, eval_pred['duration'], eval_pred['event'], horizon),
            'ibs_null_model': null_ibs,
            'ipa': ipa,
            'brier_horizon_days': horizon,
        })
        # Cache the raw per-sample predictions: every metric above is a pure function of
        # these arrays, so changing a metric definition later costs seconds instead of
        # another ~20 min/fold pass over the external scGEP.
        if prediction_dir is not None:
            np.savez_compressed(
                Path(prediction_dir) / f'{cancer}_{arm_dir.name}_fold{fold}.npz',
                bin_edges=np.asarray(bin_edges if bin_edges is not None else [], dtype=np.float64),
                **{f'train_{k}': v for k, v in train_pred.items()},
                **{f'external_{k}': v for k, v in eval_pred.items()},
            )
        del model, vae, train_pred, eval_pred, surv_df
        gc.collect(); torch.cuda.empty_cache()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cancer', default='BRCA')
    ap.add_argument('--arms', nargs='+', required=True,
                    help='arm directories, e.g. output/f4090/BRCA/h_cox')
    ap.add_argument('--out', default='output/fusion_external_metrics')
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=120)
    ap.add_argument("--save-predictions", action='store_true',
                    help='Cache per-sample risk/logits/duration/event for GPU-free metric recomputation')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = ROOT / args.out; out_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = out_dir / 'predictions' if args.save_predictions else None
    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for arm in args.arms:
        arm_dir = ROOT / arm
        print(f'=== {arm_dir} ===', flush=True)
        for r in evaluate_arm(args.cancer, arm_dir, device, args.batch_size, args.chunk, prediction_dir):
            r.update({'cancer': args.cancer, 'arm': arm_dir.name, 'arm_dir': arm,
                      'external_cohort': EXTERNAL_COHORT[args.cancer]})
            all_rows.append(r)
            print('   ', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / 'external_fold_metrics.csv', index=False)
    print('wrote', out_dir / 'external_fold_metrics.csv')


if __name__ == '__main__':
    main()
