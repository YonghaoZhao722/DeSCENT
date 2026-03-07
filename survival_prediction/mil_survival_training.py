import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsurv.loss.cox import neg_partial_log_likelihood
from torch.utils.data import Dataset, DataLoader
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
from pycox.evaluation import EvalSurv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any, Optional, Union
import warnings
from tqdm.notebook import tqdm

warnings.filterwarnings('ignore')

# Import our MIL model
from mil_survival_model import MILSurvivalNet, create_mil_survival_model


# In-memory cache to avoid re-normalizing the same sample across folds
_PREPROCESSED_SCRNA_CACHE: Dict[tuple, np.ndarray] = {}


def clear_preprocessed_scrna_cache() -> None:
    """Clear the in-memory preprocessing cache."""
    _PREPROCESSED_SCRNA_CACHE.clear()


class MILSurvivalDataset(Dataset):
    """Patient-level MIL dataset that optionally resamples K cells per epoch.

    Optionally carries per-cell type ids (ints, -1 indicates padding/invalid).
    """
    def __init__(
        self,
        X: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
        sample_cells_k: Optional[int] = None,
        resample_with_replacement: bool = True,
        types: Optional[np.ndarray] = None,
    ) -> None:
        self.X = X  # (num_patients, num_cells, num_genes)
        self.durations = durations.astype(np.float32)
        self.events = events.astype(np.float32)
        self.sample_cells_k = sample_cells_k
        self.resample_with_replacement = resample_with_replacement
        self._rng = np.random.default_rng()
        self.types = types if types is not None else None  # (num_patients, num_cells) or None

    def set_epoch(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        cells = self.X[idx]  # (num_cells, num_genes)
        types_row = None if self.types is None else self.types[idx]
        if self.sample_cells_k is not None and self.sample_cells_k > 0 and cells.shape[0] > self.sample_cells_k:
            indices = self._rng.choice(cells.shape[0], self.sample_cells_k, replace=self.resample_with_replacement)
            cells = cells[indices]
            if types_row is not None:
                types_row = types_row[indices]
        return cells.astype(np.float32), self.durations[idx], self.events[idx], (None if types_row is None else types_row.astype(np.int64))


def collate_mil_batch(batch):
    # Support optional types vector per sample
    if len(batch[0]) == 4:
        xs, ds, es, ts = zip(*batch)
        have_types = any(t is not None for t in ts)
    else:
        xs, ds, es = zip(*batch)
        ts = None
        have_types = False
    X = torch.tensor(np.stack(xs, axis=0), dtype=torch.float32)
    durations = torch.tensor(ds, dtype=torch.float32)
    events = torch.tensor(es, dtype=torch.float32)
    if have_types:
        # Replace None with pad (-1) arrays matching shape
        t_list = []
        for i, t in enumerate(ts):
            if t is None:
                t_list.append(np.full((xs[i].shape[0],), -1, dtype=np.int64))
            else:
                t_list.append(t)
        types = torch.tensor(np.stack(t_list, axis=0), dtype=torch.long)
        return X, durations, events, types
    return X, durations, events


def cox_ph_loss(risk_scores: torch.Tensor, durations: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    """Negative Cox partial log-likelihood.

    Args:
        risk_scores: (B,) raw risks (higher means higher hazard)
        durations: (B,) positive times
        events: (B,) 1 if event observed, 0 if censored
    Returns:
        Scalar loss tensor
    """
    risk = risk_scores.view(-1)
    durations = durations.view(-1)
    events = events.view(-1)

    # Sort by descending durations so risk set R_i is prefix [0..i]
    order = torch.argsort(durations, descending=True)
    sorted_risk = risk[order]
    sorted_events = events[order]

    # log sum exp over cumulative risk sets
    log_cumsum_exp = torch.logcumsumexp(sorted_risk, dim=0)
    event_mask = sorted_events == 1.0

    if event_mask.sum() == 0:
        # Keep the graph; zero contribution if no events
        return torch.zeros((), dtype=risk.dtype, device=risk.device) + 0.0 * risk.sum()

    per_event = sorted_risk - log_cumsum_exp
    neg_log_lik = -torch.mean(per_event[event_mask])
    return neg_log_lik


def cox_ph_loss_efron(risk_scores: torch.Tensor, durations: torch.Tensor, events: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Negative Cox partial log-likelihood with Efron tie handling.

    Handles tied event times using Efron's approximation, which is more accurate than
    Breslow when multiple events share the same time. Operates on mini-batches by
    constructing batch-local risk sets following standard Cox batch training practice.

    Args:
        risk_scores: (B,) raw risks (higher means higher hazard)
        durations: (B,) positive times
        events: (B,) 1 if event observed, 0 if censored
        eps: numerical stability epsilon
    Returns:
        Scalar loss tensor
    """
    r = risk_scores.view(-1)
    t = durations.view(-1)
    e = events.view(-1)

    # Sort by descending durations so risk set R_i is prefix [0..i]
    order = torch.argsort(t, descending=True)
    r = r[order]
    t = t[order]
    e = e[order]

    exp_r = torch.exp(r)
    cumsum_exp = torch.cumsum(exp_r, dim=0)  # at-risk cumulative sums

    # Find groups of equal time (tied groups)
    # time_change[k] == True indicates start of a new time group at index k
    time_change = torch.ones_like(t, dtype=torch.bool)
    if t.numel() > 1:
        time_change[1:] = t[1:] != t[:-1]
    starts = torch.nonzero(time_change, as_tuple=False).flatten()
    if starts.numel() == 0:
        # Degenerate batch
        return torch.zeros((), dtype=r.dtype, device=r.device) + 0.0 * r.sum()
    ends = torch.cat([starts[1:], torch.tensor([t.numel()], device=t.device)])

    sum_num = r.new_tensor(0.0)
    sum_den = r.new_tensor(0.0)
    num_events_total = (e > 0.5).sum().clamp(min=1)

    for s, eidx in zip(starts.tolist(), ends.tolist()):
        group_events = e[s:eidx] > 0.5
        d = int(group_events.sum().item())
        if d == 0:
            continue
        # Numerator: sum of linear predictors for events at this time
        sum_num = sum_num + r[s:eidx][group_events].sum()
        # Risk set sum is cumulative exp at last index of this time group
        S = cumsum_exp[eidx - 1]
        # Sum of exp(r) over events in this time group
        E = exp_r[s:eidx][group_events].sum()
        # Efron denominator correction: average over d subtractions
        l_range = torch.arange(d, device=r.device, dtype=r.dtype)
        denom_terms = S - (l_range / max(d, 1)) * E + eps
        sum_den = sum_den + torch.log(denom_terms).sum()

    neg_log_lik = -(sum_num - sum_den) / num_events_total
    return neg_log_lik

def make_time_bins(durations: np.ndarray, num_bins: int = 50, strategy: str = 'quantile') -> np.ndarray:
    """Create non-decreasing bin edges of length num_bins+1 covering durations.

    strategy: 'quantile' or 'linear'
    """
    durations = np.asarray(durations, dtype=np.float32)
    durations = durations[np.isfinite(durations) & (durations > 0)]
    if len(durations) == 0:
        raise ValueError("No valid durations to make time bins")
    num_bins = int(num_bins)
    strategy = str(strategy).lower()
    if strategy == 'quantile':
        qs = np.linspace(0.0, 1.0, num_bins + 1)
        edges = np.quantile(durations, qs)
    else:
        t_min, t_max = float(np.min(durations)), float(np.max(durations))
        edges = np.linspace(t_min, t_max, num_bins + 1)
    # Ensure strict monotonicity by small jitter if necessary
    edges = np.maximum.accumulate(edges)
    # Guard against identical edges
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6
    return edges.astype(np.float32)


def discretize_durations(durations: torch.Tensor, bin_edges: np.ndarray) -> torch.Tensor:
    """Map durations to bin index in [0, num_bins-1]. torch output on same device."""
    # Use numpy searchsorted on CPU, then convert back
    d_np = durations.detach().cpu().numpy()
    idx = np.searchsorted(bin_edges, d_np, side='right') - 1
    idx = np.clip(idx, 0, len(bin_edges) - 2)
    return torch.tensor(idx, device=durations.device, dtype=torch.long)


def deepsurv_like_risk_from_output(logits: torch.Tensor, loss_type: str, bin_edges: Optional[np.ndarray] = None) -> torch.Tensor:
    """Map model outputs to a scalar risk for ranking/C-index.

    - cox: return raw log-risk (higher means higher hazard)
    - deephit: expected event time from softmax over bins, risk = -E[time]
    - mtlr: compute hazards via sigmoid, pmf = S_{k-1} * h_k, risk = -E[time]
    """
    if loss_type == 'cox':
        return logits.view(-1)
    if bin_edges is None:
        # fallback to average over bins index if edges not provided
        bin_centers_np = None
    else:
        centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
        bin_centers_np = centers.astype(np.float32)
    if loss_type == 'deephit':
        probs = torch.softmax(logits, dim=-1)  # (B, T)
        if bin_centers_np is None:
            idx = torch.arange(probs.size(-1), device=probs.device, dtype=probs.dtype)
            expected = torch.sum(probs * idx, dim=-1)
        else:
            centers = torch.tensor(bin_centers_np, device=probs.device, dtype=probs.dtype)
            expected = torch.sum(probs * centers, dim=-1)
        return -expected
    # mtlr / discrete hazard
    hazards = torch.sigmoid(logits)  # (B, T)
    one_minus_h = 1.0 - hazards
    # Survival till end of each bin: S_k = prod_{j<=k} (1-h_j)
    logS = torch.cumsum(torch.log(one_minus_h + 1e-12), dim=-1)
    S_prev = torch.cat([torch.zeros((logits.size(0), 1), device=logits.device, dtype=logits.dtype), logS[:, :-1]], dim=-1).exp()
    pmf = S_prev * hazards
    if bin_centers_np is None:
        idx = torch.arange(pmf.size(-1), device=pmf.device, dtype=pmf.dtype)
        expected = torch.sum(pmf * idx, dim=-1)
    else:
        centers = torch.tensor(bin_centers_np, device=pmf.device, dtype=pmf.dtype)
        expected = torch.sum(pmf * centers, dim=-1)
    return -expected


def deephit_single_loss(logits: torch.Tensor, durations: torch.Tensor, events: torch.Tensor, bin_edges: np.ndarray) -> torch.Tensor:
    """DeepHit-style single-risk loss.
    - Event: cross-entropy on event bin
    - Censor: negative log survival after censor time = -log sum_{j>c} p_j
    """
    probs_log = torch.log_softmax(logits, dim=-1)  # (B, T)
    all_logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)  # for stability in censored term
    idx = discretize_durations(durations, bin_edges)  # (B,)
    event_mask = (events > 0.5)
    loss_event = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    if event_mask.any():
        ce = F.nll_loss(probs_log[event_mask], idx[event_mask], reduction='mean')
        loss_event = ce
    censor_mask = ~event_mask
    loss_censor = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    if censor_mask.any():
        idx_c = idx[censor_mask]
        # mask out bins <= c
        Bc = idx_c.shape[0]
        T = logits.size(-1)
        arange_t = torch.arange(T, device=logits.device).unsqueeze(0).expand(Bc, T)
        mask_after = (arange_t > idx_c.unsqueeze(1)).float()
        # log sum p_{>c} = logsumexp(logits[>c]) - logsumexp(logits[:])
        sel = logits[censor_mask]
        logsum_after = torch.logsumexp(sel + torch.log(mask_after + 1e-12), dim=-1)
        logsum_all = torch.logsumexp(sel, dim=-1)
        neg_log_surv = -(logsum_after - logsum_all)
        loss_censor = torch.mean(neg_log_surv)
    return loss_event + loss_censor


def discrete_hazard_loss_mtlr(logits: torch.Tensor, durations: torch.Tensor, events: torch.Tensor, bin_edges: np.ndarray) -> torch.Tensor:
    """Discrete-time hazard loss (MTLR-like):
    Event at bin k: log S(k-1) + log h(k)
    Censor at c: log S(c)
    """
    hazards = torch.sigmoid(logits)  # (B, T)
    one_minus_h = 1.0 - hazards + 1e-12
    log_one_minus_h = torch.log(one_minus_h)
    logS = torch.cumsum(log_one_minus_h, dim=-1)  # log survival up to each bin end
    idx = discretize_durations(durations, bin_edges)  # (B,)
    event_mask = (events > 0.5)
    loss_terms = []
    if event_mask.any():
        idx_e = idx[event_mask]
        # log S(k-1)
        logS_prev = torch.zeros_like(idx_e, dtype=logits.dtype, device=logits.device)
        sel_logS = logS[event_mask]
        has_prev = (idx_e > 0)
        if has_prev.any():
            logS_prev[has_prev] = sel_logS[has_prev, idx_e[has_prev] - 1]
        log_h_k = torch.log(hazards[event_mask, idx_e] + 1e-12)
        loss_event = -(logS_prev + log_h_k)
        loss_terms.append(loss_event)
    if (~event_mask).any():
        idx_c = idx[~event_mask]
        sel_logS = logS[~event_mask]
        logS_c = sel_logS.gather(1, idx_c.unsqueeze(1)).squeeze(1)
        loss_censor = -logS_c
        loss_terms.append(loss_censor)
    if len(loss_terms) == 0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    return torch.mean(torch.cat(loss_terms, dim=0))


def build_ipcw_ghat(durations: np.ndarray, events: np.ndarray):
    """Build Ghat(t) estimator for censoring survival using KM on censoring indicator.
    Returns a callable ghat(t_array) -> P(C >= t).
    """
    km = KaplanMeierFitter()
    # censoring event observed when event == 0
    km.fit(durations, event_observed=1 - events)
    timeline = km.survival_function_.index.values.astype(np.float32)
    surv_vals = km.survival_function_.values.reshape(-1).astype(np.float32)
    if len(timeline) == 0:
        def ghat_fn(t):
            return np.ones_like(np.asarray(t, dtype=np.float32))
        return ghat_fn
    def ghat_fn(t):
        t_arr = np.asarray(t, dtype=np.float32)
        # step function: take last timeline <= t
        idx = np.searchsorted(timeline, t_arr, side='right') - 1
        idx = np.clip(idx, 0, len(timeline) - 1)
        return surv_vals[idx]
    return ghat_fn


def uno_pairwise_aux_loss(risk_scores: torch.Tensor, durations: torch.Tensor, events: torch.Tensor, ghat_fn) -> torch.Tensor:
    """Differentiable pairwise ranking surrogate approximating Uno's C-index.
    Pairs: i is event, d_i < d_j. Weight: 1 / G(d_i)^2.
    """
    B = risk_scores.shape[0]
    if B < 2:
        return torch.zeros((), device=risk_scores.device, dtype=risk_scores.dtype)
    d = durations.view(-1)
    e = events.view(-1)
    # Build pair masks
    d_i = d.unsqueeze(1)
    d_j = d.unsqueeze(0)
    e_i = e.unsqueeze(1)
    comparable = (e_i > 0.5) & (d_i < d_j)
    if not torch.any(comparable):
        return torch.zeros((), device=risk_scores.device, dtype=risk_scores.dtype)
    # Weights from IPCW
    with torch.no_grad():
        w_np = 1.0 / np.maximum(ghat_fn(d.detach().cpu().numpy()), 1e-6) ** 2
        w = torch.tensor(w_np, device=risk_scores.device, dtype=risk_scores.dtype)
    w_i = w.unsqueeze(1)
    s = risk_scores.unsqueeze(1) - risk_scores.unsqueeze(0)  # r_i - r_j
    pair_loss = F.softplus(-s)  # log(1 + exp(-(r_i - r_j)))
    weighted = pair_loss * w_i
    masked = weighted * comparable.float()
    num_pairs = torch.clamp(masked.sum(), min=1.0)
    return masked.sum() / num_pairs


# Removed pycox wrapper; training is now pure PyTorch


class EarlyStopping:
    """Early stopping callback for training"""
    def __init__(self, patience: int = 10, min_delta: float = 0.001, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print(f"Early stopping triggered after {self.counter} epochs without improvement")
                
        return self.early_stop


def preprocess_scrna_data(
    scrna_data: np.ndarray, 
    normalization: str = 'standardize',
    max_cells: Optional[int] = None
) -> np.ndarray:
    """
    Preprocess single-cell RNA-seq data
    Note: VAE decoded data is already log1p transformed, so we only do standardization
    
    Args:
        scrna_data: (num_cells, num_genes) array - VAE decoded data (already log1p)
        normalization: 'standardize', 'none' 
        max_cells: Maximum number of cells to keep (for memory management)
    Returns:
        Preprocessed data
    """
    
    # Sample cells if too many
    if max_cells is not None and scrna_data.shape[0] > max_cells:
        indices = np.random.choice(scrna_data.shape[0], max_cells, replace=False)
        scrna_data = scrna_data[indices]
    # Apply normalization (VAE data is already log1p transformed)
    if normalization == 'standardize':
        # Only standardization (data is already log1p from VAE)
        scaler = StandardScaler()
        scrna_data = scaler.fit_transform(scrna_data)
    elif normalization == 'none':
        # No additional normalization
        print("No additional normalization")
    else:
        # Default to standardization
        scaler = StandardScaler()
        scrna_data = scaler.fit_transform(scrna_data)
        print("Applied standardization")
    
    return scrna_data.astype(np.float32)


def prepare_mil_data_fold(
    samples_dict: Dict[str, pd.DataFrame], 
    train_survival_data: pd.DataFrame,
    test_survival_data: pd.DataFrame,
    normalization: str = 'standardize',
    use_preprocess_cache: bool = True,
    max_cells_per_sample: int = 2048,
    early_stop_split: float = 0.2,  # 10% from train for early stopping
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple, Tuple, Tuple, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Prepare MIL data for fold-based cross-validation
    
    Args:
        samples_dict: Dictionary with sample_id -> DataFrame(cells x genes)
        train_survival_data: DataFrame from train_data_X.csv (training set)
        test_survival_data: DataFrame from val_data_X.csv (test set)
        normalization: Preprocessing method
        use_preprocess_cache: Whether to cache per-sample standardization results across folds
        max_cells_per_sample: Max cells to keep per sample
        early_stop_split: Fraction of train set to use for early stopping validation
        random_state: Random seed
        
    Returns:
        X_train, X_early_stop, X_test: Training, early stop validation, and test data
        train_target, early_stop_target, test_target: Corresponding targets as numpy arrays
    """
    print("Preparing MIL data with fold-based split...")
    
    def collect_samples(survival_data, dataset_name):
        """Helper function to collect samples for a given survival dataset"""
        X_list = []
        T_list = []  # optional per-sample type ids (unpadded)
        y_duration = []
        y_event = []
        sample_ids = []
        invalid_details = []  # collect rows with invalid duration for diagnostics
        
        for _, row in tqdm(survival_data.iterrows(), total=len(survival_data), desc=f"Collecting samples for {dataset_name}"):
            # Extract TCGA barcode (assuming it's in 'tcga_barcode' column)
            tcga_barcode = row['tcga_barcode']
            
            # Find matching sample in samples_dict
            matching_sample = None
            for sample_id, sample_data in samples_dict.items():
                if tcga_barcode in sample_id:
                    matching_sample = (sample_id, sample_data)
                    break
            
            if matching_sample is None:
                print(f"No single-cell data found for {tcga_barcode}")
                continue
                
            sample_id, sample_data = matching_sample
            
            # Preprocess single-cell data with optional caching (no sampling here; padding is handled later)
            cache_key = (sample_id, normalization)
            if use_preprocess_cache and cache_key in _PREPROCESSED_SCRNA_CACHE:
                processed_data = _PREPROCESSED_SCRNA_CACHE[cache_key]
            else:
                # Support dict with {'X_df', 'type_ids'} or plain DataFrame
                if isinstance(sample_data, dict):
                    X_np = sample_data.get('X_df', sample_data.get('X'))
                    if isinstance(X_np, pd.DataFrame):
                        X_np = X_np.values
                else:
                    X_np = sample_data.values
                processed_data = preprocess_scrna_data(
                    X_np,
                    normalization=normalization,
                    max_cells=None
                )
                if use_preprocess_cache:
                    _PREPROCESSED_SCRNA_CACHE[cache_key] = processed_data
            
            X_list.append(processed_data)
            # collect types (no normalization); if dict without types, append None
            if isinstance(sample_data, dict) and ('type_ids' in sample_data):
                # Ensure length matches processed_data rows (no sampling done here)
                type_ids_np = np.asarray(sample_data['type_ids'], dtype=np.int64)
                if type_ids_np.shape[0] != processed_data.shape[0]:
                    # Best-effort truncate or pad with -1
                    n = min(type_ids_np.shape[0], processed_data.shape[0])
                    fixed = np.full((processed_data.shape[0],), -1, dtype=np.int64)
                    fixed[:n] = type_ids_np[:n]
                    type_ids_np = fixed
                T_list.append(type_ids_np)
            else:
                T_list.append(None)
            # Coerce targets to numeric and record invalids for diagnostics
            duration_value = pd.to_numeric(row.get('duration', np.nan), errors='coerce')
            event_value = pd.to_numeric(row.get('event', np.nan), errors='coerce')

            # Shift all durations by +1 to avoid zero values
            if not pd.isna(duration_value) and np.isfinite(duration_value):
                duration_value = float(duration_value) + 1.0

            if pd.isna(duration_value) or not np.isfinite(duration_value) or float(duration_value) <= 0:
                invalid_details.append({
                    'tcga_barcode': tcga_barcode,
                    'sample_id': sample_id,
                    'duration': duration_value,
                    'event': event_value
                })
            y_duration.append(float(duration_value) if not pd.isna(duration_value) else np.nan)
            y_event.append(float(event_value) if not pd.isna(event_value) else np.nan)
            sample_ids.append(sample_id)
        
        if len(X_list) == 0:
            return None, None, None
        
        # Convert to arrays and pad
        y_duration = np.array(y_duration, dtype=np.float32)
        y_event = np.array(y_event, dtype=np.float32)
        
        # Pad sequences
        num_genes = X_list[0].shape[1]
        X_padded = []
        T_padded = []
        
        for i, x in enumerate(X_list):
            if x.shape[0] < max_cells_per_sample:
                padding = np.zeros((max_cells_per_sample - x.shape[0], num_genes), dtype=np.float32)
                x_padded = np.vstack([x, padding])
                # types
                t_i = T_list[i]
                if t_i is None:
                    t_pad = np.full((max_cells_per_sample - x.shape[0],), -1, dtype=np.int64)
                    t_padded = np.concatenate([np.full((x.shape[0],), -1, dtype=np.int64), t_pad], axis=0)
                else:
                    t_i = np.asarray(t_i, dtype=np.int64)
                    if t_i.shape[0] != x.shape[0]:
                        t_i = (t_i[:x.shape[0]] if t_i.shape[0] > x.shape[0] else np.pad(t_i, (0, x.shape[0]-t_i.shape[0]), constant_values=-1))
                    t_pad = np.full((max_cells_per_sample - x.shape[0],), -1, dtype=np.int64)
                    t_padded = np.concatenate([t_i, t_pad], axis=0)
            else:
                x_padded = x[:max_cells_per_sample]
                t_i = T_list[i]
                if t_i is None:
                    t_padded = np.full((max_cells_per_sample,), -1, dtype=np.int64)
                else:
                    t_i = np.asarray(t_i, dtype=np.int64)
                    t_padded = t_i[:max_cells_per_sample] if t_i.shape[0] >= max_cells_per_sample else np.pad(t_i, (0, max_cells_per_sample - t_i.shape[0]), constant_values=-1)
            X_padded.append(x_padded)
            T_padded.append(t_padded)
        
        X = np.stack(X_padded, axis=0)
        T = np.stack(T_padded, axis=0) if any(t is not None for t in T_list) else None
        
        # Print a brief report of invalid durations, if any
        if len(invalid_details) > 0:
            print(f"{dataset_name}: detected {len(invalid_details)} invalid durations (<=0, NaN, or Inf). Showing up to 10 examples:")
            for d in invalid_details[:10]:
                print(f"  - {d['tcga_barcode']} | {d['sample_id']} | duration={d['duration']} | event={d['event']}")
        target = (y_duration, y_event)
        
        print(f"{dataset_name}: {X.shape[0]} samples, event rate: {y_event.mean():.3f}")
        return X, target, sample_ids, T
    
    # Collect training data
    X_train_full, train_target_full, train_sample_ids, T_train_full = collect_samples(train_survival_data, "Train")

    if X_train_full is None:
        raise ValueError("No training samples found!")

    # Collect test data first to shape the early-stop split to match its distribution
    X_test, test_target, test_sample_ids, T_test_full = collect_samples(test_survival_data, "Test")

    if X_test is None:
        raise ValueError("No test samples found!")

    # Build an early-stopping validation set whose duration distribution and event ratio
    # match the test set as closely as possible by stratifying on (duration_bin, event)
    n_train = X_train_full.shape[0]
    n_early_stop = max(1, int(n_train * early_stop_split))

    rng = np.random.RandomState(random_state)

    train_duration_all = np.array(train_target_full[0], dtype=np.float32)
    train_event_all = (np.array(train_target_full[1], dtype=np.float32) > 0.5).astype(int)
    test_duration_all = np.array(test_target[0], dtype=np.float32)
    test_event_all = (np.array(test_target[1], dtype=np.float32) > 0.5).astype(int)

    def _safe_duration_for_split(d: np.ndarray) -> np.ndarray:
        d = np.array(d, dtype=np.float32)
        invalid = ~np.isfinite(d) | (d <= 0)
        if np.any(~invalid):
            median_val = float(np.median(d[~invalid]))
        else:
            median_val = 1.0
        d = d.copy()
        d[invalid] = median_val
        return d

    train_duration_split = _safe_duration_for_split(train_duration_all)
    test_duration_split = _safe_duration_for_split(test_duration_all)

    # Use quantile bins based on the TEST durations to define strata
    num_bins_for_val = 10
    bin_edges_np = make_time_bins(test_duration_split, num_bins=num_bins_for_val, strategy='quantile')

    def _digitize_to_bins(d: np.ndarray, edges: np.ndarray) -> np.ndarray:
        # Map to [0, num_bins-1]
        return np.clip(np.digitize(d, edges[1:-1], right=True), 0, len(edges) - 2)

    train_bins = _digitize_to_bins(train_duration_split, bin_edges_np)
    test_bins = _digitize_to_bins(test_duration_split, bin_edges_np)

    num_bins = len(bin_edges_np) - 1
    num_strata = num_bins * 2  # per duration bin x event{0,1}

    train_strata_ids = train_bins * 2 + train_event_all
    test_strata_ids = test_bins * 2 + test_event_all

    counts_test = np.bincount(test_strata_ids, minlength=num_strata).astype(float)
    total_test = counts_test.sum()

    probs_test = counts_test / total_test
    desired_counts = np.floor(probs_test * n_early_stop).astype(int)
    # Distribute rounding residuals by largest fractional parts
    fractional = probs_test * n_early_stop - desired_counts
    remaining = n_early_stop - int(desired_counts.sum())
    if remaining > 0:
        take_more = np.argsort(-fractional)[:remaining]
        desired_counts[take_more] += 1

    indices_all = np.arange(n_train)
    early_stop_mask = np.zeros(n_train, dtype=bool)

    for s in range(num_strata):
        avail_idx = indices_all[train_strata_ids == s]
        if avail_idx.size == 0:
            continue
        k = int(min(desired_counts[s], avail_idx.size))
        if k > 0:
            chosen = rng.choice(avail_idx, size=k, replace=False)
            early_stop_mask[chosen] = True

    # If some strata were scarce, top up to required size while nudging overall event ratio toward test
    selected_idx = np.where(early_stop_mask)[0]
    deficit = n_early_stop - selected_idx.size
    if deficit > 0:
        target_event_total = int(round(test_event_all.mean() * n_early_stop)) if len(test_event_all) > 0 else int(round(train_event_all.mean() * n_early_stop))
        have_event_total = int(train_event_all[selected_idx].sum()) if selected_idx.size > 0 else 0
        need_event_more = max(0, target_event_total - have_event_total)

        remaining_idx = np.arange(n_train)[~early_stop_mask]
        remaining_events = train_event_all[~early_stop_mask]

        # First, pick events if we need to increase event ratio
        if need_event_more > 0:
            candidate = remaining_idx[remaining_events == 1]
            k1 = int(min(need_event_more, candidate.size, deficit))
            if k1 > 0:
                chosen = rng.choice(candidate, size=k1, replace=False)
                early_stop_mask[chosen] = True
                deficit -= k1

        # Fill any remaining deficit randomly
        if deficit > 0:
            remaining_idx = np.arange(n_train)[~early_stop_mask]
            k2 = int(min(deficit, remaining_idx.size))
            if k2 > 0:
                chosen = rng.choice(remaining_idx, size=k2, replace=False)
                early_stop_mask[chosen] = True

        early_stop_indices = np.where(early_stop_mask)[0]
        train_indices = np.arange(n_train)[~early_stop_mask]
    else:
        early_stop_indices = selected_idx
        train_indices = np.arange(n_train)[~early_stop_mask]

    X_train = X_train_full[train_indices]
    X_early_stop = X_train_full[early_stop_indices]
    T_train = None if T_train_full is None else T_train_full[train_indices]
    T_early_stop = None if T_train_full is None else T_train_full[early_stop_indices]

    train_target = (
        train_target_full[0][train_indices],
        train_target_full[1][train_indices]
    )
    early_stop_target = (
        train_target_full[0][early_stop_indices],
        train_target_full[1][early_stop_indices]
    )
    
    print(f"\nFinal data split:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Early stopping: {X_early_stop.shape[0]} samples") 
    print(f"  Test: {X_test.shape[0]} samples")
    
    # Align T_test to test indices order (already collected separately)
    T_test = T_test_full
    return X_train, X_early_stop, X_test, train_target, early_stop_target, test_target, T_train, T_early_stop, T_test


def train_mil_survival_model_fold(
    X_train: np.ndarray,
    X_early_stop: np.ndarray,
    X_test: np.ndarray,
    train_target: Tuple[np.ndarray, np.ndarray],
    early_stop_target: Tuple[np.ndarray, np.ndarray],
    test_target: Tuple[np.ndarray, np.ndarray],
    model_params: Dict[str, Any],
    training_params: Dict[str, Any],
    save_path: Optional[str] = None,
    T_train: Optional[np.ndarray] = None,
    T_early_stop: Optional[np.ndarray] = None,
    T_test: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Train MIL survival model end-to-end with a custom Cox loss (pure PyTorch).
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Training/loss configuration
    loss_type = str(training_params.get('loss_type', 'cox')).lower()  # 'cox' | 'deephit' | 'mtlr'
    aux_cindex_weight = float(training_params.get('aux_cindex_weight', 0.0))
    num_time_bins = int(training_params.get('num_time_bins', model_params.get('num_time_bins', 50)))
    time_bins_strategy = str(training_params.get('time_bins_strategy', 'quantile')).lower()

    # Ensure model output matches loss type unless explicitly overridden
    if 'output_mode' not in model_params:
        model_params['output_mode'] = 'cox' if loss_type == 'cox' else ('deephit' if loss_type == 'deephit' else 'mtlr')
    if 'num_time_bins' not in model_params:
        model_params['num_time_bins'] = num_time_bins

    # Create MIL model (filter out non-factory args like pretrained paths)
    _factory_allowed_keys = {
        'num_genes', 'cell_embed_dim', 'cell_encoder_hidden',
        'num_attention_heads', 'num_attention_layers', 'final_hidden_dim',
        'dropout', 'use_layer_norm', 'use_batch_norm', 'l1_reg',
        'output_mode', 'num_time_bins', 'preencoded_input','pooling'
    }
    _factory_kwargs = {k: v for k, v in model_params.items() if k in _factory_allowed_keys}
    # Provide defaults for hierarchical pooling keys if not present
    _factory_kwargs.setdefault('use_hierarchical_pooling', False)
    _factory_kwargs.setdefault('gamma_intra', 0.5)
    _factory_kwargs.setdefault('eta_inter', 0.5)
    _factory_kwargs.setdefault('attn_tau_intra', 1.0)
    _factory_kwargs.setdefault('attn_tau_inter', 1.0)
    model = create_mil_survival_model(**_factory_kwargs).to(device)

    # Optionally load a pretrained cell encoder and/or freeze it
    pretrained_encoder_path = model_params.get('pretrained_encoder_path', None)
    if isinstance(pretrained_encoder_path, str) and len(pretrained_encoder_path) > 0 and os.path.exists(pretrained_encoder_path):
        try:
            enc_state = torch.load(pretrained_encoder_path, map_location=device)
            missing, unexpected = model.cell_encoder.load_state_dict(enc_state, strict=False)
            print(f"Loaded pretrained cell encoder from: {pretrained_encoder_path}")
            if missing:
                print(f"[Encoder load] Missing keys: {len(missing)} (showing up to 10): {missing[:10]}")
            if unexpected:
                print(f"[Encoder load] Unexpected keys: {len(unexpected)} (showing up to 10): {unexpected[:10]}")
        except Exception as e:
            print(f"Warning: failed to load pretrained encoder from {pretrained_encoder_path}: {e}")
    freeze_encoder = bool(training_params.get('freeze_encoder', False))
    if freeze_encoder:
        for p in model.cell_encoder.parameters():
            p.requires_grad = False
        print("Cell encoder parameters are frozen (no gradient updates).")

    # Training params
    lr = float(training_params.get('lr', 1e-4))
    weight_decay = float(training_params.get('weight_decay', 1e-5))
    batch_size = int(training_params.get('batch_size', 8))
    epochs = int(training_params.get('epochs', 200))
    patience = int(training_params.get('patience', 20))
    min_delta = float(training_params.get('min_delta', 1e-4))
    grad_accum_steps = int(training_params.get('grad_accum_steps', 1))
    # Attention entropy regularization on pooling weights: -lambda * sum_i a_i log a_i
    attn_entropy_lambda = float(training_params.get('attn_entropy_lambda', 1e-3))
    sample_cells_k = training_params.get('sample_cells_k', None)
    resample_with_replacement = bool(training_params.get('resample_with_replacement', True))
    type_prop_lambda = float(training_params.get('type_prop_lambda', 0.0))
    # Optional temperature schedules
    tau_intra_start = float(training_params.get('attn_tau_intra_start', _factory_kwargs.get('attn_tau_intra', 1.0)))
    tau_intra_end = float(training_params.get('attn_tau_intra_end', tau_intra_start))
    tau_inter_start = float(training_params.get('attn_tau_inter_start', _factory_kwargs.get('attn_tau_inter', 1.0)))
    tau_inter_end = float(training_params.get('attn_tau_inter_end', tau_inter_start))
    early_stop_metric = str(training_params.get('early_stop_metric', 'cindex')).lower()  # accepts 'cindex' | 'c_index' | 'loss'
    early_stop_metric = early_stop_metric.replace('_', '')
    if early_stop_metric not in ('cindex', 'loss'):
        print(f"Warning: unknown early_stop_metric '{training_params.get('early_stop_metric')}', defaulting to 'cindex'")
        early_stop_metric = 'cindex'
    # Smoothing for C-index based early stopping: use moving average window and keep the middle epoch
    cindex_smooth_window = int(training_params.get('cindex_smooth_window', 5))
    use_cindex_smoothing = (early_stop_metric == 'cindex' and cindex_smooth_window >= 2)

    # Build optimizer only over trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    # Optional LR scheduler configuration
    scheduler_type = training_params.get('lr_scheduler', None)  # 'plateau' | 'cosine' | None
    scheduler_monitor = training_params.get('scheduler_monitor', 'val_cindex' if early_stop_metric == 'cindex' else 'val_loss')  # 'val_loss' | 'val_cindex'
    scheduler = None
    if scheduler_type == 'plateau':
        factor = float(training_params.get('plateau_factor', 0.5))
        plateau_patience = int(training_params.get('plateau_patience', 10))
        min_lr_sched = float(training_params.get('plateau_min_lr', 1e-6))
        threshold = float(training_params.get('plateau_threshold', 1e-4))
        mode = 'max' if scheduler_monitor == 'val_cindex' else 'min'
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=plateau_patience,
            threshold=threshold,
            min_lr=min_lr_sched,
        )
    elif scheduler_type == 'cosine':
        t_max = int(training_params.get('cosine_t_max', epochs))
        eta_min = float(training_params.get('cosine_eta_min', 1e-6))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

    # Clean targets
    def clean_targets(duration, event, name):
        duration = np.array(duration, dtype=np.float32)
        event = np.array(event, dtype=np.float32)
        duration_invalid = np.isnan(duration) | np.isinf(duration) | (duration <= 0)
        event_invalid = np.isnan(event) | np.isinf(event)
        if np.any(duration_invalid):
            print(f"Warning: Found {np.sum(duration_invalid)} invalid duration values in {name}")
            duration[duration_invalid] = np.median(duration[~duration_invalid])
        if np.any(event_invalid):
            print(f"Warning: Found {np.sum(event_invalid)} invalid event values in {name}")
            event[event_invalid] = 0.0
        event = np.clip(event, 0, 1)
        print(f"{name}: duration range [{duration.min():.1f}, {duration.max():.1f}], events: {event.sum():.0f}/{len(event)}")
        return duration, event

    train_duration, train_event = clean_targets(train_target[0], train_target[1], "Train")
    early_duration, early_event = clean_targets(early_stop_target[0], early_stop_target[1], "Early stop")
    test_duration, test_event = clean_targets(test_target[0], test_target[1], "Test")

    # Datasets and loaders
    # Apply cell sampling only to the training set. Validation and test use full cells (no sampling).
    if sample_cells_k is not None and sample_cells_k > 0:
        print(f"Cell sampling enabled for training: K={sample_cells_k}. Validation/Test: no sampling.")
    train_ds = MILSurvivalDataset(
        X_train,
        train_duration,
        train_event,
        sample_cells_k=sample_cells_k,
        resample_with_replacement=resample_with_replacement,
        types=T_train,
    )
    val_ds = MILSurvivalDataset(
        X_early_stop,
        early_duration,
        early_event,
        sample_cells_k=None,
        resample_with_replacement=False,
        types=T_early_stop,
    )
    test_ds = MILSurvivalDataset(
        X_test,
        test_duration,
        test_event,
        sample_cells_k=None,
        resample_with_replacement=False,
        types=T_test,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0, collate_fn=collate_mil_batch)
    val_loader = DataLoader(val_ds, batch_size=max(1, batch_size), shuffle=False, drop_last=False, num_workers=0, collate_fn=collate_mil_batch)

    # Time discretization for non-cox losses
    bin_edges_np: Optional[np.ndarray] = None
    if loss_type in ('deephit', 'mtlr'):
        # Build bins using only training durations
        bin_edges_np = make_time_bins(train_duration, num_bins=num_time_bins, strategy=time_bins_strategy)

    # IPCW for Uno auxiliary term
    ghat_fn = None
    if loss_type == 'cox' and aux_cindex_weight > 0:
        ghat_fn = build_ipcw_ghat(train_duration, train_event)

    # Early stopping based on validation C-index (higher is better)
    train_losses: list = []
    val_losses: list = []
    best_val_loss = float('inf')  # tracked at the epoch of best C-index (for reference)
    best_val_cindex = float('-inf')
    epochs_without_improve = 0
    best_epoch = 0
    best_state_dict = None
    # Track per-epoch validation C-index and a small in-memory ring buffer of recent checkpoints
    val_cindices: list = []
    recent_checkpoints: list = []  # list of dict(epoch, state_dict, val_cindex, val_loss)
    smooth_best = float('-inf')  # best moving-average C-index

    print(f"Training for max {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_ds.set_epoch(seed=epoch)
        # Anneal hierarchical attention temperatures if enabled
        if hasattr(model, 'use_hier') and model.use_hier and getattr(model, 'hier_pool', None) is not None:
            if epochs > 1:
                frac = epoch / float(max(epochs - 1, 1))
            else:
                frac = 1.0
            current_tau_intra = tau_intra_start + (tau_intra_end - tau_intra_start) * frac
            current_tau_inter = tau_inter_start + (tau_inter_end - tau_inter_start) * frac
            model.hier_pool.tau_intra = current_tau_intra
            model.hier_pool.tau_inter = current_tau_inter
        running_loss = 0.0
        num_batches = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch_data in enumerate(train_loader):
            if len(batch_data) == 4:
                x, d, e, t_ids = batch_data
            else:
                x, d, e = batch_data
                t_ids = None
            # Ensure batch has at least 2 patients for meaningful risk sets
            if x.size(0) < 2:
                continue
            x = x.to(device, non_blocking=True)
            d = d.to(device)
            e = e.to(device)
            if t_ids is not None:
                t_ids = t_ids.to(device)

            # Forward; request attention weights when we will use entropy reg
            if attn_entropy_lambda != 0.0 or type_prop_lambda != 0.0:
                out_tuple = model(x, type_ids=t_ids, return_attention=True)
                if isinstance(out_tuple, tuple) and len(out_tuple) == 3:
                    logits, attn_weights, l1_loss_val = out_tuple
                else:
                    # Fallback if model doesn't support return_attention
                    out = model(x)
                    logits, l1_loss_val = (out if isinstance(out, tuple) else (out, torch.zeros((), device=device)))
                    attn_weights = None
            else:
                out = model(x, type_ids=t_ids)
                if isinstance(out, tuple):
                    logits, l1_loss_val = out
                else:
                    logits, l1_loss_val = out, torch.zeros((), device=device)
                attn_weights = None

            if loss_type == 'cox':
                risk_scores = logits.view(-1)
                # TorchSurv expects log hazards, event (1/0), time
                base_loss = neg_partial_log_likelihood(
                    risk_scores, e.bool(), d, ties_method="efron"
                )
                aux_loss = torch.tensor(0.0, device=device)
                if ghat_fn is not None and aux_cindex_weight > 0:
                    with torch.no_grad():
                        pass
                    aux_loss = uno_pairwise_aux_loss(risk_scores, d, e, ghat_fn)
                loss = base_loss + aux_cindex_weight * aux_loss + l1_loss_val
            elif loss_type == 'deephit':
                loss = deephit_single_loss(logits, d, e, bin_edges_np) + l1_loss_val
            else:  # 'mtlr'
                loss = discrete_hazard_loss_mtlr(logits, d, e, bin_edges_np) + l1_loss_val

            # Attention entropy regularization (maximize entropy → subtract sum a log a)
            if attn_entropy_lambda != 0.0 and attn_weights is not None:
                # attn_weights shape: (B, N)
                pw = torch.clamp(attn_weights, min=1e-12)
                entropy = -torch.sum(pw * torch.log(pw), dim=1).mean()
                # We add negative entropy with minus sign to penalize low entropy
                # Objective adds: -lambda * sum a log a (so larger entropy reduces loss)
                loss = loss - attn_entropy_lambda * entropy
            # Type proportion regularization: encourage sum_i alpha_i within type ~ q_t (size proportion)
            if type_prop_lambda != 0.0 and (attn_weights is not None) and (t_ids is not None):
                # Only consider valid cells (types >= 0)
                valid_mask = (t_ids >= 0).float()
                weights = torch.clamp(attn_weights, min=0.0)
                weights = weights * valid_mask
                # Compute target proportions per sample
                # q_t = K_t / K_valid
                reg_terms = []
                B = t_ids.size(0)
                for b in range(B):
                    types_b = t_ids[b]  # (N,)
                    valid_b = (types_b >= 0)
                    if valid_b.sum() == 0:
                        continue
                    Kb = float(valid_b.sum().item())
                    uniq = torch.unique(types_b[valid_b])
                    for t in uniq:
                        mask_t = (types_b == t)
                        Kt = float(mask_t.sum().item())
                        q_t = Kt / max(Kb, 1.0)
                        sum_alpha_t = weights[b][mask_t].sum()
                        reg_terms.append((sum_alpha_t - q_t) ** 2)
                if len(reg_terms) > 0:
                    prop_reg = torch.stack(reg_terms).mean()
                    loss = loss + type_prop_lambda * prop_reg
            loss = loss / grad_accum_steps
            loss.backward()

            if ((step + 1) % grad_accum_steps == 0) or (step + 1 == len(train_loader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.detach().item()
            num_batches += 1

        epoch_train_loss = running_loss / max(1, num_batches)
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_running = 0.0
            val_batches = 0
            val_preds_list = []
            val_d_list = []
            val_e_list = []
            for batch_data in val_loader:
                if len(batch_data) == 4:
                    x, d, e, t_ids = batch_data
                else:
                    x, d, e = batch_data
                    t_ids = None
                # For validation loss, keep the Cox loss stable by requiring >=2 samples.
                # For C-index, we will align durations/events with predictions collected below.
                if x.size(0) < 2:
                    # Skip this batch entirely for both loss and predictions to keep shapes aligned
                    continue
                x = x.to(device, non_blocking=True)
                d = d.to(device)
                e = e.to(device)
                if attn_entropy_lambda != 0.0 or type_prop_lambda != 0.0:
                    out_tuple = model(x, type_ids=t_ids, return_attention=True)
                    if isinstance(out_tuple, tuple) and len(out_tuple) == 3:
                        logits, attn_weights, l1_loss_val = out_tuple
                    else:
                        out = model(x)
                        logits, l1_loss_val = (out if isinstance(out, tuple) else (out, torch.zeros((), device=device)))
                        attn_weights = None
                else:
                    out = model(x, type_ids=t_ids)
                    if isinstance(out, tuple):
                        logits, l1_loss_val = out
                    else:
                        logits, l1_loss_val = out, torch.zeros((), device=device)
                    attn_weights = None
                if loss_type == 'cox':
                    vloss = neg_partial_log_likelihood(
                        logits.view(-1), e.bool(), d, ties_method="efron"
                    ) + l1_loss_val
                elif loss_type == 'deephit':
                    vloss = deephit_single_loss(logits, d, e, bin_edges_np) + l1_loss_val
                else:
                    vloss = discrete_hazard_loss_mtlr(logits, d, e, bin_edges_np) + l1_loss_val

                # Apply same entropy regularization on validation loss (for early stopping consistency)
                if attn_entropy_lambda != 0.0 and attn_weights is not None:
                    pw = torch.clamp(attn_weights, min=1e-12)
                    entropy = -torch.sum(pw * torch.log(pw), dim=1).mean()
                    vloss = vloss - attn_entropy_lambda * entropy
                # Type proportion regularization also applied on validation (keeps consistency)
                if type_prop_lambda != 0.0 and (attn_weights is not None) and (t_ids is not None):
                    valid_mask = (t_ids >= 0).float()
                    weights = torch.clamp(attn_weights, min=0.0)
                    weights = weights * valid_mask
                    reg_terms = []
                    Bv = t_ids.size(0)
                    for b in range(Bv):
                        types_b = t_ids[b]
                        valid_b = (types_b >= 0)
                        if valid_b.sum() == 0:
                            continue
                        Kb = float(valid_b.sum().item())
                        uniq = torch.unique(types_b[valid_b])
                        for t in uniq:
                            mask_t = (types_b == t)
                            Kt = float(mask_t.sum().item())
                            q_t = Kt / max(Kb, 1.0)
                            sum_alpha_t = weights[b][mask_t].sum()
                            reg_terms.append((sum_alpha_t - q_t) ** 2)
                    if len(reg_terms) > 0:
                        prop_reg = torch.stack(reg_terms).mean()
                        vloss = vloss + type_prop_lambda * prop_reg
                val_running += vloss.item()
                val_batches += 1
                # Collect predictions and matching durations/events for C-index
                risks = deepsurv_like_risk_from_output(logits, loss_type, bin_edges_np)
                val_preds_list.append(risks.view(-1).detach().cpu().numpy())
                val_d_list.append(d.detach().cpu().numpy().reshape(-1))
                val_e_list.append(e.detach().cpu().numpy().reshape(-1))
            epoch_val_loss = val_running / max(1, val_batches)
            val_losses.append(epoch_val_loss)
            if len(val_preds_list) > 0:
                val_pred_epoch = np.concatenate(val_preds_list, axis=0)
                val_d_epoch = np.concatenate(val_d_list, axis=0).astype(np.float32)
                val_e_epoch = np.concatenate(val_e_list, axis=0).astype(np.float32)
                # Align shapes defensively (should already match)
                n = min(len(val_pred_epoch), len(val_d_epoch), len(val_e_epoch))
                val_pred_epoch = val_pred_epoch[:n]
                val_d_epoch = val_d_epoch[:n]
                val_e_epoch = val_e_epoch[:n]
                epoch_val_cindex = float(concordance_index(val_d_epoch, -val_pred_epoch, val_e_epoch))
            else:
                epoch_val_cindex = float('nan')

        current_lr = optimizer.param_groups[0]['lr']
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}, Val C-index: {epoch_val_cindex:.4f}, LR: {current_lr:.2e}")

        # Record current epoch metrics for smoothing logic
        val_cindices.append(epoch_val_cindex)

        # Early stopping and model saving
        if early_stop_metric == 'cindex' and use_cindex_smoothing:
            # Keep a small buffer of recent checkpoints
            checkpoint_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            recent_checkpoints.append({
                'epoch': epoch + 1,
                'state_dict': checkpoint_state,
                'val_cindex': epoch_val_cindex,
                'val_loss': epoch_val_loss,
            })
            if len(recent_checkpoints) > cindex_smooth_window:
                recent_checkpoints.pop(0)

            # Only evaluate improvement once we have a full window
            if len(val_cindices) >= cindex_smooth_window and len(recent_checkpoints) == cindex_smooth_window:
                window_vals = val_cindices[-cindex_smooth_window:]
                window_avg = float(np.nanmean(window_vals))
                mid_idx = cindex_smooth_window // 2
                mid_ckpt = recent_checkpoints[mid_idx]
                if window_avg > smooth_best + min_delta:
                    smooth_best = window_avg
                    best_epoch = int(mid_ckpt['epoch'])
                    best_val_cindex = float(mid_ckpt['val_cindex'])
                    best_val_loss = float(mid_ckpt['val_loss'])
                    best_state_dict = {k: v.clone() for k, v in mid_ckpt['state_dict'].items()}
                    if save_path:
                        torch.save(mid_ckpt['state_dict'], save_path)
                        print(
                            f"Saved best model at epoch {best_epoch} (middle of window size {cindex_smooth_window}), "
                            f"val C-index (middle) = {best_val_cindex:.6f}, window avg = {window_avg:.6f}"
                        )
                    epochs_without_improve = 0
                else:
                    epochs_without_improve += 1
                    if epochs_without_improve >= patience:
                        print(f"Early stopping on C-index (smoothed, window={cindex_smooth_window}) after {epochs_without_improve} epochs without improvement")
                        break
            else:
                # Not enough epochs to form a full window yet; do not count against patience
                pass
        else:
            # Default behavior (either monitoring loss, or cindex without smoothing)
            improved = False
            if early_stop_metric == 'cindex':
                if epoch_val_cindex > best_val_cindex + min_delta:
                    improved = True
            else:  # 'loss'
                if epoch_val_loss < best_val_loss - min_delta:
                    improved = True

            if improved:
                best_val_cindex = epoch_val_cindex
                best_val_loss = epoch_val_loss
                best_epoch = epoch + 1
                epochs_without_improve = 0
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if save_path:
                    torch.save(model.state_dict(), save_path)
                    metric_name = 'C-index' if early_stop_metric == 'cindex' else 'loss'
                    metric_value = epoch_val_cindex if early_stop_metric == 'cindex' else epoch_val_loss
                    print(f"Saved best model at epoch {best_epoch}, val {metric_name} = {metric_value:.6f}")
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= patience:
                    stop_metric_name = 'C-index' if early_stop_metric == 'cindex' else 'loss'
                    print(f"Early stopping on {stop_metric_name} after {epochs_without_improve} epochs without improvement")
                    break

        # Step LR scheduler after validation metrics are computed
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                monitor_value = epoch_val_cindex if scheduler_monitor == 'val_cindex' else epoch_val_loss
                scheduler.step(monitor_value)
            else:
                scheduler.step()

    # Load best weights
    if save_path and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        print(f"Loaded best model from epoch {best_epoch}")
    elif best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Evaluate C-index and integrated Brier score using dataset-specific predictions
    ibs_debug = bool(training_params.get('ibs_debug', False)) or (str(os.environ.get('IBS_DEBUG', '0')).strip() == '1')
    def _ibs_dbg(*args):
        if ibs_debug:
            print("[IBS]", *args)
    def _collect_predictions(ds: MILSurvivalDataset, bs: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        loader = DataLoader(ds, batch_size=bs, shuffle=False, drop_last=False, collate_fn=collate_mil_batch)
        risk_list = []
        extra_list = []
        model.eval()
        with torch.no_grad():
            for batch_data in loader:
                if len(batch_data) == 4:
                    x, _, _, t_ids = batch_data
                else:
                    x, _, _ = batch_data
                    t_ids = None
                x = x.to(device)
                t_ids_t = None if t_ids is None else t_ids.to(device)
                out = model(x, type_ids=t_ids_t)
                if isinstance(out, tuple):
                    out = out[0]
                risks = deepsurv_like_risk_from_output(out, loss_type, bin_edges_np)
                risk_list.append(risks.view(-1).cpu().numpy())
                extra_array: np.ndarray
                if loss_type == 'cox':
                    extra_array = out.view(-1).cpu().numpy().copy()
                elif loss_type == 'deephit':
                    probs = torch.softmax(out, dim=-1).cpu().numpy()
                    extra_array = probs.copy()
                else:  # mtlr
                    hazards = torch.sigmoid(out).cpu().numpy()
                    extra_array = hazards.copy()
                extra_list.append(extra_array)
        risks_all = np.concatenate(risk_list, axis=0) if risk_list else np.array([], dtype=np.float64)
        extra_all = np.concatenate(extra_list, axis=0) if extra_list else np.array([], dtype=np.float64)
        _ibs_dbg("collect_predictions:", "loss=", loss_type, "risks_shape=", risks_all.shape, "extra_shape=", extra_all.shape, "num_time_bins=", (None if bin_edges_np is None else (bin_edges_np.shape[0]-1)))
        return risks_all, extra_all

    print("Evaluating model...")
    train_pred, train_extra = _collect_predictions(train_ds)
    val_pred, val_extra = _collect_predictions(val_ds)
    test_pred, test_extra = _collect_predictions(test_ds)

    train_c_index = float(concordance_index(train_duration, -train_pred, train_event))
    early_stop_c_index = float(concordance_index(early_duration, -val_pred, early_event))
    test_c_index = float(concordance_index(test_duration, -test_pred, test_event))

    def _build_survival_dataframe(extra: np.ndarray, durations: np.ndarray, events: np.ndarray) -> Optional[pd.DataFrame]:
        durations = np.asarray(durations, dtype=np.float64)
        events = np.asarray(events, dtype=np.int64)
        if durations.size == 0 or extra.size == 0:
            _ibs_dbg("surv_df: empty durations or extra", durations.size, extra.size)
            return None
        if loss_type == 'cox':
            log_risk = extra.reshape(-1).astype(np.float64)
            event_mask = events > 0
            if not np.any(event_mask):
                _ibs_dbg("surv_df: no events")
                return None
            event_times = np.unique(durations[event_mask])
            if event_times.size == 0:
                _ibs_dbg("surv_df: no event_times after unique")
                return None
            event_times.sort()
            exp_risk = np.exp(np.clip(log_risk, -50.0, 50.0))
            cum_hazards = []
            cumulative = 0.0
            for t in event_times:
                at_risk = exp_risk[durations >= t]
                risk_sum = at_risk.sum()
                if risk_sum <= 0:
                    continue
                d_i = events[(durations == t) & event_mask].sum()
                if d_i <= 0:
                    continue
                cumulative += float(d_i) / max(risk_sum, 1e-12)
                cum_hazards.append(cumulative)
            if len(cum_hazards) == 0:
                _ibs_dbg("surv_df: empty cum_hazards")
                return None
            cum_hazards = np.asarray(cum_hazards, dtype=np.float64)
            surv_matrix = np.exp(-np.outer(exp_risk, cum_hazards))
            surv_matrix = np.vstack([np.ones(exp_risk.shape[0]), surv_matrix.T])
            timeline = np.concatenate(([0.0], event_times.astype(np.float64)))
            _ibs_dbg("surv_df(cox): timeline_len=", timeline.shape[0], "cols=", surv_matrix.shape[0])
            return pd.DataFrame(surv_matrix, index=timeline)
        if bin_edges_np is None:
            _ibs_dbg("surv_df: bin_edges_np is None for", loss_type)
            return None
        if loss_type == 'deephit':
            probs = np.asarray(extra, dtype=np.float64)
            if probs.ndim != 2 or probs.shape[1] == 0:
                _ibs_dbg("surv_df(deephit): probs bad shape", probs.shape)
                return None
            surv_matrix = 1.0 - np.cumsum(probs, axis=1)
            surv_matrix = np.clip(surv_matrix, 0.0, 1.0)
            surv_matrix = np.hstack([np.ones((probs.shape[0], 1)), surv_matrix])
            if bin_edges_np.shape[0] - 1 != probs.shape[1]:
                _ibs_dbg("surv_df(deephit): bins mismatch", (bin_edges_np.shape[0]-1), probs.shape[1])
                return None
            timeline = bin_edges_np.astype(np.float64)
            _ibs_dbg("surv_df(deephit): timeline_len=", timeline.shape[0], "cols=", surv_matrix.shape[0])
            return pd.DataFrame(surv_matrix.T, index=timeline)
        # mtlr
        hazards = np.asarray(extra, dtype=np.float64)
        if hazards.ndim != 2 or hazards.shape[1] == 0:
            _ibs_dbg("surv_df(mtlr): hazards bad shape", hazards.shape)
            return None
        num_bins = bin_edges_np.shape[0] - 1
        if hazards.shape[1] != num_bins:
            _ibs_dbg("surv_df(mtlr): bins mismatch", num_bins, hazards.shape[1])
            return None
        hazards = np.clip(hazards, 1e-6, 1 - 1e-6)
        surv_matrix = np.cumprod(1.0 - hazards, axis=1)
        surv_matrix = np.clip(surv_matrix, 0.0, 1.0)
        surv_matrix = np.hstack([np.ones((hazards.shape[0], 1)), surv_matrix])
        timeline = bin_edges_np.astype(np.float64)
        _ibs_dbg("surv_df(mtlr): timeline_len=", timeline.shape[0], "cols=", surv_matrix.shape[0])
        return pd.DataFrame(surv_matrix.T, index=timeline)

    def _compute_ibs(extra: np.ndarray, durations: np.ndarray, events: np.ndarray) -> float:
        surv_df = _build_survival_dataframe(extra, durations, events)
        if surv_df is None or len(surv_df.index) < 2:
            _ibs_dbg("compute_ibs: surv_df missing or too short")
            return float('nan')
        times = surv_df.index.values.astype(np.float64)
        positive_times = times[times > 0]
        if positive_times.size == 0:
            _ibs_dbg("compute_ibs: no positive times in timeline")
            return float('nan')
        lower = positive_times[0]
        upper = times[-1]
        # Clip upper bound where censoring KM > 0 to avoid infinite IPCW weights
        try:
            km_c = KaplanMeierFitter()
            durations_np_c = np.asarray(durations, dtype=np.float64)
            events_np_c = np.asarray(events, dtype=np.int64)
            km_c.fit(durations_np_c, event_observed=1 - events_np_c)
            km_times = km_c.survival_function_.index.values.astype(np.float64)
            km_vals = km_c.survival_function_.values.reshape(-1).astype(np.float64)
            # Find last time where G(t) > eps
            eps_g = 1e-6
            if km_vals.size > 0:
                pos_idx = np.where(km_vals > eps_g)[0]
                if pos_idx.size > 0:
                    last_pos_time = float(km_times[pos_idx[-1]])
                    _ibs_dbg("KM censoring: last G(t)>eps at", last_pos_time, "orig_upper=", upper)
                    if np.isfinite(last_pos_time):
                        upper = min(upper, last_pos_time)
        except Exception as e:
            _ibs_dbg("KM censoring fit error:", repr(e))
        if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
            _ibs_dbg("compute_ibs: invalid bounds", lower, upper)
            return float('nan')
        durations_np = np.asarray(durations, dtype=np.float64)
        events_np = np.asarray(events, dtype=np.int64)
        try:
            eval_surv = EvalSurv(surv_df, durations_np, events_np, censor_surv='km')
            grid = times[(times >= lower) & (times <= upper)]
            if grid.size < 2:
                _ibs_dbg("compute_ibs: grid too small", grid.size)
                return float('nan')
            bs = eval_surv.brier_score(grid)
            ibs = float(np.trapz(bs.values.astype(np.float64), x=grid) / (grid[-1] - grid[0]))
            _ibs_dbg("compute_ibs: OK lower=", lower, "upper=", upper, "grid_n=", grid.size, "ibs=", ibs)
            return ibs
        except Exception as e:
            _ibs_dbg("EvalSurv IBS error:", repr(e), "bounds:", (lower, upper), "grid_n:", int((times[(times >= lower) & (times <= upper)]).size),
                     "dur_range:", (float(np.min(durations_np)), float(np.max(durations_np))), "events=", int(np.sum(events_np)))
            return float('nan')

    train_ibs = _compute_ibs(train_extra, train_duration, train_event)
    early_stop_ibs = _compute_ibs(val_extra, early_duration, early_event)
    test_ibs = _compute_ibs(test_extra, test_duration, test_event)

    results = {
        'train_c_index': train_c_index,
        'early_stop_c_index': early_stop_c_index,
        'test_c_index': test_c_index,
        'train_integrated_brier_score': train_ibs,
        'early_stop_integrated_brier_score': early_stop_ibs,
        'test_integrated_brier_score': test_ibs,
        'best_val_loss': best_val_loss,
        'best_val_c_index': best_val_cindex,
        'best_epoch': best_epoch if best_epoch > 0 else epochs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'model': model,  # return the trained PyTorch model
        'mil_model': model,
        'early_stop_metric': early_stop_metric,
    }

    print("\nTraining completed!")
    print(f"Train C-index: {train_c_index:.4f}")
    print(f"Early stop C-index: {early_stop_c_index:.4f}")
    print(f"Test C-index: {test_c_index:.4f}")
    print(f"Train Integrated Brier Score: {train_ibs:.4f}")
    print(f"Early stop Integrated Brier Score: {early_stop_ibs:.4f}")
    print(f"Test Integrated Brier Score: {test_ibs:.4f}")
    if early_stop_metric == 'cindex':
        print(f"Best validation C-index: {best_val_cindex:.4f} at epoch {results['best_epoch']}")
    else:
        print(f"Best validation loss: {best_val_loss:.6f} at epoch {results['best_epoch']}")

    return results


def plot_training_curves(train_losses: list, val_losses: list, save_path: Optional[str] = None):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', alpha=0.8)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', alpha=0.8)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.show()


def plot_losses_from_results(results: Dict[str, Any], save_path: Optional[str] = None):
    """Plot loss curves directly from the results dict returned by training.
    Args:
        results: Dict containing 'train_losses' and 'val_losses'
        save_path: Optional path to save the figure
    """
    train_losses = results.get('train_losses', [])
    val_losses = results.get('val_losses', [])
    if not train_losses or not val_losses:
        print("No losses found in results to plot.")
        return
    plot_training_curves(train_losses, val_losses, save_path)


# Example usage
if __name__ == "__main__":
    # This is an example of how to use the training pipeline
    # You would replace this with your actual data loading
    
    print("MIL Survival Training Pipeline Example")
    
    # Example model parameters
    model_params = {
        'num_genes': 2000,  # Will be updated based on actual data
        'cell_embed_dim': 128,
        'cell_encoder_hidden': [512, 256],
        'num_attention_heads': 8,
        'num_attention_layers': 2,
        'dropout': 0.1,
        'l1_reg': 1e-5  # Small L1 regularization for gene selection
    }
    
    # Example training parameters
    training_params = {
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'batch_size': 16,  # Small batch size for memory
        'epochs': 100,
        'patience': 15,
        'min_delta': 1e-4
    }
    
    print("Configuration:")
    print("Model parameters:", model_params)
    print("Training parameters:", training_params)
    
    # Note: To run this with your actual data, you would need to:
    # 1. Load your decoded_samples_filtered from the notebook
    # 2. Create a survival_data DataFrame with columns ['sample_id', 'duration', 'event']
    # 3. Call prepare_mil_data() and train_mil_survival_model()
    
    print("\nTo use with your data:")
    print("1. Load decoded_samples_filtered from your notebook")
    print("2. Prepare survival data with sample_id, duration, event columns")
    print("3. Call prepare_mil_data() and train_mil_survival_model()")
    print("4. Use plot_training_curves() to visualize training progress")
