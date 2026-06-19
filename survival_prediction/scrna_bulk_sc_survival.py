import os
import sys
from pathlib import Path

# Add DeSCENT scgep_generation to path for VAE, guided_diffusion imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_DESCENT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_DESCENT_ROOT / "scgep_generation"))

import json
import time
import math
import argparse
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Reuse existing components when possible
sys.path.append(os.path.dirname(__file__))
from mil_survival_model import CellEncoder, MultiHeadSelfAttention, AttentionPooling
from VAE.BulkEncoder import BulkEncoder as VAEBulkEncoder, CellEncoderSymmetric
from VAE.VAE_model import VAE
from mil_survival_training import (
    deephit_single_loss,
    discrete_hazard_loss_mtlr,
    deepsurv_like_risk_from_output,
)


class BulkMLPEncoder(nn.Module):
    """Lightweight BulkEncoder with configurable MLP.

    Default (when hidden_dims is None):
    LayerNorm → Linear(G→512) → GELU → Dropout(0.2) → Linear(512→256) → GELU → Linear(256→d) → LayerNorm
    """
    def __init__(self, num_genes: int, out_dim: int = 256, hidden_dims: Optional[List[int]] = None, dropout: float = 0.2):
        super().__init__()
        mlp_hidden = hidden_dims if (hidden_dims is not None and len(hidden_dims) > 0) else [512, 256]
        layers: List[nn.Module] = [nn.LayerNorm(num_genes)]
        in_dim = num_genes
        for i, h in enumerate(mlp_hidden):
            layers.append(nn.Linear(in_dim, int(h)))
            layers.append(nn.GELU())
            # Apply dropout after hidden layers
            layers.append(nn.Dropout(dropout))
            in_dim = int(h)
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.LayerNorm(out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionFusion(nn.Module):
    """Bulk-guided cross-attention: h_B as query, Z_sc as key/value → h_fuse."""
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.kv_proj = nn.Linear(embed_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.out = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, h_bulk: torch.Tensor, z_cells: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # h_bulk: [B, d]; z_cells: [B, N, d]; mask: [B, N] 1=valid
        q = self.q_proj(h_bulk).unsqueeze(1)  # [B,1,d]
        kv = self.kv_proj(z_cells)            # [B,N,d]
        attn_mask = None
        if mask is not None:
            # convert to boolean mask where True indicates "to be ignored"
            attn_mask = ~mask.bool()  # [B,N]
        fused, _ = self.attn(query=q, key=kv, value=kv, key_padding_mask=attn_mask)
        fused = fused.squeeze(1)  # [B,d]
        return self.out(fused)


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)


class ConcatMLPFusion(nn.Module):
    """Early fusion via concatenation: [h_B; h_S] → MLP → h_fuse (d).

    Input: two vectors in R^d; output: vector in R^d.
    """
    def __init__(self, embed_dim: int, dropout: float = 0.1, hidden_dims: Optional[List[int]] = None):
        super().__init__()
        in_dim0 = embed_dim * 2
        if hidden_dims is None:
            # Preserve previous default structure when not specified
            self.proj = nn.Sequential(
                nn.LayerNorm(in_dim0),
                nn.Linear(in_dim0, in_dim0),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(in_dim0, embed_dim),
                nn.LayerNorm(embed_dim),
            )
        else:
            layers: List[nn.Module] = [nn.LayerNorm(in_dim0)]
            in_dim = in_dim0
            for h in hidden_dims:
                layers.append(nn.Linear(in_dim, int(h)))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
                in_dim = int(h)
            layers.append(nn.Linear(in_dim, embed_dim))
            layers.append(nn.LayerNorm(embed_dim))
            self.proj = nn.Sequential(*layers)

    def forward(self, h_bulk: torch.Tensor, h_sc_pooled: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h_bulk, h_sc_pooled], dim=1)
        return self.proj(x)


def info_nce_loss(h_bulk: torch.Tensor, h_sc: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    # h_*: [B, d], assumed unnormalized
    h_b = l2_normalize(h_bulk)
    h_s = l2_normalize(h_sc)
    logits = (h_b @ h_s.t()) / float(temperature)
    labels = torch.arange(h_b.size(0), device=h_b.device)
    return F.cross_entropy(logits, labels)


def hard_negative_matching_loss(h_bulk: torch.Tensor, h_sc: torch.Tensor) -> torch.Tensor:
    # Create logits with pos and most similar negative for each instance
    h_b = l2_normalize(h_bulk)
    h_s = l2_normalize(h_sc)
    sims = h_b @ h_s.t()  # [B,B]

    # Most similar negative
    sims_no_diag = sims - torch.diag_embed(torch.diag(sims))
    neg_idx = sims_no_diag.argmax(dim=1)  # [B]


    pos = torch.sum(h_b * h_s, dim=1, keepdim=True)         # [B,1]
    neg = torch.sum(h_b * h_s[neg_idx], dim=1, keepdim=True)  # [B,1]
    
    two_class_logits = torch.cat([pos, neg], dim=1)  # [B,2], col0=pos, col1=neg
    labels = torch.zeros(h_b.size(0), dtype=torch.long, device=h_b.device)
    return F.cross_entropy(two_class_logits, labels)


def load_vae(device: torch.device, num_genes: int, ckpt_path: str) -> VAE:
    """Load pretrained VAE for decoding cell embeddings to gene space.

    Parameters
    - num_genes: number of genes for the VAE decoder output
    - ckpt_path: path to the pretrained VAE checkpoint (.pt)
    """
    autoencoder = VAE(
        num_genes=int(num_genes),
        device='cuda' if device.type == 'cuda' else 'cpu',
        seed=0,
        loss_ae='mse',
        hidden_dim=128,
        decoder_activation='ReLU',
    )
    state = torch.load(ckpt_path, map_location=device)
    autoencoder.load_state_dict(state)
    autoencoder.to(device)
    autoencoder.eval()
    return autoencoder


class BulkSCFusionSurvival(nn.Module):
    """Dual-modality model: BulkEncoder + CellEncoder(+SA) + Fusion + Cox head."""
    def __init__(
        self,
        num_genes: int,
        embed_dim: int = 256,
        cell_encoder_hidden: List[int] = [512, 256],
        num_attention_heads: int = 8,
        num_attention_layers: int = 2,
        dropout: float = 0.1,
        bulk_encoder_type: str = 'vae_mlp',  # 'vae_mlp' | 'light_mlp'
        cell_encoder_type: str = 'mil',      # 'mil' | 'symmetric'
        fusion_type: str = 'cross_attn',     # 'cross_attn' | 'concat_mlp'
        pooling: str = 'attn',               # 'attn' | 'mean'
        risk_head: str = 'mlp',              # 'mlp' | 'cosine'
        output_mode: str = 'cox',            # 'cox' | 'deephit' | 'mtlr'
        num_time_bins: int = 50,
        concat_sc_with_fusion: bool = False, # if True, concat pooled h_s with fusion vector
        residual_hfuse_to_bulk: bool = False, # if True, add h_bulk as residual to h_fuse (h_fuse += h_bulk)
        # New: configurable MLP structures
        bulk_mlp_hidden: Optional[List[int]] = None,
        bulk_mlp_dropout: float = 0.2,
        fusion_mlp_hidden: Optional[List[int]] = None,
        pred_head_hidden: Optional[List[int]] = None,
        use_sc: bool = True,
    ):
        super().__init__()
        assert fusion_type in ('cross_attn', 'concat_mlp')
        assert pooling in ('attn', 'mean')
        assert risk_head in ('mlp', 'cosine')
        assert output_mode in ('cox', 'deephit', 'mtlr')
        assert cell_encoder_type in ('mil', 'symmetric')
        self.fusion_type = fusion_type
        self.pooling = pooling
        self.risk_head = risk_head
        self.output_mode = output_mode
        self.num_time_bins = int(num_time_bins)
        self.concat_sc_with_fusion = bool(concat_sc_with_fusion)
        self.residual_hfuse_to_bulk = bool(residual_hfuse_to_bulk)
        self.cell_encoder_type = cell_encoder_type
        self.use_sc = bool(use_sc)
        # Bulk encoder
        if bulk_encoder_type == 'light_mlp':
            self.bulk_encoder = BulkMLPEncoder(num_genes=num_genes, out_dim=embed_dim, hidden_dims=bulk_mlp_hidden, dropout=bulk_mlp_dropout)
        else:
            # Reuse VAEBulkEncoder but project to embed_dim if needed
            self.bulk_encoder = VAEBulkEncoder(num_genes=num_genes, latent_dim=embed_dim, hidden_dim=[1024, 1024], dropout=0.5, input_dropout=0.1, residual=False)

        # Cell encoder + self-attention stack
        if self.cell_encoder_type == 'symmetric':
            # Mirror BulkEncoder structure and normalization
            self.cell_encoder = CellEncoderSymmetric(num_genes=num_genes, latent_dim=embed_dim, hidden_dim=[1024, 1024], dropout=0.5, input_dropout=0.1, residual=False)
        else:
            self.cell_encoder = CellEncoder(input_dim=num_genes, hidden_dims=cell_encoder_hidden, output_dim=embed_dim, dropout=dropout)
        self.attention_layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': MultiHeadSelfAttention(embed_dim, num_attention_heads, dropout),
                'norm1': nn.LayerNorm(embed_dim),
                'ff': nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim * 2, embed_dim),
                ),
                'norm2': nn.LayerNorm(embed_dim),
            })
            for _ in range(num_attention_layers)
        ])

        # Pooling for h_S (used for alignment losses and optional concat)
        self.attn_pool = AttentionPooling(embed_dim, hidden_dim=128)

        # Fusion options
        if fusion_type == 'concat_mlp':
            self.fusion = ConcatMLPFusion(embed_dim=embed_dim, dropout=dropout, hidden_dims=fusion_mlp_hidden)
        else:
            self.fusion = CrossAttentionFusion(embed_dim=embed_dim, num_heads=num_attention_heads, dropout=dropout)

        # Heads: scalar risk (cox) and discrete-time logits (deephit/mtlr)
        # If we concatenate [h_fuse; h_s], the head input dim becomes 2*embed_dim
        head_in_dim = embed_dim * 2 if (self.use_sc and self.concat_sc_with_fusion) else embed_dim
        # Prediction head (configurable). If pred_head_hidden is empty list, use direct Linear.
        pred_hidden_dims = pred_head_hidden if (pred_head_hidden is not None) else [head_in_dim // 2]
        pred_layers: List[nn.Module] = []
        in_dim = head_in_dim
        if len(pred_hidden_dims) > 0:
            for h in pred_hidden_dims:
                pred_layers.append(nn.Linear(in_dim, int(h)))
                pred_layers.append(nn.GELU())
                pred_layers.append(nn.Dropout(dropout))
                in_dim = int(h)
        self.pred_head = nn.Sequential(*(
            pred_layers + [nn.Linear(in_dim, 1)]
        ))
        self.time_head = nn.Linear(head_in_dim, self.num_time_bins)

        # Masked gene reconstruction head: conditioned on bulk embedding
        # Input dimension = num_genes (cell) + embed_dim (bulk)
        recon_hidden = max(512, (num_genes + embed_dim) // 2)
        self.mask_recon_head = nn.Sequential(
            nn.Linear(num_genes + embed_dim, recon_hidden),
            nn.GELU(),
            nn.Linear(recon_hidden, num_genes),
        )

    def forward_outputs(self, x_bulk: torch.Tensor, x_sc: torch.Tensor, mask: Optional[torch.Tensor] = None, return_logits: bool = False):
        # x_bulk: [B,G], x_sc: [B,N,G]
        h_b = self.bulk_encoder(x_bulk)  # [B,d]
        if not self.use_sc:
            h_s = torch.zeros_like(h_b)
            attn_weights = None
            head_input = h_b
            if self.output_mode == 'cox':
                risk = self.pred_head(head_input).squeeze(-1)
                logits = None
            else:
                logits = self.time_head(head_input)
                risk = deepsurv_like_risk_from_output(logits, self.output_mode, None)
            if return_logits:
                return risk, h_b, h_s, attn_weights, logits
            return risk, h_b, h_s, attn_weights

        # Encode cells → self-attn → pooling
        z = self.cell_encoder(x_sc)      # [B,N,d]
        for layer in self.attention_layers:
            attn_out = layer['self_attn'](z, mask)
            z = layer['norm1'](z + attn_out)
            ff_out = layer['ff'](z)
            z = layer['norm2'](z + ff_out)
        if self.pooling == 'attn':
            # Compute attention pooling only when needed.
            # Needed in training (alignment losses), or when downstream uses h_s:
            # - concat_mlp fusion (requires pooled h_s)
            # - concat_sc_with_fusion (head input concatenates h_s)
            # - cosine risk head (uses h_s for risk)
            need_hs = (
                self.training
                or self.fusion_type == 'concat_mlp'
                or self.concat_sc_with_fusion
                or self.risk_head == 'cosine'
            )
            if need_hs:
                h_s, attn_weights = self.attn_pool(z, mask)  # [B,d], [B,N]
            else:
                # Eval-time fast path for cross-attn fusion without h_s usage
                B = z.size(0)
                d = z.size(-1)
                h_s = torch.zeros(B, d, device=z.device, dtype=z.dtype)
                attn_weights = None
        else:
            # Mean pooling with mask over valid cells
            if mask is None:
                h_s = z.mean(dim=1)
            else:
                m = mask.unsqueeze(-1)  # [B,N,1]
                h_s = (z * m).sum(dim=1) / (m.sum(dim=1) + 1e-6)
            attn_weights = None
        # Fusion
        if self.fusion_type == 'concat_mlp':
            h_fuse = self.fusion(h_b, h_s)
        else:
            h_fuse = self.fusion(h_b, z, mask)           # [B,d]
        # Residual: optionally add bulk embedding to fusion output
        if self.residual_hfuse_to_bulk:
            h_fuse = h_fuse + h_b
        # Optionally concatenate pooled cell vector with fusion output
        if self.concat_sc_with_fusion:
            head_input = torch.cat([h_fuse, h_s], dim=1)  # [B, 2d]
        else:
            head_input = h_fuse
        # Outputs depending on mode
        logits = None
        if self.output_mode == 'cox':
            if self.risk_head == 'cosine':
                risk = torch.sum(l2_normalize(h_b) * l2_normalize(h_s), dim=1)
            else:
                risk = self.pred_head(head_input).squeeze(-1)    # [B]
        else:
            logits = self.time_head(head_input)  # [B, T]
            # Map to scalar risk for ranking (bin_edges not provided → index-based centers)
            risk = deepsurv_like_risk_from_output(logits, self.output_mode, None)
        if return_logits:
            return risk, h_b, h_s, attn_weights, logits
        return risk, h_b, h_s, attn_weights

    def forward(self, x_bulk: torch.Tensor, x_sc: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward_outputs(x_bulk, x_sc, mask, return_logits=False)

    def masked_gene_reconstruction_loss(self, x_sc: torch.Tensor, h_bulk: torch.Tensor, mask_ratio: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute masked gene reconstruction loss using a reusable head.
        x_sc: [B,N,G], h_bulk: [B,d]
        """
        B, N, G = x_sc.shape
        device = x_sc.device
        num_mask = max(1, int(G * mask_ratio))
        gene_indices = torch.randperm(G, device=device)[:num_mask]
        mask = torch.zeros(B, N, G, device=device, dtype=torch.bool)
        mask[:, :, gene_indices] = True
        # Build conditional input [x_cell, h_bulk]
        recon_input = torch.cat([x_sc, h_bulk.unsqueeze(1).expand(B, N, h_bulk.size(1))], dim=2)  # [B,N,G+d]
        recon = self.mask_recon_head(recon_input)
        loss = F.smooth_l1_loss(recon[mask], x_sc[mask])
        return loss, mask.any(dim=1).any(dim=1).float().mean()


def cox_ph_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    # Sort by time desc
    order = torch.argsort(time, descending=True)
    r = risk[order]
    e = event[order].to(torch.bool)
    # If no events in this batch, return 0 to avoid NaN
    if e.numel() == 0 or torch.count_nonzero(e) == 0:
        return risk.sum() * 0.0
    cum_logsumexp = torch.logcumsumexp(r, dim=0)
    log_risk = r - cum_logsumexp
    return -(log_risk[e]).mean()


class PatientBatchDataset(Dataset):
    """Dataset yielding per-patient (bulk, single-cell, time, event)."""
    def __init__(
        self,
        patient_ids: List[str],
        sc_samples: Dict[str, Dict],
        bulk_df: pd.DataFrame,
        durations: Dict[str, float],
        events: Dict[str, int],
        max_cells: int = 2048,
        seed: int = 42,
    ):
        self.ids = patient_ids
        self.sc = sc_samples
        self.bulk = bulk_df
        self.time = durations
        self.event = events
        self.max_cells = int(max_cells)
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        pid = self.ids[idx]
        sc_entry = self.sc[pid]
        X_df: pd.DataFrame = sc_entry['X_df']  # cells x genes (common order across modalities)
        cells = X_df.values.astype(np.float32)
        if cells.shape[0] > self.max_cells:
            sel = self.rng.choice(cells.shape[0], self.max_cells, replace=False)
            cells = cells[sel]
            if sc_entry.get('type_ids') is not None:
                type_ids = sc_entry['type_ids'][sel]
            else:
                type_ids = None
        else:
            type_ids = sc_entry.get('type_ids')
        x_sc = torch.from_numpy(cells)  # [N,G]
        # Build mask and pad to max_cells within batch later via collate_fn
        x_bulk = torch.from_numpy(self.bulk.loc[pid].values.astype(np.float32))  # [G]
        duration = float(self.time[pid])
        event = int(self.event[pid])
        sample = {
            'x_bulk': x_bulk,
            'x_sc': x_sc,
            'type_ids': None if type_ids is None else torch.from_numpy(type_ids.astype(np.int64)),
            'duration': torch.tensor(duration, dtype=torch.float32),
            'event': torch.tensor(event, dtype=torch.float32),
        }
        return sample


def pad_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    # Determine max N in this batch
    B = len(batch)
    G = batch[0]['x_bulk'].numel()
    Ns = [b['x_sc'].shape[0] for b in batch]
    Nmax = max(Ns)
    x_sc = torch.zeros(B, Nmax, G, dtype=torch.float32)
    mask = torch.zeros(B, Nmax, dtype=torch.float32)
    type_ids_all = []
    for i, b in enumerate(batch):
        n = b['x_sc'].shape[0]
        x_sc[i, :n] = b['x_sc']
        mask[i, :n] = 1.0
        type_ids = b['type_ids']
        if type_ids is None:
            type_ids_all.append(torch.full((Nmax,), -1, dtype=torch.int64))
        else:
            padded = torch.full((Nmax,), -1, dtype=torch.int64)
            padded[:n] = type_ids
            type_ids_all.append(padded)
    x_bulk = torch.stack([b['x_bulk'] for b in batch], dim=0)
    duration = torch.stack([b['duration'] for b in batch], dim=0)
    event = torch.stack([b['event'] for b in batch], dim=0)
    type_ids_tensor = torch.stack(type_ids_all, dim=0)
    return {
        'x_bulk': x_bulk,
        'x_sc': x_sc,
        'mask': mask,
        'type_ids': type_ids_tensor,
        'duration': duration,
        'event': event,
    }


def load_survival_fold(fold_num: int, base_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_path = os.path.join(base_dir, f"train_data_{fold_num}.csv")
    val_path = os.path.join(base_dir, f"val_data_{fold_num}.csv")
    train = pd.read_csv(train_path).rename(columns={'Unnamed: 0': 'tcga_barcode', 'OS': 'event', 'OS.time': 'duration'})
    val = pd.read_csv(val_path).rename(columns={'Unnamed: 0': 'tcga_barcode', 'OS': 'event', 'OS.time': 'duration'})
    return train, val


def extract_tcga_from_key(sample_key: str) -> Optional[str]:
    # Try to retrieve substring starting at 'TCGA-'
    if 'TCGA-' in sample_key:
        idx = sample_key.find('TCGA-')
        return sample_key[idx: idx + 12] if len(sample_key) >= idx + 12 else sample_key[idx:]
    return None


def build_bulk_matrix_for_samples(bulk_df: pd.DataFrame, sc_samples: Dict[str, Dict]) -> pd.DataFrame:
    # Reindex bulk to sc sample keys, attempting TCGA barcode matching
    mapping = {}
    for key in sc_samples.keys():
        tcga = extract_tcga_from_key(key)
        if tcga is not None:
            # Prefer exact index match first
            if tcga in bulk_df.index:
                mapping[key] = tcga
                continue
            # Try relaxed: any index that startswith tcga
            candidates = [idx for idx in bulk_df.index if str(idx).startswith(tcga)]
            if len(candidates) > 0:
                mapping[key] = candidates[0]
                continue
        # Fallback: exact key match
        if key in bulk_df.index:
            mapping[key] = key
    common_keys = [k for k in sc_samples.keys() if k in mapping]
    if len(common_keys) == 0:
        raise ValueError("No overlapping patients between sc samples and bulk_df after mapping.")
    bulk_aligned = bulk_df.loc[[mapping[k] for k in common_keys]].copy()
    bulk_aligned.index = common_keys
    return bulk_aligned


def train_one_epoch(
    model: BulkSCFusionSurvival,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_weights: Dict[str, float],
    gene_idx_tensor: Optional[torch.Tensor] = None,
    loss_type: str = 'cox',
    bin_edges: Optional[np.ndarray] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> Dict[str, float]:
    model.train()
    total = {k: 0.0 for k in ['loss', 'cox', 'nce', 'match', 'mask']}
    count = 0
    for batch in loader:
        x_bulk = batch['x_bulk'].to(device)
        x_sc = batch['x_sc'].to(device)
        mask = batch['mask'].to(device)
        time_t = batch['duration'].to(device)
        event_t = batch['event'].to(device)

        optimizer.zero_grad()
        # Get outputs (and logits when needed)
        out = model.forward_outputs(x_bulk, x_sc, mask, return_logits=True)
        risk, h_b, h_s, _, logits = out
        # Survival loss depending on type
        if str(loss_type).lower() == 'cox':
            # Clamp risk to avoid exploding values early in training
            risk = torch.clamp(risk, min=-50.0, max=50.0)
            loss_surv = cox_ph_loss(risk, time_t, event_t)
        elif str(loss_type).lower() == 'deephit':
            if bin_edges is None or logits is None:
                raise ValueError('bin_edges must be provided and logits must be available for DeepHit loss')
            loss_surv = deephit_single_loss(logits, time_t, event_t, bin_edges)
        else:  # 'mtlr'
            if bin_edges is None or logits is None:
                raise ValueError('bin_edges must be provided and logits must be available for MTLR loss')
            loss_surv = discrete_hazard_loss_mtlr(logits, time_t, event_t, bin_edges)
        # Guard against NaN/Inf due to extreme scores
        if not torch.isfinite(loss_surv):
            loss_surv = risk.sum() * 0.0

        # Pretrain losses
        if loss_weights.get('nce', 1.0) != 0.0:
            loss_nce = info_nce_loss(h_b, h_s)
        else:
            loss_nce = torch.zeros((), device=device, dtype=risk.dtype)
        if loss_weights.get('match', 0.5) != 0.0:
            loss_match = hard_negative_matching_loss(h_b, h_s)
        else:
            loss_match = torch.zeros((), device=device, dtype=risk.dtype)
        if loss_weights.get('mask', 0.5) != 0.0:
            loss_mask, _ = model.masked_gene_reconstruction_loss(x_sc, h_b)
        else:
            loss_mask = torch.zeros((), device=device, dtype=risk.dtype)

        # Apply weights to each component for both optimization and logging
        w_cox = loss_surv * loss_weights.get('cox', 1.0)
        w_nce = loss_nce * loss_weights.get('nce', 1.0)
        w_match = loss_match * loss_weights.get('match', 0.5)
        w_mask = loss_mask * loss_weights.get('mask', 0.5)

        loss = w_cox + w_nce + w_match + w_mask
        if not torch.isfinite(loss):
            # Skip this step gracefully
            optimizer.zero_grad(set_to_none=True)
            continue
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total['loss'] += float(loss.item())
        # Log weighted component values so logs reflect current weights
        total['cox'] += float(w_cox.item())
        total['nce'] += float(w_nce.item())
        total['match'] += float(w_match.item())
        total['mask'] += float(w_mask.item())
        count += 1

    for k in total:
        total[k] /= max(1, count)
    return total


@torch.no_grad()
def evaluate_cindex(model: BulkSCFusionSurvival, loader: DataLoader, device: torch.device) -> float:
    # Simple Harrell's C-index on predictions
    model.eval()
    risks: List[float] = []
    durations: List[float] = []
    events: List[int] = []
    for batch in loader:
        x_bulk = batch['x_bulk'].to(device)
        x_sc = batch['x_sc'].to(device)
        mask = batch['mask'].to(device)
        time_t = batch['duration'].cpu().numpy().tolist()
        event_t = batch['event'].cpu().numpy().astype(int).tolist()
        risk, _, _, _ = model(x_bulk, x_sc, mask)
        risks.extend(risk.detach().cpu().numpy().tolist())
        durations.extend(time_t)
        events.extend(event_t)
    # Compute C-index
    risks_np = np.array(risks, dtype=np.float64)
    times_np = np.array(durations, dtype=np.float64)
    events_np = np.array(events, dtype=np.int32)
    # Pairwise comparisons
    num = 0
    den = 0
    order = np.argsort(times_np)
    times_sorted = times_np[order]
    events_sorted = events_np[order]
    risks_sorted = risks_np[order]
    for i in range(len(times_sorted)):
        if events_sorted[i] == 0:
            continue
        for j in range(i + 1, len(times_sorted)):
            if times_sorted[j] > times_sorted[i]:
                den += 1
                if risks_sorted[i] > risks_sorted[j]:
                    num += 1
                elif risks_sorted[i] == risks_sorted[j]:
                    num += 0.5
    return float(num / den) if den > 0 else float('nan')


def plot_losses_from_history(history: List[Dict[str, float]], save_path: str) -> None:
    """Plot total loss and each component over epochs and save to PNG."""
    if not history:
        return
    epochs = [int(h.get('epoch', i + 1)) for i, h in enumerate(history)]
    total_loss = [float(h.get('loss', float('nan'))) for h in history]
    cox_loss = [float(h.get('cox', float('nan'))) for h in history]
    nce_loss = [float(h.get('nce', float('nan'))) for h in history]
    match_loss = [float(h.get('match', float('nan'))) for h in history]
    mask_loss = [float(h.get('mask', float('nan'))) for h in history]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, total_loss, label='total', linewidth=2.0)
    plt.plot(epochs, cox_loss, label='cox')
    plt.plot(epochs, nce_loss, label='nce')
    plt.plot(epochs, match_loss, label='match')
    plt.plot(epochs, mask_loss, label='mask')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss and Components')
    plt.legend()
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=150)
    finally:
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sc_npz_root', type=str, default='/data/zhaoyh/scDiffusion-main/output/cell_embs_2048/')
    parser.add_argument('--gene_list_csv', type=str, default='/data/zhaoyh/scDiffusion-main/VAE/28952genes.csv')
    parser.add_argument('--deg_csv', type=str, default='/data/zhaoyh/SHAP/coad_degs_deseq2_new.csv')
    parser.add_argument('--bulk_type', type=str, default='bulk')
    parser.add_argument('--results_dir', type=str, default='/data/zhaoyh/scDiffusion-main/bulk_sc_fusion')
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
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
    # VAE and survival label CLI parameters
    parser.add_argument('--vae_ckpt_path', type=str, default='/data/zhaoyh/scDiffusion-main/VAE/output/htan_VAE_CLTS_without_cp10k/model_seed=0_step=150000.pt')
    parser.add_argument('--vae_num_genes', type=int, default=28952)
    parser.add_argument('--surv_label_dir', type=str, default='/data/zhaoyh/SHAP/data/surv_label')
    parser.add_argument('--bulk_dir', type=str, default='/data/zhaoyh/SHAP/data/bulk')
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1) Load per-sample sc npz arrays and merge
    folder_path = args.sc_npz_root
    sample_folders = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d)) and d.startswith('cells_2048_TCGA-')]
    samples_data: Dict[str, Dict] = {}
    for sample_folder in sample_folders:
        sample_path = os.path.join(folder_path, sample_folder)
        npz_files = [os.path.join(sample_path, f) for f in os.listdir(sample_path) if f.endswith('.npz')]
        cell_gen_list = []
        type_ids = []
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
            samples_data[sample_folder] = { 'X': X_concat, 'type_ids': t_concat, 'type_names': type_names }

    # 2) Load gene names and DEG list (full gene list for VAE decoding)
    gene_order = pd.read_csv(args.gene_list_csv)
    gene_names_full = gene_order.iloc[:, 1].astype(str).tolist()
    deg_df = pd.read_csv(args.deg_csv, index_col=0)
    deg_genes = deg_df.index.astype(str).tolist()
    common_genes = [g for g in deg_genes if g in gene_names_full]
    if len(common_genes) == 0:
        raise ValueError("No overlapping DEG genes found in the model's gene list.")

    # 3) Decode cell embeddings to gene expression using VAE, then select DEG intersection
    decoded_samples_filtered: Dict[str, Dict] = {}
    vae = load_vae(device, num_genes=args.vae_num_genes, ckpt_path=args.vae_ckpt_path)
    for sample_name, sd in tqdm(samples_data.items(), total=len(samples_data), desc='Decoding cells'):
        X = sd['X']  # shape [num_cells, 128] (embeddings)
        type_ids = sd.get('type_ids', None)
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
            decoded = vae(X_tensor, return_decoded=True).detach().cpu().numpy()  # [num_cells, num_genes]
        df = pd.DataFrame(decoded, columns=gene_names_full)
        df_f = df[common_genes]
        decoded_samples_filtered[sample_name] = {
            'X_df': df_f,
            'type_ids': (type_ids.astype(np.int64) if type_ids is not None else None),
            'type_names': sd.get('type_names', []),
        }

    # 4) Load bulk expression tables from directory and select genes using DEG
    bulk_dir = args.bulk_dir
    bulk_files = [os.path.join(bulk_dir, f) for f in os.listdir(bulk_dir) if f.endswith('.csv')]
    bulk_frames: List[pd.DataFrame] = []
    for fp in bulk_files:
        try:
            df = pd.read_csv(fp, index_col=0)
            bulk_frames.append(df)
        except Exception:
            continue
    if len(bulk_frames) == 0:
        raise ValueError(f"No bulk CSVs found under {bulk_dir}")
    bulk_all = pd.concat(bulk_frames, axis=0, sort=False)
    # Drop duplicate samples, keep first occurrence
    bulk_all = bulk_all[~bulk_all.index.duplicated(keep='first')]
    # Determine final gene columns: DEG ∩ VAE gene list ∩ available bulk columns
    final_gene_cols = [g for g in common_genes if g in bulk_all.columns]
    if len(final_gene_cols) == 0:
        raise ValueError("No overlapping bulk genes found between DEG list and bulk CSV columns.")
    # Narrow bulk to selected genes
    bulk_all = bulk_all[final_gene_cols].astype(float)

    # 5) Load survival labels for the fold (labels only)
    train_surv, val_surv = load_survival_fold(args.fold, args.surv_label_dir)

    # If SC decoded used a superset, narrow SC dataframes to final_gene_cols
    for k in decoded_samples_filtered.keys():
        decoded_samples_filtered[k]['X_df'] = decoded_samples_filtered[k]['X_df'][final_gene_cols]

    # Align SC sample ids to bulk indices via TCGA barcode containment
    def map_ids(ids_list: List[str]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for pid in ids_list:
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

    def collect_targets(df: pd.DataFrame) -> Tuple[List[str], Dict[str, float], Dict[str, int]]:
        ids: List[str] = []
        durs: Dict[str, float] = {}
        evs: Dict[str, int] = {}
        for _, row in df.iterrows():
            barcode = str(row['tcga_barcode'])
            # Match by containment in sample key (sample keys contain TCGA-...)
            matched_keys = [k for k in decoded_samples_filtered.keys() if barcode in k]
            if len(matched_keys) == 0:
                continue
            pid = matched_keys[0]
            ids.append(pid)
            durs[pid] = float(row['duration'])
            evs[pid] = int(row['event'])
        return ids, durs, evs

    train_ids, train_durs, train_evs = collect_targets(train_surv)
    val_ids, val_durs, val_evs = collect_targets(val_surv)

    # Filter to patients available in sc
    train_ids = [pid for pid in train_ids if pid in decoded_samples_filtered]
    val_ids = [pid for pid in val_ids if pid in decoded_samples_filtered]

    # Build mapping from SC ids to bulk CSV indices using unified bulk_all
    tr_map = map_ids(train_ids)
    vl_map = map_ids(val_ids)
    # Keep only ids that can be mapped to rows in bulk_all
    train_ids = [pid for pid in train_ids if pid in tr_map and tr_map[pid] in bulk_all.index]
    val_ids = [pid for pid in val_ids if pid in vl_map and vl_map[pid] in bulk_all.index]
    # Reindex bulk matrices to SC ids from bulk_all
    bulk_train = bulk_all.loc[[tr_map[pid] for pid in train_ids]]
    bulk_train.index = train_ids
    bulk_val = bulk_all.loc[[vl_map[pid] for pid in val_ids]]
    bulk_val.index = val_ids

    # Build datasets/loaders using full train set (no early-stop split)
    bulk_train_sub = bulk_train.loc[train_ids]
    train_durs_sub = {pid: train_durs[pid] for pid in train_ids}
    train_evs_sub = {pid: train_evs[pid] for pid in train_ids}
    train_ds = PatientBatchDataset(train_ids, decoded_samples_filtered, bulk_train_sub, train_durs_sub, train_evs_sub, max_cells=args.max_cells, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=pad_collate_fn, drop_last=False)

    val_ds = PatientBatchDataset(val_ids, decoded_samples_filtered, bulk_val, val_durs, val_evs, max_cells=args.max_cells, seed=args.seed)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=pad_collate_fn, drop_last=False)

    # 6) Build model
    num_genes = len(final_gene_cols)
    model = BulkSCFusionSurvival(
        num_genes=num_genes,
        embed_dim=args.embed_dim,
        cell_encoder_hidden=[512, 256],
        num_attention_heads=args.num_heads,
        num_attention_layers=args.num_layers,
        dropout=args.dropout,
        bulk_encoder_type=args.bulk_encoder,
        cell_encoder_type=args.cell_encoder,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Cosine schedule with 5% warmup
    total_steps = max(1, args.epochs * max(1, math.ceil(len(train_loader))))
    warmup_steps = max(1, int(0.05 * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss weights
    weights = { 'cox': 1.0, 'nce': args.alpha_nce, 'match': args.beta_match, 'mask': args.gamma_mask }
    gene_idx_tensor = None  # could be a subset selection tensor if desired

    best_val_cindex = -1.0
    last_val_cindex = float('nan')
    best_path = os.path.join(args.results_dir, f'bulk_sc_fuse_fold{args.fold}.pt')
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, weights, gene_idx_tensor, scheduler=scheduler)

        # Evaluate on validation set only
        val_c = evaluate_cindex(model, val_loader, device)
        last_val_cindex = float(val_c) if np.isfinite(val_c) else last_val_cindex

        log_row = {
            'epoch': epoch,
            **train_metrics,
            'val_c_index': float(val_c) if np.isfinite(val_c) else float('nan'),
            'lr': float(optimizer.param_groups[0]['lr'])
        }
        history.append(log_row)
        print(json.dumps(log_row))
        step += len(train_loader)

        # Track best validation C-index and save best checkpoint
        if np.isfinite(val_c) and float(val_c) > best_val_cindex:
            best_val_cindex = float(val_c)
            torch.save({ 'model': model.state_dict(), 'epoch': epoch, 'val_c_index': best_val_cindex, 'args': vars(args) }, best_path)

    # Save training history
    with open(os.path.join(args.results_dir, f'history_fold{args.fold}.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Best val C-index: {best_val_cindex:.4f}")
    if np.isfinite(last_val_cindex):
        print(f"Last epoch val C-index: {last_val_cindex:.4f}")

    # Plot and save losses
    try:
        plot_path = os.path.join(args.results_dir, f'losses_fold{args.fold}.png')
        plot_losses_from_history(history, plot_path)
    except Exception:
        pass


if __name__ == '__main__':
    main()
