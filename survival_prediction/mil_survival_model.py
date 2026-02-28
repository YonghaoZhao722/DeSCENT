import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple


class CellEncoder(nn.Module):
    """
    Cell Encoder: Encodes each cell's gene expression into a d-dimensional embedding
    """
    def __init__(
        self, 
        input_dim: int = 2000, 
        hidden_dims: list = [512, 256], 
        output_dim: int = 128,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        use_batch_norm: bool = False,
        l1_reg: float = 0.0
    ):
        super(CellEncoder, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build MLP layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm and not use_batch_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            elif use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        if use_layer_norm and not use_batch_norm:
            layers.append(nn.LayerNorm(output_dim))
        elif use_batch_norm:
            layers.append(nn.BatchNorm1d(output_dim))
        
        self.encoder = nn.Sequential(*layers)
        self.l1_reg = l1_reg
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_cells, num_genes) or (num_cells, num_genes)
        Returns:
            embeddings: (batch_size, num_cells, output_dim) or (num_cells, output_dim)
        """
        original_shape = x.shape
        if len(original_shape) == 3:
            batch_size, num_cells, num_genes = original_shape
            x = x.view(-1, num_genes)  # (batch_size * num_cells, num_genes)
            embeddings = self.encoder(x)  # (batch_size * num_cells, output_dim)
            embeddings = embeddings.view(batch_size, num_cells, -1)  # (batch_size, num_cells, output_dim)
        else:
            embeddings = self.encoder(x)  # (num_cells, output_dim)
        
        return embeddings
    
    def get_l1_loss(self):
        """Calculate L1 regularization loss for gene selection"""
        l1_loss = 0.0
        if self.l1_reg > 0:
            for param in self.parameters():
                l1_loss += torch.sum(torch.abs(param))
        return self.l1_reg * l1_loss


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention for learning cell interactions
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, embed_dim) or (seq_len, embed_dim)
            mask: Optional mask for padded sequences
        Returns:
            output: same shape as input
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # Add batch dimension if needed
            batch_added = True
        else:
            batch_added = False
            
        B, N, C = x.shape
        
        # Generate Q, K, V with shape (B, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Build attention mask compatible with PyTorch SDPA
        attn_mask = None
        if mask is not None:
            # Expecting mask where 1 indicates valid, 0 indicates pad
            if mask.dim() == 2:  # (B, N)
                # Convert to boolean mask where True = masked
                attn_mask = (~mask.bool()).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            else:
                # Convert to boolean and broadcast over heads if needed
                m = (mask == 0)
                if m.dim() == 3:  # (B, N, N)
                    attn_mask = m.unsqueeze(1)  # (B, 1, N, N)
                else:
                    attn_mask = m  # assume already broadcastable to (B, H, N, N)

        # Use torch's fused scaled dot-product attention (Flash/Math/Memory-Efficient)
        out_ctx = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )  # (B, H, N, head_dim)

        out = out_ctx.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        
        if batch_added:
            out = out.squeeze(0)
            
        return out


class AttentionPooling(nn.Module):
    """
    Attention-based pooling to aggregate cell embeddings into patient-level representation
    """
    def __init__(self, embed_dim: int, hidden_dim: int = 128):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, num_cells, embed_dim) or (num_cells, embed_dim)
            mask: Optional mask for variable number of cells
        Returns:
            pooled: (batch_size, embed_dim) or (embed_dim,)
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            batch_added = True
        else:
            batch_added = False
            
        # Calculate attention weights
        attn_weights = self.attention(x)  # (batch_size, num_cells, 1)
        
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
            
        attn_weights = F.softmax(attn_weights, dim=1)  # (batch_size, num_cells, 1)
        
        # Weighted aggregation
        pooled = torch.sum(x * attn_weights, dim=1)  # (batch_size, embed_dim)
        
        if batch_added:
            pooled = pooled.squeeze(0)
            
        return pooled, attn_weights.squeeze(-1)


class TypeAwareHierarchicalPooling(nn.Module):
    """
    Two-level hierarchical pooling with per-type intra attention + mean residual,
    followed by inter-type attention + mean residual.

    Returns bag embedding and per-cell attention weights (attention component only).
    """
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 128,
        gamma_intra: float = 0.5,
        eta_inter: float = 0.5,
        tau_intra: float = 1.0,
        tau_inter: float = 1.0,
    ):
        super(TypeAwareHierarchicalPooling, self).__init__()
        self.embed_dim = int(embed_dim)
        self.gamma_intra = float(gamma_intra)
        self.eta_inter = float(eta_inter)
        self.tau_intra = float(tau_intra)
        self.tau_inter = float(tau_inter)

        # MLPs to produce unnormalized attention logits
        self.intra_attn_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.inter_attn_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def _softmax_with_temperature(self, logits: torch.Tensor, dim: int, tau: float) -> torch.Tensor:
        if tau <= 0:
            tau = 1.0
        return torch.softmax(logits / tau, dim=dim)

    def forward(
        self,
        embeddings: torch.Tensor,      # (B, N, C)
        type_ids: torch.Tensor,         # (B, N) with -1 for pads
        mask: Optional[torch.Tensor] = None,  # (B, N) 1 for valid
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = embeddings.shape
        device = embeddings.device
        # Output containers
        bag_embeddings: list = []
        per_cell_weights: list = []  # attention component only

        for b in range(B):
            emb_b = embeddings[b]              # (N, C)
            types_b = type_ids[b].long()       # (N,)
            if mask is not None:
                valid_b = (mask[b] > 0)
            else:
                valid_b = (types_b >= 0) | torch.ones_like(types_b, dtype=torch.bool)
            # Consider only valid cells
            valid_idx = torch.nonzero(valid_b, as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                # Degenerate bag: return zeros
                bag_embeddings.append(torch.zeros(C, device=device, dtype=embeddings.dtype))
                per_cell_weights.append(torch.zeros(N, device=device, dtype=embeddings.dtype))
                continue

            types_valid = types_b[valid_idx]
            emb_valid = emb_b[valid_idx]       # (Nv, C)

            # Unique types (>=0); if none provided, treat all as single type 0
            uniq = torch.unique(types_valid[types_valid >= 0])
            if uniq.numel() == 0:
                uniq = torch.tensor([0], device=device, dtype=torch.long)
                types_valid = torch.zeros_like(types_valid)

            # Intra-type pooling
            z_t_list = []
            w_intra_list = []  # per-cell attn within each type, placed back to valid indices later
            per_cell_weights_b = torch.zeros(N, device=device, dtype=embeddings.dtype)
            type_to_indices = {}

            for t in uniq.tolist():
                t_mask = (types_valid == t)
                idx_t_valid = valid_idx[t_mask]
                if idx_t_valid.numel() == 0:
                    continue
                x_t = emb_b[idx_t_valid]  # (K_t, C)
                logits_t = self.intra_attn_mlp(x_t).squeeze(-1)  # (K_t,)
                a_t = self._softmax_with_temperature(logits_t, dim=0, tau=self.tau_intra)  # (K_t,)
                pooled_attn_t = torch.sum(x_t * a_t.unsqueeze(-1), dim=0)  # (C,)
                pooled_mean_t = torch.mean(x_t, dim=0)  # (C,)
                z_t = self.gamma_intra * pooled_attn_t + (1.0 - self.gamma_intra) * pooled_mean_t
                z_t_list.append(z_t)
                # store attention component only for entropy/prop
                w_intra_list.append((idx_t_valid, a_t))
                type_to_indices[t] = idx_t_valid

            if len(z_t_list) == 0:
                bag_embeddings.append(torch.zeros(C, device=device, dtype=embeddings.dtype))
                per_cell_weights.append(torch.zeros(N, device=device, dtype=embeddings.dtype))
                continue

            Z = torch.stack(z_t_list, dim=0)  # (T, C)
            logits_T = self.inter_attn_mlp(Z).squeeze(-1)  # (T,)
            w_T = self._softmax_with_temperature(logits_T, dim=0, tau=self.tau_inter)  # (T,)
            pooled_attn_B = torch.sum(Z * w_T.unsqueeze(-1), dim=0)  # (C,)
            pooled_mean_B = torch.mean(Z, dim=0)  # (C,)
            z_B = self.eta_inter * pooled_attn_B + (1.0 - self.eta_inter) * pooled_mean_B
            bag_embeddings.append(z_B)

            # Compose per-cell weights (attention-only component): w_T[t] * a_t(i)
            for t_idx, (idxs, a_t) in enumerate(w_intra_list):
                per_cell_weights_b[idxs] = w_T[t_idx] * a_t
            per_cell_weights.append(per_cell_weights_b)

        bag_embeddings_tensor = torch.stack(bag_embeddings, dim=0)  # (B, C)
        per_cell_weights_tensor = torch.stack(per_cell_weights, dim=0)  # (B, N)
        return bag_embeddings_tensor, per_cell_weights_tensor

class MILSurvivalNet(nn.Module):
    """
    Multi-Instance Learning Network for Single-Cell Survival Analysis
    
    Architecture:
    1. Cell Encoder: genes → cell embeddings
    2. Set-level Self-Attention: learn cell interactions
    3. Attention Pooling: aggregate to patient representation
    4. Output: single risk score for Cox regression
    """
    def __init__(
        self,
        num_genes: int,
        cell_embed_dim: int = 128,
        cell_encoder_hidden: list = [512, 256],
        num_attention_heads: int = 8,
        num_attention_layers: int = 2,
        final_hidden_dim: int = 64,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        use_batch_norm: bool = False,
        l1_reg: float = 0.0,
        output_mode: str = 'cox',  # 'cox' | 'deephit' | 'mtlr'
        num_time_bins: int = 50,
        pooling: str = 'attention',  # 'attention' | 'cls'
        # Hierarchical pooling options
        use_hierarchical_pooling: bool = False,
        gamma_intra: float = 0.5,
        eta_inter: float = 0.5,
        attn_tau_intra: float = 1.0,
        attn_tau_inter: float = 1.0,
    ):
        super(MILSurvivalNet, self).__init__()
        
        # Cell Encoder
        self.cell_encoder = CellEncoder(
            input_dim=num_genes,
            hidden_dims=cell_encoder_hidden,
            output_dim=cell_embed_dim,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
            use_batch_norm=use_batch_norm,
            l1_reg=l1_reg
        )
        
        # Multi-layer Self-Attention
        self.attention_layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': MultiHeadSelfAttention(cell_embed_dim, num_attention_heads, dropout),
                'norm1': nn.LayerNorm(cell_embed_dim),
                'ff': nn.Sequential(
                    nn.Linear(cell_embed_dim, cell_embed_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(cell_embed_dim * 2, cell_embed_dim)
                ),
                'norm2': nn.LayerNorm(cell_embed_dim),
            })
            for _ in range(num_attention_layers)
        ])
        
        # Pooling configuration
        self.pooling = str(pooling).lower()
        if self.pooling not in ['attention', 'cls']:
            print(f"[Warning] Unknown pooling '{self.pooling}', defaulting to 'attention'.")
            self.pooling = 'attention'
        self.use_hier = bool(use_hierarchical_pooling)
        if self.pooling == 'attention':
            # Standard attention pooling as fallback
            self.attention_pool = AttentionPooling(cell_embed_dim, final_hidden_dim)
            # Type-aware hierarchical pooling (optional)
            self.hier_pool = TypeAwareHierarchicalPooling(
                embed_dim=cell_embed_dim,
                hidden_dim=final_hidden_dim,
                gamma_intra=gamma_intra,
                eta_inter=eta_inter,
                tau_intra=attn_tau_intra,
                tau_inter=attn_tau_inter,
            ) if self.use_hier else None
            self.cls_token = None
        else:
            # Learnable CLS token for CLS pooling
            self.attention_pool = None
            self.hier_pool = None
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cell_embed_dim))
            nn.init.normal_(self.cls_token, std=0.02)
        
        # Output configuration
        self.output_mode = output_mode.lower()
        self.num_time_bins = int(num_time_bins)

        # Final prediction head
        out_dim = 1 if self.output_mode == 'cox' else self.num_time_bins
        self.prediction_head = nn.Sequential(
            nn.Linear(cell_embed_dim, final_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(final_hidden_dim, out_dim)
        )
        
    def forward(self, x, mask=None, type_ids: Optional[torch.Tensor] = None, return_attention=False):
        """
        Args:
            x: (batch_size, num_cells, num_genes) or (num_cells, num_genes)
            mask: Optional mask for variable number of cells
            return_attention: Whether to return attention weights
        Returns:
            risk_score: (batch_size, 1) or (1,) - Cox risk score
            attention_weights: Optional, cell-level attention weights
        """
        # Encode cells
        cell_embeddings = self.cell_encoder(x)  # (batch_size, num_cells, embed_dim)
        
        # Optionally prepend CLS token before attention layers for CLS pooling
        added_batch_dim = False
        if len(cell_embeddings.shape) == 2:
            cell_embeddings = cell_embeddings.unsqueeze(0)
            added_batch_dim = True
        B, N, C = cell_embeddings.shape
        current_mask = None
        if mask is not None:
            current_mask = mask
            if current_mask.dim() == 2:
                # (B, N)
                pass
            elif current_mask.dim() == 1:
                current_mask = current_mask.unsqueeze(0)
            else:
                # For other shapes, best-effort: reduce to (B, N) valid mask
                current_mask = (current_mask.sum(dim=-1) != 0).to(cell_embeddings.dtype)

        if self.pooling == 'cls':
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, C)
            cell_embeddings = torch.cat([cls_tokens, cell_embeddings], dim=1)  # (B, 1+N, C)
            if current_mask is not None:
                ones = torch.ones(B, 1, device=cell_embeddings.device, dtype=current_mask.dtype)
                current_mask = torch.cat([ones, current_mask], dim=1)  # (B, 1+N)

        # Apply self-attention layers
        for layer in self.attention_layers:
            # Self-attention with residual connection
            attn_out = layer['self_attn'](cell_embeddings, current_mask)
            cell_embeddings = layer['norm1'](cell_embeddings + attn_out)
            
            # Feed-forward with residual connection
            ff_out = layer['ff'](cell_embeddings)
            cell_embeddings = layer['norm2'](cell_embeddings + ff_out)
        
        # Aggregate cells to patient representation
        if self.pooling == 'attention' and (self.use_hier and type_ids is not None):
            # Hierarchical type-aware pooling
            if type_ids.dim() == 1:
                type_ids = type_ids.unsqueeze(0)
            patient_embedding, attention_weights = self.hier_pool(cell_embeddings, type_ids, current_mask)
        elif self.pooling == 'attention':
            patient_embedding, attention_weights = self.attention_pool(cell_embeddings, current_mask)
        else:
            # CLS pooling: take the CLS token output
            patient_embedding = cell_embeddings[:, 0, :]
            orig_len = cell_embeddings.shape[1] - 1
            attention_weights = torch.zeros(B, orig_len, device=cell_embeddings.device, dtype=cell_embeddings.dtype)
        
        if added_batch_dim:
            patient_embedding = patient_embedding.squeeze(0)
            attention_weights = attention_weights.squeeze(0)
        
        # Final prediction
        logits = self.prediction_head(patient_embedding)
        
        # Add L1 regularization loss if needed
        l1_loss = self.cell_encoder.get_l1_loss()
        
        if return_attention:
            return logits, attention_weights, l1_loss
        else:
            return logits if l1_loss == 0 else (logits, l1_loss)
    
    def get_cell_embeddings(self, x, mask=None):
        """Get cell embeddings for visualization/analysis"""
        with torch.no_grad():
            cell_embeddings = self.cell_encoder(x)
            
            # Apply self-attention layers
            for layer in self.attention_layers:
                attn_out = layer['self_attn'](cell_embeddings, mask)
                cell_embeddings = layer['norm1'](cell_embeddings + attn_out)
                ff_out = layer['ff'](cell_embeddings)
                cell_embeddings = layer['norm2'](cell_embeddings + ff_out)
                
            return cell_embeddings
    
    def get_attention_weights(self, x, mask=None):
        """Get attention weights for interpretability"""
        with torch.no_grad():
            _, attention_weights, _ = self.forward(x, mask, return_attention=True)
            return attention_weights


class MILSurvivalNetPreencoded(nn.Module):
    """
    MIL network variant that assumes inputs are already cell embeddings.
    Skips the per-cell encoder and applies only set-level self-attention,
    attention pooling, and an MLP prediction head.
    """
    def __init__(
        self,
        embed_dim: int,
        num_attention_heads: int = 8,
        num_attention_layers: int = 2,
        final_hidden_dim: int = 64,
        dropout: float = 0.1,
        output_mode: str = 'cox',  # 'cox' | 'deephit' | 'mtlr'
        num_time_bins: int = 50,
        pooling: str = 'attention',  # 'attention' | 'cls'
    ):
        super(MILSurvivalNetPreencoded, self).__init__()

        self.embed_dim = int(embed_dim)

        # Multi-layer Self-Attention
        self.attention_layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': MultiHeadSelfAttention(self.embed_dim, num_attention_heads, dropout),
                'norm1': nn.LayerNorm(self.embed_dim),
                'ff': nn.Sequential(
                    nn.Linear(self.embed_dim, self.embed_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.embed_dim * 2, self.embed_dim)
                ),
                'norm2': nn.LayerNorm(self.embed_dim),
            })
            for _ in range(num_attention_layers)
        ])

        # Pooling configuration
        self.pooling = str(pooling).lower()
        if self.pooling not in ['attention', 'cls']:
            print(f"[Warning] Unknown pooling '{self.pooling}', defaulting to 'attention'.")
            self.pooling = 'attention'
        if self.pooling == 'attention':
            self.attention_pool = AttentionPooling(self.embed_dim, final_hidden_dim)
            self.cls_token = None
        else:
            self.attention_pool = None
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            nn.init.normal_(self.cls_token, std=0.02)

        # Output configuration
        self.output_mode = output_mode.lower()
        self.num_time_bins = int(num_time_bins)

        # Final prediction head
        out_dim = 1 if self.output_mode == 'cox' else self.num_time_bins
        self.prediction_head = nn.Sequential(
            nn.Linear(self.embed_dim, final_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(final_hidden_dim, out_dim)
        )

    def forward(self, x, mask=None, return_attention: bool = False):
        # x is already (batch, num_cells, embed_dim)
        cell_embeddings = x

        added_batch_dim = False
        if len(cell_embeddings.shape) == 2:
            cell_embeddings = cell_embeddings.unsqueeze(0)
            added_batch_dim = True
        B, N, C = cell_embeddings.shape
        current_mask = None
        if mask is not None:
            current_mask = mask
            if current_mask.dim() == 2:
                pass
            elif current_mask.dim() == 1:
                current_mask = current_mask.unsqueeze(0)
            else:
                current_mask = (current_mask.sum(dim=-1) != 0).to(cell_embeddings.dtype)

        if self.pooling == 'cls':
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cell_embeddings = torch.cat([cls_tokens, cell_embeddings], dim=1)
            if current_mask is not None:
                ones = torch.ones(B, 1, device=cell_embeddings.device, dtype=current_mask.dtype)
                current_mask = torch.cat([ones, current_mask], dim=1)

        for layer in self.attention_layers:
            attn_out = layer['self_attn'](cell_embeddings, current_mask)
            cell_embeddings = layer['norm1'](cell_embeddings + attn_out)
            ff_out = layer['ff'](cell_embeddings)
            cell_embeddings = layer['norm2'](cell_embeddings + ff_out)

        if self.pooling == 'attention':
            patient_embedding, attention_weights = self.attention_pool(cell_embeddings, current_mask)
        else:
            patient_embedding = cell_embeddings[:, 0, :]
            orig_len = cell_embeddings.shape[1] - 1
            attention_weights = torch.zeros(B, orig_len, device=cell_embeddings.device, dtype=cell_embeddings.dtype)

        if added_batch_dim:
            patient_embedding = patient_embedding.squeeze(0)
            attention_weights = attention_weights.squeeze(0)
        logits = self.prediction_head(patient_embedding)

        if return_attention:
            # Keep API consistent with MILSurvivalNet (no L1 term here)
            dummy_l1 = torch.zeros((), device=logits.device, dtype=logits.dtype)
            return logits, attention_weights, dummy_l1
        else:
            return logits


def create_mil_survival_model(
    num_genes: int = 2000,
    cell_embed_dim: int = 128,
    cell_encoder_hidden: list = [512, 256],
    num_attention_heads: int = 8,
    num_attention_layers: int = 2,
    final_hidden_dim: int = 64,
    dropout: float = 0.1,
    use_layer_norm: bool = True,
    use_batch_norm: bool = False,
    l1_reg: float = 0.0,
    output_mode: str = 'cox',
    num_time_bins: int = 50,
    preencoded_input: bool = False,
    pooling: str = 'attention',
    use_hierarchical_pooling: bool = False,
    gamma_intra: float = 0.5,
    eta_inter: float = 0.5,
    attn_tau_intra: float = 1.0,
    attn_tau_inter: float = 1.0,
) -> MILSurvivalNet:
    """
    Factory function to create MIL survival model with reasonable defaults
    """
    if preencoded_input:
        # In pre-encoded mode, we expect the input feature dimension to match the
        # attention embedding dimension. Use num_genes as the embed_dim.
        embed_dim = int(num_genes)
        if cell_embed_dim != embed_dim:
            # Soft warning without raising to keep compatibility
            print(f"[Warning] preencoded_input=True: overriding cell_embed_dim ({cell_embed_dim}) to match input dim ({embed_dim}).")
        return MILSurvivalNetPreencoded(
            embed_dim=embed_dim,
            num_attention_heads=num_attention_heads,
            num_attention_layers=num_attention_layers,
            final_hidden_dim=final_hidden_dim,
            dropout=dropout,
            output_mode=output_mode,
            num_time_bins=num_time_bins,
            pooling=pooling,
        )
    else:
        return MILSurvivalNet(
            num_genes=num_genes,
            cell_embed_dim=cell_embed_dim,
            cell_encoder_hidden=cell_encoder_hidden,
            num_attention_heads=num_attention_heads,
            num_attention_layers=num_attention_layers,
            final_hidden_dim=final_hidden_dim,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
            use_batch_norm=use_batch_norm,
            l1_reg=l1_reg,
            output_mode=output_mode,
            num_time_bins=num_time_bins,
            pooling=pooling,
            use_hierarchical_pooling=use_hierarchical_pooling,
            gamma_intra=gamma_intra,
            eta_inter=eta_inter,
            attn_tau_intra=attn_tau_intra,
            attn_tau_inter=attn_tau_inter,
        )


# Example usage and testing
if __name__ == "__main__":
    # Test with sample data shape
    batch_size = 2
    num_cells = 2048
    num_genes = 2000
    
    # Create model
    model = create_mil_survival_model(
        num_genes=num_genes,
        cell_embed_dim=128,
        num_attention_heads=8,
        num_attention_layers=2,
        dropout=0.1,
        l1_reg=0.001
    )
    
    # Test forward pass
    x = torch.randn(batch_size, num_cells, num_genes)
    
    print("Testing MIL Survival Model:")
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = model(x)
    if isinstance(output, tuple):
        risk_scores, l1_loss = output
        print(f"Risk scores shape: {risk_scores.shape}")
        print(f"L1 loss: {l1_loss.item():.6f}")
    else:
        risk_scores = output
        print(f"Risk scores shape: {risk_scores.shape}")
    
    # Test attention weights
    attention_weights = model.get_attention_weights(x)
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Attention weights sum (should be close to 1): {attention_weights.sum(dim=1)}")
    
    print("\nModel architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
