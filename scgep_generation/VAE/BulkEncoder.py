import torch
import torch.nn as nn
import torch.nn.functional as F

class BulkEncoder(nn.Module):
    def __init__(self, num_genes, latent_dim=128, hidden_dim=[1024, 1024], dropout=0.5, input_dropout=0.4, residual=False):
        """
        BulkEncoder model to map bulk RNA-seq data to the latent space of VAE.
        
        Parameters:
        - num_genes: int, Number of input genes
        - latent_dim: int, Dimension of latent space (must match VAE's latent space)
        - hidden_dim: list, Dimensions of hidden layers
        - dropout: float, Dropout rate for hidden layers
        - input_dropout: float, Dropout rate for input layer
        - residual: bool, Whether to use residual connections
        """
        super(BulkEncoder, self).__init__()
        self.residual = residual
        self.network = nn.ModuleList()

        self.network.append(nn.Sequential(
            nn.Dropout(p=input_dropout),
            nn.Linear(num_genes, hidden_dim[0]),
            nn.BatchNorm1d(hidden_dim[0]),
            nn.PReLU()
        ))

        for i in range(1, len(hidden_dim)):
            self.network.append(nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim[i-1], hidden_dim[i]),
                nn.BatchNorm1d(hidden_dim[i]),
                nn.PReLU()
            ))

        self.network.append(nn.Linear(hidden_dim[-1], latent_dim))

    def forward(self, x):
        for i, layer in enumerate(self.network):
            if self.residual and (0 < i < len(self.network) - 1):
                x = layer(x) + x  # Residual connection
            else:
                x = layer(x)
        return F.normalize(x, p=2, dim=1)

    def save_state(self, filename: str):
        torch.save({"state_dict": self.state_dict()}, filename)
        print(f"BulkEncoder saved to: {filename}")

    def load_state(self, filename: str, use_gpu: bool = False):
        if not use_gpu:
            ckpt = torch.load(filename, map_location=torch.device("cpu"))
        else:
            ckpt = torch.load(filename)

        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        self.load_state_dict(state_dict, strict=False)
        print(f"BulkEncoder loaded: {filename}")


class CellEncoderSymmetric(nn.Module):
    """
    Symmetric CellEncoder that mirrors BulkEncoder architecture exactly to ensure
    structural symmetry and identical embedding normalization/scale.
    """
    def __init__(self, num_genes, latent_dim=128, hidden_dim=[1024, 1024], dropout=0.5, input_dropout=0.4, residual=False):
        super(CellEncoderSymmetric, self).__init__()
        self.residual = residual
        self.network = nn.ModuleList()

        self.network.append(nn.Sequential(
            nn.Dropout(p=input_dropout),
            nn.Linear(num_genes, hidden_dim[0]),
            nn.BatchNorm1d(hidden_dim[0]),
            nn.PReLU()
        ))

        for i in range(1, len(hidden_dim)):
            self.network.append(nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim[i-1], hidden_dim[i]),
                nn.BatchNorm1d(hidden_dim[i]),
                nn.PReLU()
            ))

        self.network.append(nn.Linear(hidden_dim[-1], latent_dim))

    def forward(self, x):
        # x can be (B, N, G) or (B, G) or (N, G); we encode last dim G → latent_dim per item
        original_shape = x.shape
        if x.dim() == 3:
            b, n, g = original_shape
            x = x.view(b * n, g)
            for i, layer in enumerate(self.network):
                if self.residual and (0 < i < len(self.network) - 1):
                    x = layer(x) + x
                else:
                    x = layer(x)
            x = F.normalize(x, p=2, dim=1)
            x = x.view(b, n, -1)
            return x
        else:
            # (B, G) or (N, G)
            for i, layer in enumerate(self.network):
                if self.residual and (0 < i < len(self.network) - 1):
                    x = layer(x) + x
                else:
                    x = layer(x)
            return F.normalize(x, p=2, dim=1)