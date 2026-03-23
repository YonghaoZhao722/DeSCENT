import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

import scanpy as sc
import torch

from VAE.VAE_model import VAE


def _resolve_celltype_column(adata):
    for column in ("celltype", "Cell_type", "cell_type", "CellType", "Cell Type"):
        if column in adata.obs.columns:
            if column != "celltype":
                adata.obs = adata.obs.rename(columns={column: "celltype"})
            return "celltype"
    raise ValueError(
        "Single-cell h5ad must contain a cell-type column named one of: "
        "celltype, Cell_type, cell_type, CellType, Cell Type"
    )


def load_vae(vae_path, num_gene, hidden_dim):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    autoencoder = VAE(
        num_genes=num_gene,
        device=device,
        seed=0,
        loss_ae="mse",
        hidden_dim=hidden_dim,
        decoder_activation="ReLU",
    )
    autoencoder.load_state_dict(torch.load(vae_path, map_location=device))
    autoencoder.eval()
    return autoencoder


def load_data(
    *,
    data_dir,
    batch_size,
    vae_path=None,
    deterministic=False,
    train_vae=False,
    hidden_dim=128,
):
    """
    Create a generator over `(cells, kwargs)` pairs from a single-cell h5ad file.

    The input h5ad is expected to contain raw counts. We follow the original
    scDiffusion preprocessing here: `log1p` before VAE training, and VAE latent
    encoding before diffusion/classifier training.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    adata = sc.read_h5ad(data_dir)
    _resolve_celltype_column(adata)
    adata.var_names_make_unique()
    sc.pp.log1p(adata)

    classes = adata.obs["celltype"].astype(str).values
    label_encoder = LabelEncoder()
    encoded_classes = label_encoder.fit_transform(classes)

    if hasattr(adata.X, "toarray"):
        cell_data = adata.X.toarray()
    else:
        cell_data = np.asarray(adata.X)

    if not train_vae:
        if not vae_path:
            raise ValueError("vae_path is required when train_vae=False")
        autoencoder = load_vae(vae_path, num_gene=cell_data.shape[1], hidden_dim=hidden_dim)
        device = autoencoder.device if hasattr(autoencoder, "device") else ("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            tensor = torch.tensor(cell_data, dtype=torch.float32, device=device)
            cell_data = autoencoder(tensor, return_latent=True).detach().cpu().numpy()

    dataset = CellDataset(cell_data, encoded_classes)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=1,
        drop_last=True,
    )
    while True:
        yield from loader


class CellDataset(Dataset):
    def __init__(self, cell_data, class_name):
        super().__init__()
        self.data = cell_data
        self.class_name = class_name

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        arr = self.data[idx].astype(np.float32, copy=False)
        out_dict = {"y": np.array(self.class_name[idx], dtype=np.int64)}
        return arr, out_dict
