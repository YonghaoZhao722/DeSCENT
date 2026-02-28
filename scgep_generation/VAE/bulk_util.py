import pandas as pd
import scanpy as sc
import numpy as np
from scipy.sparse import issparse
import anndata
from tqdm import tqdm
from numpy.random import choice

def convert_to_tpm(raw_counts, feature_length):
    """
    Convert raw counts to TPM using gene lengths.
    """
    gene_lengths = feature_length.reindex(raw_counts.columns)
    gene_lengths[gene_lengths <= 0] = gene_lengths[gene_lengths > 0].min()  # Avoid zero lengths
    
    rpk = raw_counts.div(gene_lengths, axis=1) * 1000
    scale_factor = rpk.sum(axis=1) / 1e6
    scale_factor = scale_factor.replace(0, scale_factor[scale_factor > 0].min())
    return rpk.div(scale_factor, axis=0).fillna(0)

def generate_simulated_data(sc_data, feature_length, outname=None, d_prior=None, dirichlet=False, 
                            n=500, samplenum=5000, random_state=None, sparse=False, sparse_prob=0.2, rare=False, rare_percentage=0.2):
    np.random.seed(42)
    print('Generating pseudo-bulk raw counts')
    celltype_groups = sc_data.obs.groupby("Cell_type").groups
    sc_data_matrix = sc_data.X
    if issparse(sc_data_matrix):
        sc_data_matrix = sc_data_matrix.toarray()
    
    sc_data_matrix = pd.DataFrame(sc_data_matrix, 
                                  index=sc_data.obs_names, 
                                  columns=sc_data.var_names)
    
    num_celltype = len(celltype_groups)
    if dirichlet:
        print('using dirichlet')
        prop = np.random.dirichlet(d_prior, samplenum)
    else:
        print('sampling manually')
        prop = (d_prior / d_prior.sum()).reshape(1, -1).repeat(samplenum, axis=0)
    prop = prop / np.sum(prop, axis=1).reshape(-1, 1)
        
    # sparse cell fractions
    if sparse:
        print("You set sparse as True, some cell's fraction will be zero, the probability is", sparse_prob)
        for i in range(int(prop.shape[0] * sparse_prob)):
            indices = np.random.choice(np.arange(prop.shape[1]), replace=False, size=int(prop.shape[1] * sparse_prob))
            prop[i, indices] = 0
        prop = prop / np.sum(prop, axis=1).reshape(-1, 1)
    
    # rare cell fractions
    if rare:
        print(
            'You will set some cell type fractions as very small (<3%), '
            'these cell types are randomly chosen based on the set percentage.')
        np.random.seed(0)
        indices = np.random.choice(np.arange(prop.shape[1]), replace=False, size=int(prop.shape[1] * rare_percentage))
        prop = prop / np.sum(prop, axis=1).reshape(-1, 1)
    
        for i in range(int(0.5 * prop.shape[0]) + int(int(rare_percentage * 0.5 * prop.shape[0]))):
            prop[i, indices] = np.random.uniform(0, 0.03, len(indices))
            buf = prop[i, indices].copy()
            prop[i, indices] = 0
            prop[i] = (1 - np.sum(buf)) * prop[i] / np.sum(prop[i])
            prop[i, indices] = buf
    
    cell_num = np.floor(n * prop)
    prop = cell_num / np.sum(cell_num, axis=1).reshape(-1, 1)
    print("Mean of adjusted proportions after sampling:", prop.mean(axis=0))
    sample = np.zeros((samplenum, sc_data_matrix.shape[1]))
    for i, sample_prop in tqdm(enumerate(cell_num)):
        for j, celltype in enumerate(celltype_groups.keys()):
            select_index = choice(celltype_groups[celltype], size=int(sample_prop[j]), replace=True)
            sample[i] += sc_data_matrix.loc[select_index].sum(axis=0)
    
    sample_df = pd.DataFrame(sample, columns=sc_data_matrix.columns)
    print('Converting to TPM')
    sample_tpm = convert_to_tpm(sample_df, feature_length)
    
    simudata = anndata.AnnData(X=sample_tpm, obs=pd.DataFrame(prop, columns=celltype_groups.keys()), var=pd.DataFrame(index=sc_data_matrix.columns))

    simudata_df = pd.DataFrame(simudata.X, index=simudata.obs_names, columns=simudata.var_names)
    simudata_df.to_csv(f"{outname}.csv", index=True)
    
    if outname:
        simudata.write_h5ad(outname + '.h5ad')
    return simudata

def read_h5ad_data(file_path):
    adata = sc.read_h5ad(file_path)

    common_genes = pd.read_csv('/data/zhaoyh/scDiffusion-main/VAE/28952genes.csv')
    common_genes = common_genes[common_genes.columns[1]].tolist()
    adata._inplace_subset_var(adata.var_names.isin(common_genes))
    adata = adata[:, common_genes].copy()
    
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=10)
    # sc.pp.filter_genes(adata, min_cells=3)
    feature_length = adata.var["feature_length"] if "feature_length" in adata.var.columns else None
    if feature_length is None:
        raise ValueError("feature_length is required for TPM conversion")
    
    return adata, feature_length

def generate_pseudo_bulk(samples=10, cell_per_sample=5000, adata='/data/zhaoyh/htan_clts_train.h5ad', outname=None, dirichlet=False):
    adata, feature_length = read_h5ad_data(adata)
    print(adata.X.shape)
    # cell_types = list(adata.obs["Cell_Type"].unique())
    cell_types = sorted(adata.obs["Cell_type"].unique())

    d_prior = adata.obs["Cell_type"].value_counts(normalize=True).reindex(cell_types, fill_value=0).values
    print(d_prior)
    return generate_simulated_data(adata, feature_length, outname=outname, d_prior=d_prior, n=cell_per_sample, dirichlet=dirichlet, samplenum=samples)

if __name__ == "__main__":
    pseudo_bulk = generate_pseudo_bulk(samples=1, cell_per_sample=5000, dirichlet=True, outname='pseudo_bulk_100_dirichlet')
