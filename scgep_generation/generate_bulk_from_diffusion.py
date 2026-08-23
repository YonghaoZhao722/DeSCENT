#!/usr/bin/env python
"""
This script generates synthetic single-cell data for specified cell type proportions
using a diffusion model, then converts the generated cells into TPM-normalized 
pseudo-bulk data for different total cell counts.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import scanpy as sc
import anndata
from tqdm import tqdm
from pathlib import Path
import subprocess
import sys
# Set environment variables for distributed training to avoid warnings
# Get GPU ID from environment variable, default to 0
gpu_id = os.environ.get('CUDA_DEVICE_ID', '0')
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'

# Import functions from existing modules
from VAE.bulk_util import convert_to_tpm
from VAE.VAE_model import VAE

def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic bulk RNA-seq data from diffusion-generated single cells")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the diffusion model checkpoint')
    parser.add_argument('--classifier_path', type=str, required=True,
                        help='Path to the classifier model checkpoint')
    parser.add_argument('--vae_path', type=str, required=True,
                        help='Path to the VAE model checkpoint')
    parser.add_argument('--cell_ratios_file', type=str, required=True,
                        help='Path to the CSV file containing cell type proportions for all samples')
    # parser.add_argument('--feature_length_path', type=str, required=True,
    #                     help='Path to the feature length file for TPM conversion')
    parser.add_argument('--num_class', type=int, default=9,
                        help='Number of cell types in the classifier')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory for generated data')
    parser.add_argument('--cell_counts', type=str, default='10000,20000,50000',
                        help='Comma-separated list of total cell counts to generate')
    parser.add_argument('--num_genes', type=int, default=28952,
                        help='Number of genes in the gene expression data')
    parser.add_argument('--gene_order_file', type=str, default='/data/zhaoyh/scDiffusion-main/VAE/28952genes.csv',
                        help='Path to the gene order file')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress nested per-cell-type progress output while keeping top-level sample progress')
    return parser.parse_args()

def load_VAE(vae_path, num_genes):
    """Load the VAE model"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    autoencoder = VAE(
        num_genes=num_genes,
        device=device,
        seed=0,
        hidden_dim=128,
        decoder_activation='ReLU',
    )
    autoencoder.load_state_dict(torch.load(vae_path, map_location=device, weights_only=True))
    return autoencoder

def format_cell_ratios(ratios, total_cells):
    """Format cell ratios for command line argument, skipping cell types with too few cells"""
    filtered_ratios = {}
    for cell, ratio in ratios.items():
        if float(ratio) > 0:
            num_cells = int(float(ratio) * total_cells)
            if num_cells >= 2:  # Only include cell types that will generate at least 2 cells
                filtered_ratios[cell] = ratio
    return ",".join([f"{cell}:{ratio}" for cell, ratio in filtered_ratios.items()])

def generate_single_cells(args, cell_count, cell_ratios, sample_id):
    """Generate single cells using the diffusion model"""
    # Create output directory with count and sample_id
    output_dir = Path(f"{args.out_dir}/cells_{cell_count}_{sample_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command to run the classifier_sample.py script
    cmd = [
        sys.executable, str(Path(__file__).parent / "classifier_sample.py"),
        "--model_path", args.model_path,
        "--classifier_path", args.classifier_path,
        "--sample_dir", str(output_dir),
        "--num_class", str(args.num_class),
        "--total_cells", str(cell_count),
        "--sample_id", "",  # Empty sample_id to avoid duplication
        "--cell_ratios", cell_ratios,
        "--cell_ratios_file", args.cell_ratios_file  # Pass the cell ratios file
    ]
    if args.quiet:
        cmd.append("--quiet")
    
    # Pass environment variables to subprocess
    env = os.environ.copy()
    subprocess.run(cmd, check=True, env=env)
    
    return output_dir

def load_gene_names(args):
    """Load gene names from the 28952genes.csv file"""
    gene_order = pd.read_csv(args.gene_order_file)
    return gene_order.iloc[:, 1].tolist()  # Extract gene IDs in the specified order

def load_cell_type_mapping(cell_ratios_file):
    """Load cell type mapping from the cell ratios file"""
    # Read the file to get column names (excluding the first column which is sample/cell_type)
    if cell_ratios_file.endswith('.tsv'):
        df = pd.read_csv(cell_ratios_file, sep='\t', nrows=0)  # Only read header
    else:
        df = pd.read_csv(cell_ratios_file, nrows=0)  # Only read header
    
    # Get cell type names from columns (excluding the first column)
    cell_types = df.columns[1:].tolist()
    
    # Create mapping from cell type name to index
    cell_type_to_index = {cell_type: idx for idx, cell_type in enumerate(cell_types)}
    
    return cell_type_to_index

def collect_generated_cells(output_dir, vae, cell_type_to_index, sample_id, gene_lengths):
    """Collect generated cells from output directory and process through VAE"""
    all_cells = []
    total_cells = 0
    
    # Process cells from each cell type subdirectory
    for cell_type_name, cell_type_idx in cell_type_to_index.items():
        # The NPZ files are in the format: cells_{count}{sample_id}/{cell_type_idx}.npz
        npz_path = output_dir / f"{cell_type_idx}.npz"
        
        if npz_path.exists():
            try:
                data = np.load(npz_path, allow_pickle=True)
                if 'cell_gen' in data:
                    cells = data['cell_gen']
                    # Skip if we have too few cells
                    if len(cells) < 2:
                        continue
                        
                    # Decode through VAE if in latent space
                    decoded_cells = vae(torch.tensor(cells).to(vae.device), return_decoded=True).detach().cpu().numpy()
                    
                    # Transform from log1p normalized space to raw counts
                    # First, undo log1p transformation
                    cells_raw = np.expm1(decoded_cells)
                    
                    # Collect all individual cells for variance calculation
                    all_cells.append(cells_raw)
                    total_cells += len(cells_raw)
                    
            except Exception as e:
                print(f"Error loading cells for {cell_type_name}: {e}")
    
    if not all_cells:
        raise ValueError("No cells were loaded from the output directories")
    
    # Combine all cells into one matrix
    all_cells_matrix = np.vstack(all_cells)
    
    # # 除以基因长度
    # all_cells_matrix = all_cells_matrix / gene_lengths.values
    
    # # Calculate bulk counts (sum across cells)
    # bulk_counts = all_cells_matrix.sum(axis=0)
    
    # # Calculate variance across cells for each gene
    # bulk_variance = np.var(all_cells_matrix, axis=0)
    
    # # Create DataFrames with gene names
    # gene_names = load_gene_names(args)
    # bulk_counts_df = pd.DataFrame(bulk_counts.reshape(1, -1), columns=gene_names)
    # bulk_variance_df = pd.DataFrame(bulk_variance.reshape(1, -1), columns=gene_names)
    
    return None, None, total_cells

def load_feature_lengths(file_path):
    """Load gene lengths from the file, using Transcript end (bp) as the length"""
    df = pd.read_csv(file_path)
    # Use Transcript end (bp) as the length
    feature_length = pd.Series(df['Transcript end (bp)'].values, index=df['Gene name'])
    return feature_length

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load cell type proportions from CSV
    cell_proportions_df = pd.read_csv(args.cell_ratios_file)
    cell_proportions_df.set_index('sample/cell_type', inplace=True)
    
    # 加载基因长度信息
    # gene_lengths_df = pd.read_csv('/data/zhaoyh/scDiffusion-main/gene_length_processed.txt')
    # gene_names = load_gene_names(args)
    # gene_lengths = gene_lengths_df.set_index('Gene name').loc[gene_names]['feature_length']
    
    # Load VAE
    vae = load_VAE(args.vae_path, args.num_genes)
    
    # Load cell type mapping dynamically from file
    cell_type_to_index = load_cell_type_mapping(args.cell_ratios_file)
    
    # Generate cells and bulk data for different cell counts
    cell_counts = [int(count) for count in args.cell_counts.split(',')]
    
    # Process each cell count separately
    for cell_count in cell_counts:
        print(f"\nProcessing {cell_count} total cells...")
        all_bulk_counts = []
        all_bulk_variance = []
        sample_ids = []
        
        # Process each sample
        for sample_id, proportions in tqdm(
            cell_proportions_df.iterrows(),
            total=len(cell_proportions_df),
            desc=f"samples @ {cell_count} cells",
        ):
            try:
                # Format cell ratios for this sample, skipping cell types with too few cells
                cell_ratios = format_cell_ratios(proportions.to_dict(), cell_count)
                
                # Skip if no cell types have enough cells
                if not cell_ratios:
                    continue
                
                # Generate single cells
                output_dir = generate_single_cells(args, cell_count, cell_ratios, sample_id)
                
                # Collect generated cells and calculate bulk counts and variance
                bulk_counts, bulk_variance, total_cells = collect_generated_cells(output_dir, vae, cell_type_to_index, sample_id, None)
                
                # # Store results
                # all_bulk_counts.append(bulk_counts)
                # all_bulk_variance.append(bulk_variance)
                # sample_ids.append(sample_id)
                
                
            except Exception as e:
                print(f"Error processing sample {sample_id}: {e}")
                continue
        
        # # Combine all bulk samples into one file
        # if all_bulk_counts:
        #     # 保存求和文件
        #     combined_counts = pd.concat(all_bulk_counts, axis=0)
        #     combined_counts.index = sample_ids
        #     combined_counts.to_csv(f"{args.out_dir}/bulk_counts_{cell_count}_all_samples.csv")
        #     print(f"Saved bulk counts for {cell_count} cells to {args.out_dir}/bulk_counts_{cell_count}_all_samples.csv")
        #     
        #     # 保存方差文件
        #     combined_variance = pd.concat(all_bulk_variance, axis=0)
        #     combined_variance.index = sample_ids
        #     combined_variance.to_csv(f"{args.out_dir}/bulk_variance_{cell_count}_all_samples.csv")
        #     print(f"Saved bulk variance for {cell_count} cells to {args.out_dir}/bulk_variance_{cell_count}_all_samples.csv")
    
if __name__ == "__main__":
    main() 
