import argparse
import os
import time

import numpy as np
import torch
from VAE_model import VAE
import sys
sys.path.append("..")
# from guided_diffusion.cell_datasets import load_data
# from guided_diffusion.cell_datasets_sapiens import load_data
# from guided_diffusion.cell_datasets_WOT import load_data
# from guided_diffusion.cell_datasets_muris import load_data
from guided_diffusion.cell_datasets_loader import load_data

torch.autograd.set_detect_anomaly(True)
import random

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def prepare_vae(args, state_dict=None):
    """
    Instantiates autoencoder and dataset to run an experiment.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    datasets = load_data(
        data_dir=args["data_dir"],
        batch_size=args["batch_size"],
        train_vae=True,
    )

    autoencoder = VAE(
        num_genes=args["num_genes"],
        device=device,
        seed=args["seed"],
        loss_ae=args["loss_ae"],
        hidden_dim=128,
        decoder_activation=args["decoder_activation"],
    )
    if state_dict is not None:
        print('loading pretrained model from: \n',state_dict)
        use_gpu = device == "cuda"
        autoencoder.encoder.load_state(state_dict["encoder"], use_gpu)
        autoencoder.decoder.load_state(state_dict["decoder"], use_gpu)

    return autoencoder, datasets


def train_vae(args, return_model=False):
    """
    Trains a autoencoder
    """
    if args["state_dict"] is not None:
        filenames = {}
        checkpoint_path = {
            "encoder": os.path.join(
                args["state_dict"], filenames.get("model", "encoder.ckpt")
            ),
            "decoder": os.path.join(
                args["state_dict"], filenames.get("model", "decoder.ckpt")
            ),
            "gene_order": os.path.join(
                args["state_dict"], filenames.get("gene_order", "gene_order.tsv")
            ),
        }
        autoencoder, datasets = prepare_vae(args, checkpoint_path)
    else:
        autoencoder, datasets = prepare_vae(args)
   
    args["hparams"] = autoencoder.hparams

    # Calculate dataset size and batches per epoch
    import scanpy as sc
    adata = sc.read_h5ad(args["data_dir"])
    total_samples = adata.n_obs
    batches_per_epoch = total_samples // args["batch_size"]
    print(f"Dataset info: {total_samples} samples, {batches_per_epoch} batches per epoch")

    start_time = time.time()
    step = 0
    epoch = 0
    epoch_losses = {}
    best_loss = float('inf')
    best_epoch = 0
    best_step = 0
    
    while step < args["max_steps"]:
        # Start of new epoch
        epoch += 1
        epoch_losses = {}
        epoch_start_step = step
        
        print(f"\nEpoch {epoch} starting...")
        
        # Train for one epoch
        for batch_idx in range(batches_per_epoch):
            if step >= args["max_steps"]:
                break
                
            genes, _ = next(datasets)
            minibatch_training_stats = autoencoder.train_step(genes)
            
            # Accumulate losses for this epoch
            for key, val in minibatch_training_stats.items():
                if key not in epoch_losses:
                    epoch_losses[key] = []
                epoch_losses[key].append(val)
            
            step += 1
            
            # Check time limit
            ellapsed_minutes = (time.time() - start_time) / 60
            if ellapsed_minutes > args["max_minutes"]:
                break
        
        # Print epoch statistics
        if epoch_losses:
            print(f"Epoch {epoch} completed (steps {epoch_start_step}-{step-1}):")
            for key, losses in epoch_losses.items():
                avg_loss = sum(losses) / len(losses)
                print(f"  {key}: {avg_loss:.6f}")
                
                # Track best loss (assuming we want to track the main reconstruction loss)
                if key == "loss_reconstruction" and avg_loss < best_loss:
                    best_loss = avg_loss
                    best_epoch = epoch
                    best_step = step - 1
                    print(f"  *** New best loss: {best_loss:.6f} at epoch {best_epoch}, step {best_step} ***")
        
        # Check stopping conditions
        ellapsed_minutes = (time.time() - start_time) / 60
        stop = ellapsed_minutes > args["max_minutes"] or step >= args["max_steps"]
        
        # Save checkpoint
        if ((step % args["checkpoint_freq"]) == 0 or stop):
            os.makedirs(args["save_dir"], exist_ok=True)
            torch.save(
                autoencoder.state_dict(),
                os.path.join(
                    args["save_dir"],
                    "model_seed={}_step={}.pt".format(args["seed"], step),
                ),
            )
            
            if stop:
                break
    
    # Save best model
    if best_epoch > 0:
        print(f"\nSaving best model from epoch {best_epoch} (step {best_step}) with loss {best_loss:.6f}")
        os.makedirs(args["save_dir"], exist_ok=True)
        torch.save(
            autoencoder.state_dict(),
            os.path.join(
                args["save_dir"],
                "best_model_seed={}_epoch={}_step={}_loss={:.6f}.pt".format(
                    args["seed"], best_epoch, best_step, best_loss
                ),
            ),
        )

    if return_model:
        return autoencoder, datasets


def parse_arguments():
    """
    Read arguments if this script is called from a terminal.
    """

    parser = argparse.ArgumentParser(description="Finetune Scimilarity")
    # dataset arguments
    parser.add_argument("--data_dir", type=str, default='/data1/lep/Workspace/guided-diffusion/data/tabula_muris/all.h5ad')
    parser.add_argument("--loss_ae", type=str, default="mse")
    parser.add_argument("--decoder_activation", type=str, default="ReLU")

    # AE arguments                                             
    parser.add_argument("--local_rank", type=int, default=0)  
    parser.add_argument("--split_seed", type=int, default=1234)
    parser.add_argument("--num_genes", type=int, default=18996)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hparams", type=str, default="")

    # training arguments
    parser.add_argument("--max_steps", type=int, default=200000)
    parser.add_argument("--max_minutes", type=int, default=3000)
    parser.add_argument("--checkpoint_freq", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--state_dict", type=str, default="/data/zhaoyh/scDiffusion-main/VAE/pretrained/annotation_model_v1")  # if pretrain
    # parser.add_argument("--state_dict", type=str, default=None)   # if not pretrain

    parser.add_argument("--save_dir", type=str, default='../output/ae_checkpoint/muris_AE')
    parser.add_argument("--sweep_seeds", type=int, default=200)
    return dict(vars(parser.parse_args()))


if __name__ == "__main__":
    seed_everything(1234)
    train_vae(parse_arguments())