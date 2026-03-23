import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from tqdm.auto import tqdm

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from VAE.VAE_model import VAE
from guided_diffusion.cell_datasets_loader import load_data


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def prepare_vae(args, checkpoint_dir=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_vae=True,
    )
    autoencoder = VAE(
        num_genes=args.num_genes,
        device=device,
        seed=args.seed,
        loss_ae=args.loss_ae,
        hidden_dim=128,
        decoder_activation=args.decoder_activation,
    )
    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        encoder_ckpt = checkpoint_dir / "encoder.ckpt"
        decoder_ckpt = checkpoint_dir / "decoder.ckpt"
        if not encoder_ckpt.exists() or not decoder_ckpt.exists():
            raise FileNotFoundError(
                f"Pretrained VAE directory must contain encoder.ckpt and decoder.ckpt: {checkpoint_dir}"
            )
        use_gpu = device == "cuda"
        autoencoder.encoder.load_state(str(encoder_ckpt), use_gpu)
        autoencoder.decoder.load_state(str(decoder_ckpt), use_gpu)
    return autoencoder, datasets


def train_vae(args):
    autoencoder, datasets = prepare_vae(args, checkpoint_dir=args.state_dict or None)

    adata = sc.read_h5ad(args.data_dir)
    total_samples = adata.n_obs
    batches_per_epoch = max(total_samples // args.batch_size, 1)
    total_epochs = max(math.ceil(args.max_steps / batches_per_epoch), 1)
    print(f"Dataset info: {total_samples} samples, {batches_per_epoch} batches per epoch")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    step = 0
    epoch = 0
    best_loss = float("inf")
    best_epoch = 0
    best_step = 0
    epoch_progress = tqdm(
        total=total_epochs,
        desc="VAE epochs",
        leave=True,
        dynamic_ncols=True,
        disable=not args.quiet,
    )

    while step < args.max_steps:
        epoch += 1
        epoch_losses = {}
        epoch_start_step = step
        if not args.quiet:
            print(f"\nEpoch {epoch} starting...")

        for _ in range(batches_per_epoch):
            if step >= args.max_steps:
                break
            genes, _ = next(datasets)
            minibatch_stats = autoencoder.train_step(genes)
            for key, val in minibatch_stats.items():
                epoch_losses.setdefault(key, []).append(val)
            step += 1
            if (time.time() - start_time) / 60 > args.max_minutes:
                break

        if epoch_losses:
            avg_epoch_losses = {
                key: sum(values) / len(values) for key, values in epoch_losses.items()
            }
            avg_recon = avg_epoch_losses.get("loss_reconstruction")
            if avg_recon is not None and avg_recon < best_loss:
                best_loss = avg_recon
                best_epoch = epoch
                best_step = step - 1

            if args.quiet:
                postfix = {"epoch": f"{epoch}/{total_epochs}"}
                if avg_recon is not None:
                    postfix["loss"] = f"{avg_recon:.6f}"
                if best_epoch > 0:
                    postfix["best"] = f"{best_loss:.6f}"
                epoch_progress.set_postfix(postfix)
                epoch_progress.update(1)
            else:
                print(f"Epoch {epoch} completed (steps {epoch_start_step}-{step - 1}):")
                for key, avg_loss in avg_epoch_losses.items():
                    print(f"  {key}: {avg_loss:.6f}")
                if avg_recon is not None and best_epoch == epoch and best_step == step - 1:
                    print(
                        f"  *** New best loss: {best_loss:.6f} at epoch {best_epoch}, step {best_step} ***"
                    )

        stop = (time.time() - start_time) / 60 > args.max_minutes or step >= args.max_steps
        if step % args.checkpoint_freq == 0 or stop:
            checkpoint_path = save_dir / f"model_seed={args.seed}_step={step}.pt"
            torch.save(autoencoder.state_dict(), checkpoint_path)
            if not args.quiet:
                print(f"Saved checkpoint: {checkpoint_path}")
            if stop:
                break

    epoch_progress.close()

    if best_epoch > 0:
        best_path = save_dir / (
            f"best_model_seed={args.seed}_epoch={best_epoch}_step={best_step}_loss={best_loss:.6f}.pt"
        )
        torch.save(autoencoder.state_dict(), best_path)
        if not args.quiet:
            print(f"Saved best checkpoint: {best_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune the scDiffusion VAE")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--loss_ae", type=str, default="mse")
    parser.add_argument("--decoder_activation", type=str, default="ReLU")
    parser.add_argument("--num_genes", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=200000)
    parser.add_argument("--max_minutes", type=int, default=3000)
    parser.add_argument("--checkpoint_freq", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--state_dict",
        type=str,
        default="",
        help="Path to the downloaded scimilarity pretrained directory (annotation_model_v1).",
    )
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    seed_everything(1234)
    train_vae(args)
