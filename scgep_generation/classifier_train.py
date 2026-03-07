"""
Train a noised image classifier on ImageNet.
"""

import argparse
import os
import time

import blobfile as bf
import torch as th

import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.cell_datasets_loader import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    create_classifier_and_diffusion,
)
from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict
import torch
import torch.nn as nn
import numpy as np
import scanpy as sc


class EpochClassifierTrainer:
    """Epoch-based classifier trainer with loss tracking and best model saving."""
    
    def __init__(self, args, model, diffusion, data, val_data, mp_trainer, opt, schedule_sampler):
        self.args = args
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.val_data = val_data
        self.mp_trainer = mp_trainer
        self.opt = opt
        self.schedule_sampler = schedule_sampler
        
        # Epoch tracking
        self.epoch = 0
        self.step = 0
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.best_step = 0
        self.epoch_losses = {}
        self.epoch_val_losses = {}
        
        # Calculate dataset size and batches per epoch
        if args.data_dir:
            adata = sc.read_h5ad(args.data_dir)
            self.total_samples = adata.n_obs
            self.batches_per_epoch = self.total_samples // args.batch_size
            print(f"Dataset info: {self.total_samples} samples, {self.batches_per_epoch} batches per epoch")
        else:
            # Fallback values if data_dir is not provided
            self.total_samples = 100000  # Default fallback
            self.batches_per_epoch = self.total_samples // args.batch_size
            print(f"Warning: data_dir not provided, using default values: {self.total_samples} samples, {self.batches_per_epoch} batches per epoch")
    
    def forward_backward_log(self, data_loader, prefix="train"):
        """Forward backward pass with loss logging."""
        batch, extra = next(data_loader)
        labels = extra["y"].to(dist_util.dev())
        batch = batch.to(dist_util.dev())
        
        # Noisy cells
        if self.args.noised:
            t, _ = self.schedule_sampler.sample(batch.shape[0], dist_util.dev(), start_guide_time=self.args.start_guide_time)
            batch = self.diffusion.q_sample(batch, t)
        else:
            t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())

        losses_dict = {}
        
        for i, (sub_batch, sub_labels, sub_t) in enumerate(
            split_microbatches(self.args.microbatch, batch, labels, t)
        ):
            logits = self.model(sub_batch, sub_t)
            loss = F.cross_entropy(logits, sub_labels, reduction="none")

            losses = {}
            losses[f"{prefix}_loss"] = loss.detach()
            losses[f"{prefix}_acc@1"] = compute_top_k(
                logits, sub_labels, k=1, reduction="none"
            )

            log_loss_dict(self.diffusion, sub_t, losses)
            
            # Store losses for epoch tracking
            for key, values in losses.items():
                if key not in losses_dict:
                    losses_dict[key] = []
                losses_dict[key].append(values.mean().item())
            
            del losses
            loss = loss.mean()
            if loss.requires_grad:
                if i == 0:
                    self.mp_trainer.zero_grad()
                self.mp_trainer.backward(loss * len(sub_batch) / len(batch))
        
        # Return average losses across all micro-batches
        avg_losses = {}
        for key, values in losses_dict.items():
            avg_losses[key] = sum(values) / len(values)
        
        return avg_losses
    
    def train_epoch(self):
        """Train for one epoch."""
        self.epoch += 1
        self.epoch_losses = {}
        epoch_start_step = self.step
        
        print(f"\nEpoch {self.epoch} starting...")
        
        # Train for one epoch
        for batch_idx in range(self.batches_per_epoch):
            if self.step >= self.args.iterations:
                break
                
            # Forward backward pass
            current_losses = self.forward_backward_log(self.data, prefix="train")
            
            # Optimize
            self.mp_trainer.optimize(self.opt)
            
            # Accumulate losses for this epoch
            for key, val in current_losses.items():
                if key not in self.epoch_losses:
                    self.epoch_losses[key] = []
                self.epoch_losses[key].append(val)
            
            self.step += 1
            
            # Anneal learning rate
            if self.args.anneal_lr:
                set_annealed_lr(self.opt, self.args.lr, self.step / self.args.iterations)
            
            # Validation
            if (self.val_data is not None and 
                not self.step % self.args.eval_interval):
                with th.no_grad():
                    with self.model.no_sync():
                        self.model.eval()
                        val_losses = self.forward_backward_log(self.val_data, prefix="val")
                        self.model.train()
                        
                        # Store validation losses
                        for key, val in val_losses.items():
                            if key not in self.epoch_val_losses:
                                self.epoch_val_losses[key] = []
                            self.epoch_val_losses[key].append(val)
            
            # Logging
            if not self.step % self.args.log_interval:
                logger.dumpkvs()
            
            # Save checkpoint
            if (self.step and dist.get_rank() == 0 and 
                not self.step % self.args.save_interval):
                logger.log("saving model...")
                save_model(self.mp_trainer, self.opt, self.step, self.args.model_path)
        
        # Print epoch statistics
        if self.epoch_losses:
            print(f"Epoch {self.epoch} completed (steps {epoch_start_step}-{self.step-1}):")
            for key, losses in self.epoch_losses.items():
                avg_loss = sum(losses) / len(losses)
                print(f"  {key}: {avg_loss:.6f}")
                
                # Track best loss (assuming we want to track the main loss)
                if key == "train_loss" and avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    self.best_epoch = self.epoch
                    self.best_step = self.step - 1
                    print(f"  *** New best loss: {self.best_loss:.6f} at epoch {self.best_epoch}, step {self.best_step} ***")
        
        # Print validation statistics if available
        if self.epoch_val_losses:
            print(f"  Validation results:")
            for key, losses in self.epoch_val_losses.items():
                avg_loss = sum(losses) / len(losses)
                print(f"    {key}: {avg_loss:.6f}")
    
    def train(self):
        """Main training loop."""
        while self.step < self.args.iterations:
            self.train_epoch()
            
            if self.step >= self.args.iterations:
                break
        
        # Save final model
        if dist.get_rank() == 0:
            logger.log("saving final model...")
            save_model(self.mp_trainer, self.opt, self.step, self.args.model_path)
            
            # Save best model
            if self.best_epoch > 0:
                print(f"\nSaving best model from epoch {self.best_epoch} (step {self.best_step}) with loss {self.best_loss:.6f}")
                self.save_best_model()
        
        dist.barrier()
    
    def save_best_model(self):
        """Save the best model."""
        if dist.get_rank() == 0:
            model_dir = self.args.model_path
            os.makedirs(model_dir, exist_ok=True)
            
            # Save best model
            th.save(
                self.mp_trainer.master_params_to_state_dict(self.mp_trainer.master_params),
                os.path.join(model_dir, f"best_model_epoch={self.best_epoch}_step={self.best_step}_loss={self.best_loss:.6f}.pt"),
            )
            th.save(
                self.opt.state_dict(), 
                os.path.join(model_dir, f"best_opt_epoch={self.best_epoch}_step={self.best_step}_loss={self.best_loss:.6f}.pt")
            )


def main():
    args = create_argparser().parse_args()

    setup_seed(1234)

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_classifier_and_diffusion(
        **args_to_dict(args, classifier_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    if args.noised:
        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion
        )

    resume_step = 0
    if args.resume_checkpoint:
        resume_step = parse_resume_step_from_filename(args.resume_checkpoint)
        if dist.get_rank() == 0:
            logger.log(
                f"loading model from checkpoint: {args.resume_checkpoint}... at {resume_step} step"
            )
            model.load_state_dict(
                dist_util.load_state_dict(
                    args.resume_checkpoint, map_location=dist_util.dev()
                )
            )

    # Needed for creating correct EMAs and fp16 parameters.
    dist_util.sync_params(model.parameters())

    mp_trainer = MixedPrecisionTrainer(
        model=model, use_fp16=args.classifier_use_fp16, initial_lg_loss_scale=16.0
    )

    model = DDP(
        model,
        device_ids=[dist_util.dev()],
        output_device=dist_util.dev(),
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=True,
    )

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        vae_path=args.vae_path,
        hidden_dim=args.latent_dim,
        train_vae=False,
    )
    if args.val_data_dir:
        val_data = load_data(
            data_dir=args.val_data_dir,
            batch_size=args.batch_size,
            vae_path=args.vae_path,
            hidden_dim=args.latent_dim,
            train_vae=False,
        )
    else:
        val_data = None

    logger.log(f"creating optimizer...")
    opt = AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)
    if args.resume_checkpoint:
        opt_checkpoint = bf.join(
            bf.dirname(args.resume_checkpoint), f"opt{resume_step:06}.pt"
        )
        logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
        opt.load_state_dict(
            dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
        )

    logger.log("training classifier model...")

    # Create epoch-based trainer
    trainer = EpochClassifierTrainer(
        args=args,
        model=model,
        diffusion=diffusion,
        data=data,
        val_data=val_data,
        mp_trainer=mp_trainer,
        opt=opt,
        schedule_sampler=schedule_sampler if args.noised else None
    )
    
    # Set resume step if needed
    if resume_step > 0:
        trainer.step = resume_step
        print(f"Resuming training from step {resume_step}")
    
    # Start training
    trainer.train()


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model(mp_trainer, opt, step, model_path):
    if dist.get_rank() == 0:
        model_dir = model_path
        os.makedirs(model_dir,exist_ok=True)
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            os.path.join(model_dir, f"model{step:06d}.pt"),
        )
        th.save(opt.state_dict(), os.path.join(model_dir, f"opt{step:06d}.pt"))


def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)


def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)


def create_argparser():
    defaults = dict(
        data_dir="",
        val_data_dir="",
        noised=True,
        iterations=500000,
        lr=3e-4,
        weight_decay=1e-4,
        anneal_lr=False,
        batch_size=128,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=1000,
        eval_interval=1000,
        save_interval=100000,
        vae_path='',
        latent_dim=128,
        model_path='',
        start_guide_time=500,
        num_class=12,
    )
    num_class = defaults['num_class']
    defaults.update(classifier_and_diffusion_defaults())
    defaults['num_class']= num_class
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    main()
