"""
Train a diffusion model on images.
"""

import argparse
import time

from guided_diffusion import dist_util, logger
from guided_diffusion.cell_datasets_loader import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
import os
import torch

import numpy as np
import random
import scanpy as sc
import functools
from guided_diffusion import dist_util
from guided_diffusion.resample import LossAwareSampler


class EpochTrainLoop(TrainLoop):
    """Custom TrainLoop with epoch-based training support."""
    
    def __init__(self, *args, data_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch = 0
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.best_step = 0
        self.epoch_losses = {}
        self.data_dir = data_dir
        
        # Calculate dataset size and batches per epoch
        if self.data_dir:
            adata = sc.read_h5ad(self.data_dir)
            self.total_samples = adata.n_obs
            self.batches_per_epoch = self.total_samples // self.batch_size
            print(f"Dataset info: {self.total_samples} samples, {self.batches_per_epoch} batches per epoch")
        else:
            # Fallback values if data_dir is not provided
            self.total_samples = 100000  # Default fallback
            self.batches_per_epoch = self.total_samples // self.batch_size
            print(f"Warning: data_dir not provided, using default values: {self.total_samples} samples, {self.batches_per_epoch} batches per epoch")
    
    def set_data_dir(self, data_dir):
        """Set data directory for epoch calculation."""
        self.data_dir = data_dir
        if self.data_dir:
            adata = sc.read_h5ad(self.data_dir)
            self.total_samples = adata.n_obs
            self.batches_per_epoch = self.total_samples // self.batch_size
            print(f"Dataset info: {self.total_samples} samples, {self.batches_per_epoch} batches per epoch")
    
    def run_loop(self):
        """Run training loop with epoch-based tracking."""
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            # Start of new epoch
            self.epoch += 1
            self.epoch_losses = {}
            epoch_start_step = self.step
            
            print(f"\nEpoch {self.epoch} starting...")
            
            # Train for one epoch
            for batch_idx in range(self.batches_per_epoch):
                if (self.lr_anneal_steps and 
                    self.step + self.resume_step >= self.lr_anneal_steps):
                    break
                    
                batch, cond = next(self.data)
                self.run_step(batch, cond)
                
                if self.step % self.log_interval == 0:
                    logger.dumpkvs()
                if self.step % self.save_interval == 0:
                    self.save()
                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                self.step += 1
            
            # Print epoch statistics
            if self.epoch_losses:
                print(f"Epoch {self.epoch} completed (steps {epoch_start_step}-{self.step-1}):")
                for key, losses in self.epoch_losses.items():
                    avg_loss = sum(losses) / len(losses)
                    print(f"  {key}: {avg_loss:.6f}")
                    
                    # Track best loss (assuming we want to track the main loss)
                    if key == "loss" and avg_loss < self.best_loss:
                        self.best_loss = avg_loss
                        self.best_epoch = self.epoch
                        self.best_step = self.step - 1
                        print(f"  *** New best loss: {self.best_loss:.6f} at epoch {self.best_epoch}, step {self.best_step} ***")
            
            # Check if we should stop
            if (self.lr_anneal_steps and 
                self.step + self.resume_step >= self.lr_anneal_steps):
                break
        
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
        
        # Save best model
        if self.best_epoch > 0:
            print(f"\nSaving best model from epoch {self.best_epoch} (step {self.best_step}) with loss {self.best_loss:.6f}")
            self.save_best_model()
    
    def run_step(self, batch, cond):
        """Override run_step to accumulate epoch losses."""
        # Store current losses before forward_backward
        current_losses = self._get_current_losses_from_forward_backward(batch, cond)
        
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()
        
        # Accumulate losses for this epoch
        for key, val in current_losses.items():
            if key not in self.epoch_losses:
                self.epoch_losses[key] = []
            self.epoch_losses[key].append(val)
    
    def _get_current_losses_from_forward_backward(self, batch, cond):
        """Get losses from forward_backward method."""
        self.mp_trainer.zero_grad()
        losses_dict = {}
        
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            
            # Store losses for this micro-batch
            for key, values in losses.items():
                weighted_values = values * weights
                if key not in losses_dict:
                    losses_dict[key] = []
                losses_dict[key].append(weighted_values.mean().item())
            
            self.mp_trainer.backward(loss)
        
        # Return average losses across all micro-batches
        avg_losses = {}
        for key, values in losses_dict.items():
            avg_losses[key] = sum(values) / len(values)
        
        return avg_losses
    
    def save_best_model(self):
        """Save the best model."""
        if not os.path.exists(os.path.join(self.save_dir, self.timestamp)):
            os.makedirs(os.path.join(self.save_dir, self.timestamp))
        
        # Save best model
        state_dict = self.mp_trainer.master_params_to_state_dict(self.mp_trainer.master_params)
        filename = f"best_model_epoch={self.best_epoch}_step={self.best_step}_loss={self.best_loss:.6f}.pt"
        with open(os.path.join(self.save_dir, self.timestamp, filename), "wb") as f:
            torch.save(state_dict, f)
        
        # Save best EMA model if available
        if hasattr(self, 'ema_params') and self.ema_params:
            for rate, params in zip(self.ema_rate, self.ema_params):
                state_dict = self.mp_trainer.master_params_to_state_dict(params)
                filename = f"best_ema_{rate}_epoch={self.best_epoch}_step={self.best_step}_loss={self.best_loss:.6f}.pt"
                with open(os.path.join(self.save_dir, self.timestamp, filename), "wb") as f:
                    torch.save(state_dict, f)


def main():
    setup_seed(1234)
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir='output/logs/'+args.model_name)  # log file

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        vae_path=args.vae_path,
        train_vae=False,
    )

    logger.log("training...")
    train_loop = EpochTrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        model_name=args.model_name,
        save_dir=args.save_dir,
        data_dir=args.data_dir
    )
    train_loop.run_loop()


def create_argparser():
    defaults = dict(
        data_dir="/data1/lep/Workspace/guided-diffusion/data/tabula_muris/all.h5ad",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0001,
        lr_anneal_steps=500000,
        batch_size=256,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=1000,
        save_interval=200000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        vae_path = '',
        model_name='',
        save_dir=''
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    main()
