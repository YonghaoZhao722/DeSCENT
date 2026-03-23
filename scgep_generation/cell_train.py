"""
Train the scDiffusion backbone on latent single-cell embeddings.
"""

import argparse
import functools
import math
import os
import random

import numpy as np
import scanpy as sc
import torch
from tqdm.auto import tqdm

from guided_diffusion import dist_util, logger
from guided_diffusion.cell_datasets_loader import load_data
from guided_diffusion.resample import LossAwareSampler, create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from guided_diffusion.train_util import TrainLoop


class EpochTrainLoop(TrainLoop):
    def __init__(self, *args, data_dir=None, quiet=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch = 0
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.best_step = 0
        self.epoch_losses = {}
        self.data_dir = data_dir
        self.quiet = quiet

        adata = sc.read_h5ad(self.data_dir)
        self.total_samples = adata.n_obs
        self.batches_per_epoch = max(self.total_samples // self.batch_size, 1)
        print(f"Dataset info: {self.total_samples} samples, {self.batches_per_epoch} batches per epoch")

    def run_loop(self):
        total_epochs = None
        if self.lr_anneal_steps:
            remaining_steps = max(self.lr_anneal_steps - (self.step + self.resume_step), 0)
            total_epochs = max(math.ceil(remaining_steps / self.batches_per_epoch), 1)
        epoch_progress = tqdm(
            total=total_epochs,
            desc="Diffusion epochs",
            leave=True,
            dynamic_ncols=True,
            disable=not self.quiet,
        )

        while not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps:
            self.epoch += 1
            self.epoch_losses = {}
            epoch_start_step = self.step
            if not self.quiet:
                print(f"\nEpoch {self.epoch} starting...")

            for _ in range(self.batches_per_epoch):
                if self.lr_anneal_steps and self.step + self.resume_step >= self.lr_anneal_steps:
                    break
                batch, cond = next(self.data)
                self.run_step(batch, cond)

                if not self.quiet and self.step % self.log_interval == 0:
                    logger.dumpkvs()
                if self.step % self.save_interval == 0:
                    self.save()
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        epoch_progress.close()
                        return
                self.step += 1

            if self.epoch_losses:
                avg_epoch_losses = {
                    key: sum(losses) / len(losses) for key, losses in self.epoch_losses.items()
                }
                avg_loss = avg_epoch_losses.get("loss")
                if avg_loss is not None and avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    self.best_epoch = self.epoch
                    self.best_step = self.step - 1

                if self.quiet:
                    postfix = {}
                    if total_epochs is not None:
                        postfix["epoch"] = f"{self.epoch}/{total_epochs}"
                    else:
                        postfix["epoch"] = str(self.epoch)
                    if avg_loss is not None:
                        postfix["loss"] = f"{avg_loss:.6f}"
                    if self.best_epoch > 0:
                        postfix["best"] = f"{self.best_loss:.6f}"
                    epoch_progress.set_postfix(postfix)
                    epoch_progress.update(1)
                else:
                    print(f"Epoch {self.epoch} completed (steps {epoch_start_step}-{self.step - 1}):")
                    for key, epoch_avg_loss in avg_epoch_losses.items():
                        print(f"  {key}: {epoch_avg_loss:.6f}")
                    if avg_loss is not None and self.best_epoch == self.epoch and self.best_step == self.step - 1:
                        print(
                            f"  *** New best loss: {self.best_loss:.6f} at epoch {self.best_epoch}, step {self.best_step} ***"
                        )

            if self.lr_anneal_steps and self.step + self.resume_step >= self.lr_anneal_steps:
                break

        epoch_progress.close()

        if (self.step - 1) % self.save_interval != 0:
            self.save()
        if self.best_epoch > 0:
            self.save_best_model()

    def run_step(self, batch, cond):
        current_losses = self._get_current_losses_from_forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()
        for key, val in current_losses.items():
            self.epoch_losses.setdefault(key, []).append(val)

    def _get_current_losses_from_forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        losses_dict = {}
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {k: v[i : i + self.microbatch].to(dist_util.dev()) for k, v in cond.items()}
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
                self.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

            loss = (losses["loss"] * weights).mean()
            for key, values in losses.items():
                losses_dict.setdefault(key, []).append((values * weights).mean().item())
            self.mp_trainer.backward(loss)

        return {key: sum(values) / len(values) for key, values in losses_dict.items()}

    def save_best_model(self):
        output_dir = os.path.join(self.save_dir, self.timestamp)
        os.makedirs(output_dir, exist_ok=True)
        state_dict = self.mp_trainer.master_params_to_state_dict(self.mp_trainer.master_params)
        filename = f"best_model_epoch={self.best_epoch}_step={self.best_step}_loss={self.best_loss:.6f}.pt"
        torch.save(state_dict, os.path.join(output_dir, filename))

        for rate, params in zip(self.ema_rate, self.ema_params):
            ema_state = self.mp_trainer.master_params_to_state_dict(params)
            ema_filename = f"best_ema_{rate}_epoch={self.best_epoch}_step={self.best_step}_loss={self.best_loss:.6f}.pt"
            torch.save(ema_state, os.path.join(output_dir, ema_filename))


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0001,
        lr_anneal_steps=500000,
        batch_size=256,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=1000,
        save_interval=200000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        vae_path="",
        model_name="",
        save_dir="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument("--quiet", action="store_true")
    return parser


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    setup_seed(1234)
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(
        dir=os.path.join(args.save_dir, "logs", args.model_name),
        format_strs=["log", "csv"] if args.quiet else None,
    )

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
        data_dir=args.data_dir,
        quiet=args.quiet,
    )
    train_loop.run_loop()


if __name__ == "__main__":
    main()
