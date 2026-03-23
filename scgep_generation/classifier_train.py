"""
Train the scDiffusion classifier on latent single-cell embeddings.
"""

import argparse
import math
import os
import random

import blobfile as bf
import numpy as np
import scanpy as sc
import torch
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from tqdm.auto import tqdm

from guided_diffusion import dist_util, logger
from guided_diffusion.cell_datasets_loader import load_data
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    create_classifier_and_diffusion,
)
from guided_diffusion.train_util import log_loss_dict, parse_resume_step_from_filename


class EpochClassifierTrainer:
    def __init__(self, args, model, diffusion, data, val_data, mp_trainer, opt, schedule_sampler):
        self.args = args
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.val_data = val_data
        self.mp_trainer = mp_trainer
        self.opt = opt
        self.schedule_sampler = schedule_sampler
        self.epoch = 0
        self.step = 0
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.best_step = 0
        self.epoch_losses = {}
        self.epoch_val_losses = {}

        adata = sc.read_h5ad(args.data_dir)
        self.total_samples = adata.n_obs
        self.batches_per_epoch = max(self.total_samples // args.batch_size, 1)
        self.total_epochs = max(math.ceil(args.iterations / self.batches_per_epoch), 1)
        self.epoch_progress = tqdm(
            total=self.total_epochs,
            desc="Classifier epochs",
            leave=True,
            dynamic_ncols=True,
            disable=not self.args.quiet,
        )
        print(f"Dataset info: {self.total_samples} samples, {self.batches_per_epoch} batches per epoch")

    def forward_backward_log(self, data_loader, prefix="train"):
        batch, extra = next(data_loader)
        labels = extra["y"].to(dist_util.dev())
        batch = batch.to(dist_util.dev())

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

            losses = {
                f"{prefix}_loss": loss.detach(),
                f"{prefix}_acc@1": compute_top_k(logits, sub_labels, k=1, reduction="none"),
            }
            log_loss_dict(self.diffusion, sub_t, losses)

            for key, values in losses.items():
                losses_dict.setdefault(key, []).append(values.mean().item())

            loss = loss.mean()
            if loss.requires_grad:
                if i == 0:
                    self.mp_trainer.zero_grad()
                self.mp_trainer.backward(loss * len(sub_batch) / len(batch))

        return {key: sum(values) / len(values) for key, values in losses_dict.items()}

    def train_epoch(self):
        self.epoch += 1
        self.epoch_losses = {}
        self.epoch_val_losses = {}
        epoch_start_step = self.step
        if not self.args.quiet:
            print(f"\nEpoch {self.epoch} starting...")

        for _ in range(self.batches_per_epoch):
            if self.step >= self.args.iterations:
                break
            current_losses = self.forward_backward_log(self.data, prefix="train")
            self.mp_trainer.optimize(self.opt)

            for key, val in current_losses.items():
                self.epoch_losses.setdefault(key, []).append(val)

            self.step += 1

            if self.args.anneal_lr:
                set_annealed_lr(self.opt, self.args.lr, self.step / self.args.iterations)

            if self.val_data is not None and self.step % self.args.eval_interval == 0:
                with th.no_grad():
                    with self.model.no_sync():
                        self.model.eval()
                        val_losses = self.forward_backward_log(self.val_data, prefix="val")
                        self.model.train()
                        for key, val in val_losses.items():
                            self.epoch_val_losses.setdefault(key, []).append(val)

            if not self.args.quiet and self.step % self.args.log_interval == 0:
                logger.dumpkvs()

            if self.step and dist.get_rank() == 0 and self.step % self.args.save_interval == 0:
                save_model(self.mp_trainer, self.opt, self.step, self.args.model_path)

        if self.epoch_losses:
            avg_epoch_losses = {
                key: sum(losses) / len(losses) for key, losses in self.epoch_losses.items()
            }
            train_loss = avg_epoch_losses.get("train_loss")
            if train_loss is not None and train_loss < self.best_loss:
                self.best_loss = train_loss
                self.best_epoch = self.epoch
                self.best_step = self.step - 1

            avg_val_losses = {
                key: sum(losses) / len(losses) for key, losses in self.epoch_val_losses.items()
            }

            if self.args.quiet:
                postfix = {"epoch": f"{self.epoch}/{self.total_epochs}"}
                if train_loss is not None:
                    postfix["loss"] = f"{train_loss:.6f}"
                if "train_acc@1" in avg_epoch_losses:
                    postfix["acc"] = f"{avg_epoch_losses['train_acc@1']:.4f}"
                if "val_loss" in avg_val_losses:
                    postfix["val_loss"] = f"{avg_val_losses['val_loss']:.6f}"
                if "val_acc@1" in avg_val_losses:
                    postfix["val_acc"] = f"{avg_val_losses['val_acc@1']:.4f}"
                if self.best_epoch > 0:
                    postfix["best"] = f"{self.best_loss:.6f}"
                self.epoch_progress.set_postfix(postfix)
                self.epoch_progress.update(1)
            else:
                print(f"Epoch {self.epoch} completed (steps {epoch_start_step}-{self.step - 1}):")
                for key, avg_loss in avg_epoch_losses.items():
                    print(f"  {key}: {avg_loss:.6f}")
                if train_loss is not None and self.best_epoch == self.epoch and self.best_step == self.step - 1:
                    print(
                        f"  *** New best loss: {self.best_loss:.6f} at epoch {self.best_epoch}, step {self.best_step} ***"
                    )

                if avg_val_losses:
                    print("  Validation results:")
                    for key, avg_loss in avg_val_losses.items():
                        print(f"    {key}: {avg_loss:.6f}")

    def train(self):
        while self.step < self.args.iterations:
            self.train_epoch()
            if self.step >= self.args.iterations:
                break

        if dist.get_rank() == 0:
            save_model(self.mp_trainer, self.opt, self.step, self.args.model_path)
            if self.best_epoch > 0:
                self.save_best_model()
        self.epoch_progress.close()
        dist.barrier()

    def save_best_model(self):
        if dist.get_rank() != 0:
            return
        os.makedirs(self.args.model_path, exist_ok=True)
        th.save(
            self.mp_trainer.master_params_to_state_dict(self.mp_trainer.master_params),
            os.path.join(
                self.args.model_path,
                f"best_model_epoch={self.best_epoch}_step={self.best_step}_loss={self.best_loss:.6f}.pt",
            ),
        )
        th.save(
            self.opt.state_dict(),
            os.path.join(
                self.args.model_path,
                f"best_opt_epoch={self.best_epoch}_step={self.best_step}_loss={self.best_loss:.6f}.pt",
            ),
        )


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model(mp_trainer, opt, step, model_path):
    if dist.get_rank() != 0:
        return
    os.makedirs(model_path, exist_ok=True)
    th.save(
        mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
        os.path.join(model_path, f"model{step:06d}.pt"),
    )
    th.save(opt.state_dict(), os.path.join(model_path, f"opt{step:06d}.pt"))


def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)
    values = (top_ks == labels[:, None]).float().sum(dim=-1)
    if reduction == "mean":
        return values.mean().item()
    return values


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
        vae_path="",
        latent_dim=128,
        model_path="",
        start_guide_time=500,
        num_class=12,
    )
    num_class = defaults["num_class"]
    defaults.update(classifier_and_diffusion_defaults())
    defaults["num_class"] = num_class
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
    args = create_argparser().parse_args()
    setup_seed(1234)

    dist_util.setup_dist()
    logger.configure(
        dir=os.path.join(os.path.dirname(args.model_path), "logs"),
        format_strs=["log", "csv"] if args.quiet else None,
    )

    logger.log("creating model and diffusion...")
    model, diffusion = create_classifier_and_diffusion(
        **args_to_dict(args, classifier_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion) if args.noised else None

    resume_step = 0
    if args.resume_checkpoint:
        resume_step = parse_resume_step_from_filename(args.resume_checkpoint)
        if dist.get_rank() == 0:
            logger.log(f"loading model from checkpoint: {args.resume_checkpoint}... at {resume_step} step")
            model.load_state_dict(
                dist_util.load_state_dict(args.resume_checkpoint, map_location=dist_util.dev())
            )

    dist_util.sync_params(model.parameters())

    mp_trainer = MixedPrecisionTrainer(
        model=model,
        use_fp16=args.classifier_use_fp16,
        initial_lg_loss_scale=16.0,
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
    val_data = None
    if args.val_data_dir:
        val_data = load_data(
            data_dir=args.val_data_dir,
            batch_size=args.batch_size,
            vae_path=args.vae_path,
            hidden_dim=args.latent_dim,
            train_vae=False,
        )

    logger.log("creating optimizer...")
    opt = AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)
    if args.resume_checkpoint:
        opt_checkpoint = bf.join(bf.dirname(args.resume_checkpoint), f"opt{resume_step:06d}.pt")
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            opt.load_state_dict(dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev()))

    trainer = EpochClassifierTrainer(
        args=args,
        model=model,
        diffusion=diffusion,
        data=data,
        val_data=val_data,
        mp_trainer=mp_trainer,
        opt=opt,
        schedule_sampler=schedule_sampler,
    )
    if resume_step > 0:
        trainer.step = resume_step
    trainer.train()


if __name__ == "__main__":
    main()
