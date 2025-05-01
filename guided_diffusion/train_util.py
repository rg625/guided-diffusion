import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import numpy as np

from .losses import normal_kl
from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=True,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
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
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        # # Sanity check: capture initial encoder params for comparison
        # encoder_params_before = {
        #     name: param.clone().detach()
        #     for name, param in self.ddp_model.module.named_parameters()
        #     if param.requires_grad
        # }
        # print(encoder_params_before)
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()
        # # Sanity check: compare params before and after backward
        # for name, param in self.ddp_model.module.named_parameters():
        #     if param.requires_grad:
        #         before = encoder_params_before[name]
        #         after = param.detach()
        #         if th.allclose(before, after, atol=1e-6):
        #             print(f"[Sanity Check] Param '{name}' did NOT change — check training config!")
        #         else:
        #             pass
                    
    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
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
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"q{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    """
    Log the loss dictionary with quantile-specific logging for time steps.

    :param diffusion: The diffusion model (used for num_timesteps).
    :param ts: Tensors representing the time steps.
    :param losses: Dictionary of loss tensors.
    """
    for key, values in losses.items():
        # Log the mean of the loss
        logger.logkv_mean(key, values.mean().item())

        # Ensure ts and values are iterable
        ts = ts.view(-1) if ts.dim() == 0 else ts
        values = values.view(-1) if values.dim() == 0 else values

        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

class TrainGuidanceLoop(TrainLoop):
    """
    TrainGuidanceLoop inherits from TrainLoop and updates the forward_backward method
    to compute the ScoreVAE loss with guidance.
    """
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        beta=0.001,  # Weight for KL divergence in the loss
        debug=False,  # Debug mode flag
    ):
        super().__init__(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=batch_size,
            microbatch=microbatch,
            lr=lr,
            ema_rate=ema_rate,
            log_interval=log_interval,
            save_interval=save_interval,
            resume_checkpoint=resume_checkpoint,
            use_fp16=use_fp16,
            fp16_scale_growth=fp16_scale_growth,
            schedule_sampler=schedule_sampler,
            weight_decay=weight_decay,
            lr_anneal_steps=lr_anneal_steps,
        )
        self.beta = beta  # KL divergence weight
        self.debug = debug
        
        # Enable debug mode in the model if requested
        if hasattr(self.model, 'module'):
            self.model.module.debug = debug
        else:
            self.model.debug = debug

    def compute_losses(self, predicted_noise, true_noise):
        """
        Compute the diffusion loss (MSE between predicted and true noise).
        
        :param predicted_noise: Noise predicted by the model
        :param true_noise: True noise used to generate the perturbed input
        :return: Dictionary of losses
        """
        # Mean squared error loss
        losses = th.square(predicted_noise - true_noise)
        losses = th.sum(losses.reshape(losses.shape[0], -1), dim=-1)
        losses *= 1 / 2
        
        return {
            "loss": th.mean(losses)  # likelihood loss / reconstruction loss
        }
    
    def forward_backward(self, batch, cond):
        """
        Perform a forward and backward pass on the batch with optional conditioning.
        
        Args:
            batch (torch.Tensor): Input batch of data.
            cond (dict or None): Conditioning information if available.
        """
        device = dist_util.dev()
        self.mp_trainer.zero_grad()

        batch_size = batch.shape[0]
        for start_idx in range(0, batch_size, self.microbatch):
            end_idx = min(start_idx + self.microbatch, batch_size)
            microbatch = batch[start_idx:end_idx].to(device)
            is_last_microbatch = end_idx >= batch_size

            # Sample time steps and their corresponding weights
            t, weights = self.schedule_sampler.sample(microbatch.shape[0], device)

            # Encode the input to get latent conditioning
            # Use t=0 (clean data) for encoding since we want z_0
            zero_t = th.zeros_like(t, device=device)
            encoding = self.ddp_model.module.encode(microbatch, zero_t)
            
            if not cond:
                micro_cond = {"z_start": encoding['encoding']}
            else:
                micro_cond = {k: v[start_idx:end_idx].to(device) for k, v in cond.items()}
                if "z_start" not in micro_cond:
                    micro_cond["z_start"] = encoding['encoding']

            # Sample perturbed inputs x_t ~ q(x_t | x_0)
            noise = th.randn_like(microbatch, device=device, requires_grad=True)
            # perturbed_x = self.diffusion.q_sample(microbatch, t, noise=noise)
            # perturbed_x.requires_grad_(True)
            perturbed_x = microbatch + th.from_numpy(self.diffusion.sqrt_one_minus_alphas_cumprod).float().view(-1, 1, 1, 1).to(t.device)[t]*noise
            if self.debug:
                # Check perturbed_x
                print(f"DEBUG perturbed_x:")
                print(f"  - Shape: {perturbed_x.shape}")
                print(f"  - Requires grad: {perturbed_x.requires_grad}")
                print(f"  - Range: [{perturbed_x.min().item()}, {perturbed_x.max().item()}]")
                
                # Print encoder parameter stats before forward pass
                print("DEBUG PARAMETERS AT START:")
                for name, param in self.ddp_model.module.encoder_unet_model.named_parameters():
                    grad_norm = 0.0
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                    print(f"  {name}: grad_norm={grad_norm:.8f}")

            # Predict the noise using our ScoreVAE model
            predicted_noise = self.ddp_model(perturbed_x, t, **micro_cond)
            
            if self.debug:
                # Check predicted_noise
                print(f"DEBUG predicted_noise:")
                print(f"  - Shape: {predicted_noise.shape}")
                print(f"  - Requires grad: {predicted_noise.requires_grad}")
                print(f"  - Range: [{predicted_noise.min().item()}, {predicted_noise.max().item()}]")
                
                # Print encoder parameter stats after forward pass
                print("DEBUG PARAMETERS AT After forward pass:")
                for name, param in self.ddp_model.module.encoder_unet_model.named_parameters():
                    grad_norm = 0.0
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                    print(f"  {name}: grad_norm={grad_norm:.8f}")

            # Compute loss between predicted noise and original noise
            if is_last_microbatch or not self.use_ddp:
                losses = self.compute_losses(predicted_noise, noise)
            else:
                with self.ddp_model.no_sync():
                    losses = self.compute_losses(predicted_noise, noise)

            # If using a LossAwareSampler, update with observed losses
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

            # Total loss - for now just using the reconstruction loss
            # total_loss = losses["loss"]

            # Optional KL divergence loss
            mu, logvar = encoding['mu'], encoding['logvar']
            losses["kl_loss"] = - 0.5 * th.sum(1 + logvar.view(logvar.size(0), -1)
                                  - mu.view(mu.size(0), -1).pow(2)
                                  - logvar.view(logvar.size(0), -1).exp(), dim=1).mean()
            total_loss = losses["loss"] + self.beta * losses["kl_loss"]

            # Logging
            log_dict = {k: v * weights for k, v in losses.items()}
            log_loss_dict(self.diffusion, t, log_dict)

            # Backpropagate through the computational graph
            if is_last_microbatch or not self.use_ddp:
                self.mp_trainer.backward(total_loss)
            else:
                with self.ddp_model.no_sync():
                    self.mp_trainer.backward(total_loss)
                    
            if self.debug:
                # Check for gradients after backward
                print("DEBUG PARAMETERS AFTER BACKWARD:")
                for name, param in self.ddp_model.module.encoder_unet_model.named_parameters():
                    grad_norm = 0.0
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                    print(f"  {name}: grad_norm={grad_norm:.8f}")
            
    # def run_step(self, batch, cond):
    #     """
    #     Run a single training step.
    #     """
    #     # Forward and backward pass
    #     self.forward_backward(batch, cond)
        
    #     # Take optimizer step
    #     took_step = self.mp_trainer.optimize(self.opt)
    #     if took_step:
    #         # Update EMA parameters
    #         self._update_ema()
            
    #     # Return loss metrics for logging
    #     return took_step