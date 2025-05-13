import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import numpy as np
import wandb

from .losses import normal_kl
from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema, mean_flat
from .resample import LossAwareSampler, UniformSampler
from .gaussian_diffusion import _extract_into_tensor
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
        wandb_project="ScoreVAE",  # New argument for wandb project name
        wandb_config=None,  # New argument for additional wandb config
        ):

                # Initialize wandb
        wandb.init(
            project=wandb_project,
            config=wandb_config or {
                "batch_size": batch_size,
                "lr": lr,
                "ema_rate": ema_rate,
                "weight_decay": weight_decay,
                "lr_anneal_steps": lr_anneal_steps,
            },
        )
        wandb.watch(model, log="all")  # Log gradients and model parameters

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
        encoder_params_before = {
            name: param.clone().detach()
            for name, param in self.ddp_model.module.named_parameters()
            if param.requires_grad
        }

        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()
        self.log_param_and_grad_norms()
        # Sanity check: compare params before and after backward
        for name, param in self.ddp_model.module.named_parameters():
            if param.requires_grad:
                before = encoder_params_before[name]
                after = param.detach()
                if th.allclose(before, after, atol=1e-6):
                    print(f"[Sanity Check] Param '{name}' did NOT change — check training config!")
                else:
                    pass
                    
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
        # Log step and learning rate
        current_lr = self.opt.param_groups[0]["lr"]
        wandb.log({
            "step": self.step + self.resume_step,
            "samples": (self.step + self.resume_step + 1) * self.global_batch,
            "learning_rate": current_lr,
        })

    def log_param_and_grad_norms(self):
        """
        Log parameter and gradient norms using wandb.
        """
        total_param_norm = 0
        total_grad_norm = 0
        param_count = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_norm = param.data.norm(2).item()
                grad_norm = param.grad.norm(2).item() if param.grad is not None else 0
                total_param_norm += param_norm
                total_grad_norm += grad_norm
                param_count += 1

                # Log individual parameter and gradient norms
                wandb.log({
                    f"param_norm/{name}": param_norm,
                    f"grad_norm/{name}": grad_norm,
                })

        # Log total parameter and gradient norms
        wandb.log({
            "total_param_norm": total_param_norm / param_count,
            "total_grad_norm": total_grad_norm / param_count,
        })

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
            wandb.log({f"loss/{key}_q{quartile}": sub_loss})
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
        beta=1e-6,  # Weight for KL divergence in the loss
        guidance_scale=1.0,  # Weight for guidance loss
        debug=False,  # Debug mode flag
        wandb_project="ScoreVAE",  # Wandb project name
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
            wandb_project=wandb_project,
        )
        self.beta = beta
        self.guidance_scale = guidance_scale
        self.debug = debug
        print("Using resume checkpoint:", resume_checkpoint)
        print("Exists:", os.path.exists(resume_checkpoint))
        # Enable debug mode in the model if requested
        if hasattr(self.model, 'module'):
            self.model.module.debug = debug
        else:
            self.model.debug = debug
            
        # Print encoder parameters for debugging
        self._log_encoder_info()
    
    def _log_encoder_info(self):
        """Log information about the encoder parameters for debugging."""
        encoder = self.ddp_model.module.encoder_unet_model if hasattr(self.ddp_model, 'module') else self.ddp_model.encoder_unet_model
        
        print(f"Encoder architecture: {encoder.__class__.__name__}")
        print(f"Total encoder parameters: {sum(p.numel() for p in encoder.parameters())}")
        print(f"Trainable encoder parameters: {sum(p.numel() for p in encoder.parameters() if p.requires_grad)}")
        
    def compute_losses(self, predicted_noise, true_noise, perturbed_x=None, t=None, encoding=None):
        """
        Compute the diffusion loss (MSE between predicted and true noise).
        Optional reconstruction terms can be added to improve encoder gradient flow.
        """
        losses = {
            "mse": mean_flat((true_noise - predicted_noise)**2).sum()
        }
        
        # Add additional losses if needed
        if encoding is not None:
            # Optional: add reconstruction loss for better encoder gradients
            mu, logvar = encoding["mu"], encoding["logvar"]
            losses["kl"] = -0.5 * th.sum(
                1 + logvar.view(logvar.size(0), -1)
                - mu.view(mu.size(0), -1).pow(2)
                - logvar.view(logvar.size(0), -1).exp(),
                dim=1
            ).mean()
            
        return losses
    
    def forward_backward(self, batch, cond):
        """
        Perform a forward and backward pass with improved gradient flow to the encoder.
        """
        device = dist_util.dev()
        self.mp_trainer.zero_grad()
        batch_size = batch.shape[0]
        
        # Store gradients for all encoder parameters to verify backprop
        encoder_grads = {}
        if self.debug:
            for name, param in self.ddp_model.module.encoder_unet_model.named_parameters():
                if param.requires_grad:
                    param.register_hook(lambda grad, name=name: encoder_grads.update({name: grad.clone()}))

        for start_idx in range(0, batch_size, self.microbatch):
            end_idx = min(start_idx + self.microbatch, batch_size)
            micro = batch[start_idx:end_idx].to(device)
            is_last_microbatch = end_idx >= batch_size

            # Sample time steps and weights
            t, weights = self.schedule_sampler.sample(micro.shape[0], device)

            # Encode the input to get latent conditioning - CRITICAL CHANGE:
            # Use the same t as diffusion for encoding to ensure gradient flow
            encoding = self.ddp_model.module.encode(micro, th.full(size=(micro.size(0),), fill_value=-1).long().type_as(micro))
            
            micro_cond = cond or {}
            micro_cond = {k: v[start_idx:end_idx].to(device) for k, v in micro_cond.items()}
            
            # Store encoding with requires_grad=True to ensure gradient flow
            z_start = encoding["encoding"]
            micro_cond["z_start"] = z_start

            # Sample perturbed inputs x_t ~ q(x_t | x_0)
            noise = th.randn_like(micro, device=device)
            perturbed_x = self.diffusion.q_sample(micro, t, noise=noise)
            
            # IMPORTANT: Remove requires_grad on perturbed_x to avoid double gradient issues
            perturbed_x = perturbed_x.detach().requires_grad_(True)

            # Compute gradient from guidance module directly through encoder
            with th.enable_grad():
                # Get the guidance correction
                epsilon_correction = self.ddp_model(perturbed_x, t, **micro_cond)
                
                # Scale by the diffusion term
                diffusion_scale = _extract_into_tensor(
                    self.diffusion.sqrt_one_minus_alphas_cumprod, t, perturbed_x.shape
                ).to(t.device)
                
                epsilon_correction = -epsilon_correction * diffusion_scale

            # Get the base unconditional prediction
            with th.no_grad():
                epsilon = self.ddp_model.module.unet_model(perturbed_x, t)[:, :3, :]
            
            # Combine the two
            corrected_eps = epsilon + self.guidance_scale * epsilon_correction

            # Compute losses
            if is_last_microbatch or not self.use_ddp:
                losses = self.compute_losses(
                    corrected_eps, 
                    noise, 
                    perturbed_x=perturbed_x,
                    t=t,
                    encoding=encoding
                )
            else:
                with self.ddp_model.no_sync():
                    losses = self.compute_losses(
                        corrected_eps, 
                        noise,
                        perturbed_x=perturbed_x,
                        t=t,
                        encoding=encoding
                    )

            # Update LossAwareSampler if in use
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(t, losses["mse"].detach())

            # Log additional metrics for debugging
            mu, logvar = encoding["mu"], encoding["logvar"]
            
            # Verbose logging for debugging
            if self.debug and is_last_microbatch:
                print(f"Timestep mean: {t.float().mean().item()}")
                print(f"Mu norm: {mu.norm(2, dim=1).mean().item()}")
                print(f"LogVar mean: {logvar.mean().item()}")
                print(f"Encoding norm: {z_start.norm(2, dim=1).mean().item()}")
                print(f"Noise unconditional norm: {epsilon.norm(2, dim=1).mean().item()}")
                print(f"Epsilon correction norm: {epsilon_correction.norm(2, dim=1).mean().item()}")
            
            wandb.log({
                "encoding/mu_norm": mu.norm(2, dim=1).mean().item(),
                "encoding/var": logvar.exp().mean().item(),
                "encoding/encoding_norm": z_start.norm(2, dim=1).mean().item(),
                "noise/unconditional_norm": epsilon.norm(2, dim=1).mean().item(),
                "noise/conditional_norm": epsilon_correction.norm(2, dim=1).mean().item(),
                "noise/total_norm": corrected_eps.norm(2, dim=1).mean().item(),
                "time/mean": t.float().mean().item(),
            })

            # Calculate the total loss - scale KL to be comparable to MSE
            total_loss = losses["mse"] + self.beta * losses["kl"]

            # Log losses
            log_dict = {k: v * weights for k, v in losses.items()}
            log_loss_dict(self.diffusion, t, log_dict)

            # Explicit backward for the guidance loss
            if is_last_microbatch or not self.use_ddp:
                self.mp_trainer.backward(total_loss)
            else:
                with self.ddp_model.no_sync():
                    self.mp_trainer.backward(total_loss)
            
        # Debug gradient flow if enabled
        if self.debug:
            encoder_params = self.ddp_model.module.encoder_unet_model.named_parameters()
            print("\nEncoder gradient norms:")
            for name, param in encoder_params:
                if param.requires_grad and param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if grad_norm < 1e-4:
                        print(f"  {name}: {grad_norm:.6f} ⚠️ LOW")
                    else:
                        print(f"  {name}: {grad_norm:.6f}")
                elif param.requires_grad:
                    print(f"  {name}: None ⚠️ NO GRAD")
                    