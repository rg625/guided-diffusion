"""
Train a diffusion model on images.
"""

import argparse

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_guided_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch

def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()

    logger.log("Creating model and diffusion...")
    model = create_guided_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    model.unet_model.eval()  # freeze unconditional model
    for p in model.unet_model.parameters():
        p.requires_grad = False

    logger.log("Creating schedule sampler...")
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, model)

    logger.log("Loading data...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("Training encoder using ScoreVAE loss...")
    optimizer = torch.optim.Adam(model.encoder_unet_model.parameters(), lr=args.lr)

    for step, batch in enumerate(data):
        x_start = batch[0].to(dist_util.dev())
        t, weights = schedule_sampler.sample(x_start.shape[0], dist_util.dev())

        # 1. Diffuse input images to time t
        x_t = model.q_sample(x_start, t)

        # 2. Compute unconditional score sθ(x_t, t)
        x_t.requires_grad = True
        with torch.no_grad():
            score_torcheta = model.unet_model(x_t, t)

        # 3. Sample latent z ~ q(z|x_t) and compute ∇x log q(z | x_t)
        encoded = model.encode(x_t, t)
        z = encoded['cond_fn']
        grad_log_q = model.score(x_t, z, t)

        # 4. Combine scores
        score_phi = score_torcheta + grad_log_q

        # 5. Compute target score ∇x log p_t(x_t | x_0)
        with torch.no_grad():
            target_score = model._predict_xstart_from_eps(x_t, t, score_torcheta)  # Optional, depends on model

        # 6. Compute ScoreVAE loss
        g_t = model.g(t) if hasattr(model, "g") else 1.0

        # Main guidance loss
        score_loss = (g_t**2 * (target_score - score_phi).square().sum(dim=[1, 2, 3])).mean()

        # 7. Compute KL(q(z|x_0) || p(z)) for z ~ q(z | x_0)
        z_dist = model.encode(x_start)['cond_fn']
        mu, logvar = encoded['mu'], encoded['logvar']
        kl = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1, dim=-1)
        kl_loss = kl.mean()

        total_loss = score_loss + args.beta * kl_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % args.log_interval == 0:
            logger.log(f"step {step}: score_loss={score_loss.item():.4f}, kl={kl_loss.item():.4f}")

        if step % args.save_interval == 0:
            logger.save_model("encoder_checkpoint.pt", model.encoder_unet_model)

def create_argparser():
    defaults = dict(
        data_dir="/home/rg625/datasets/cifar10/images/",
        image_size=32,
        schedule_sampler="uniform",
        lr=1e-4,
        beta=1.0,
        batch_size=64,
        log_interval=10,
        save_interval=10000,
        class_cond=False,
        use_fp16=False,
        use_checkpoint=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()