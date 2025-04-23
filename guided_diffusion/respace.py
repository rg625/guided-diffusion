import numpy as np
import torch as th

from .gaussian_diffusion import GaussianDiffusion


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.

    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t

class GuidedDiffusionModel(SpacedDiffusion, th.nn.Module):
    def __init__(self, unet_model, encoder_unet_model, unet_ckpt=None, use_timesteps=None, **kwargs):
        if use_timesteps is None:
            raise ValueError("Must provide use_timesteps to initialize SpacedDiffusion.")
        th.nn.Module.__init__(self)           # <-- Initialize PyTorch module system first
        SpacedDiffusion.__init__(self, use_timesteps=use_timesteps, **kwargs)  # <-- Then init the diffusion base class        
        self.unet_model = unet_model
        self.encoder_unet_model = encoder_unet_model

        if unet_ckpt is not None:
            self.load_pretrained(unet_ckpt)
            for param in self.unet_model.parameters():
                param.requires_grad = False
        else:
            raise ValueError("Cannot train encoder without unconditional model")

    def load_pretrained(self, path):
        state = th.load(path, map_location="cpu")
        print(f"Loading pretrained weights from {path}, step: {state.get('global_step', 'N/A')}")
        self.unet_model.load_state_dict(state, strict=True)

    def guided_sample(self, x, t, x_start, guidance_scale=1.0):
        """
        Apply classifier-free guidance during sampling.
        Uses the gradient of the encoder's log-density to steer the diffusion process.
        """
        # Get the p_mean_var (mean, variance, and log variance)
        p_mean_var = self.p_mean_variance(x, t)

        # Compute the guidance score (∇ log p(x|z))
        z = self.encode(x_start)['cond_fn']
        score_grad = self.score(x_start, z, t)

        # Adjust the mean with the gradient (guidance)
        new_mean = p_mean_var['mean'] + guidance_scale * score_grad

        return {
            'mean': new_mean,
            'variance': p_mean_var['variance'],
            'log_variance': p_mean_var['log_variance']
        }

    def encode(self, x_start, t=None):
        """
        Encoder for conditioning. Using the encoder as the first half of the U-Net.
        """
        if t is None:
            t = th.zeros(x_start.shape[0]).type_as(x_start)

        latent_distribution_parameters = self.encoder_unet_model(x_start, t)

        channels = latent_distribution_parameters.size(1) // 2
        mean_z = latent_distribution_parameters[:, :channels]
        log_var_z = latent_distribution_parameters[:, channels:]

        cond = mean_z + (0.5 * log_var_z).exp() * th.randn_like(mean_z)

        return {
            'cond_fn': cond,
            'mu': mean_z,
            'logvar': log_var_z
        }

    def log_density(self, x, z, t):
        """
        Log density for latent variable.
        """
        latent_distribution_parameters = self.encoder_unet_model(x, t)
        channels = latent_distribution_parameters.size(1) // 2
        mean_z = latent_distribution_parameters[:, :channels]
        log_var_z = latent_distribution_parameters[:, channels:]

        mean_z_flat = mean_z.view(mean_z.size(0), -1)
        log_var_z_flat = log_var_z.view(log_var_z.size(0), -1)
        z_flat = z.view(z.size(0), -1)

        logdensity = -0.5 * th.sum(th.square(z_flat - mean_z_flat) / log_var_z_flat.exp(), dim=1)
        return logdensity

    def score(self, x, z, t):
        """
        Computes the score function for classifier-free guidance.
        """
        device = x.device
        x.requires_grad = True
        ftx = self.log_density(x, z, t)
        grad_log_density = th.autograd.grad(outputs=ftx, inputs=x,
                                                grad_outputs=th.ones(ftx.size()).to(device),
                                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        assert grad_log_density.size() == x.size()
        return grad_log_density

class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)
