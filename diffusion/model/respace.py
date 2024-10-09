import numpy as np
import torch as th

from .gaussian_diffusion import GaussianDiffusion


def space_timesteps(timesteps_n, section_cts):

    if isinstance(section_cts, str):
        if section_cts.startswith("ddim"):
            desired_count = int(section_cts[len("ddim") :])
            for i in range(1, timesteps_n):
                if len(range(0, timesteps_n, i)) == desired_count:
                    return set(range(0, timesteps_n, i))
            raise ValueError(
                f"cannot create exactly {timesteps_n} steps with an integer stride"
            )
        section_cts = [int(x) for x in section_cts.split(",")]
    size_per = timesteps_n // len(section_cts)
    extra = timesteps_n % len(section_cts)
    start_idx_n = 0
    all_steps = []
    for i, section_count_n in enumerate(section_cts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count_n:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count_n}"
            )
        frac_stride = 1 if section_count_n <= 1 else (size - 1) / (section_count_n - 1)
        cur_idx_n = 0.0
        taken_steps = []
        for _ in range(section_count_n):
            taken_steps.append(start_idx_n + round(cur_idx_n))
            cur_idx_n += frac_stride
        all_steps += taken_steps
        start_idx_n += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):


    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.tstep_map = []
        self.org_num_steps = len(kwargs["betas"])

        base_diffusion_n = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas_n = []
        for i, alpha_cumprod in enumerate(base_diffusion_n.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas_n.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.tstep_map.append(i)
        kwargs["betas"] = np.array(new_betas_n)
        super().__init__(**kwargs)

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def training_losses_diffusers(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().training_losses_diffusers(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.tstep_map, self.org_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:
    def __init__(self, model, tstep_map, org_num_steps):
        self.model_n = model
        self.timestep_map = tstep_map
        # self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = org_num_steps

    def __call__(self, x, timestep, **kwargs):
        map_tensor = th.tensor(self.timestep_map, device=timestep.device, dtype=timestep.dtype)
        new_ts = map_tensor[timestep]
        # if self.rescale_timesteps:
        #     new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model_n(x, timestep=new_ts, **kwargs)
