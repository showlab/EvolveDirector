from diffusion.model.respace import SpacedDiffusion, space_timesteps
from .model import gaussian_diffusion as gd


def IDDPM(
        timestep_respacing,
        noise_schedule="linear",
        use_kl=False,
        sigma_small=False,
        predict_xstart=False,
        learn_sigma=True,
        pred_sigma=True,
        rescale_learned_sigmas=False,
        diffusion_steps=1000,
        snr=False,
        return_startx=False,
):
    betas_n = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type_n = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type_n = gd.LossType.RESCALED_MSE
    else:
        loss_type_n = gd.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas_n,
        model_mean_type=(
            gd.ModelMeanType.START_X if predict_xstart else gd.ModelMeanType.EPSILON
        ),
        model_var_type=(
            (gd.ModelVarType.LEARNED_RANGE if learn_sigma else (
                                 gd.ModelVarType.FIXED_LARGE
                                 if not sigma_small
                                 else gd.ModelVarType.FIXED_SMALL
                             )
             )
            if pred_sigma
            else None
        ),
        loss_type=loss_type_n,
        snr=snr,
        return_startx=return_startx,
        # rescale_timesteps=rescale_timesteps,
    )