import random
import numpy as np
from tqdm import tqdm

from diffusion.model.utils import *


def ablation_sampler(
        net, latents, class_labels=None, cfg_scale=None, feat=None, randn_like=torch.randn_like,
        num_steps=18, sigma_min=None, sigma_max=None, rho=7,
        solver='heun', discretization='edm', schedule='linear', scaling='none',
        epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
        S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    assert solver in ['euler', 'heun']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']

    vp_sigma_n = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv_n = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv_n = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (
            sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv_n = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv_n = lambda sigma: sigma ** 2

    if sigma_min is None:
        vp_def = vp_sigma_n(beta_d=19.1, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma_n(beta_d=19.1, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == 'vp':
        org_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps_n = vp_sigma_n(vp_beta_d, vp_beta_min)(org_t_steps)
    elif discretization == 've':
        org_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps_n = ve_sigma(org_t_steps)
    elif discretization == 'iddpm':
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device):  # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps_n = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    else:
        assert discretization == 'edm'
        sigma_steps_n = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    if schedule == 'vp':
        sigma = vp_sigma_n(vp_beta_d, vp_beta_min)
        sigma_deriv_n = vp_sigma_deriv_n(vp_beta_d, vp_beta_min)
        sigma_inv_n = vp_sigma_inv_n(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv_n = ve_sigma_deriv_n
        sigma_inv_n = ve_sigma_inv_n
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv_n = lambda t: 1
        sigma_inv_n = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv_n(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

    # Compute final time steps based on the corresponding noise levels.
    t_stps = sigma_inv_n(net.round_sigma(sigma_steps_n))
    t_stps = torch.cat([t_stps, torch.zeros_like(t_stps[:1])])  # t_N = 0

    # Main sampling loop.
    t_next = t_stps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
    for i, (t_cur, t_next) in enumerate(zip(t_stps[:-1], t_stps[1:])):  # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
        t_hat = sigma_inv_n(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(
            t_hat) * S_noise * randn_like(x_cur)

        # Euler step.
        h = t_next - t_hat
        denoised = net(x_hat.float() / s(t_hat), sigma(t_hat), class_labels, cfg_scale, feat=feat)['x'].to(
            torch.float64)
        d_cur = (sigma_deriv_n(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv_n(t_hat) * s(
            t_hat) / sigma(t_hat) * denoised
        x_prime = x_hat + alpha * h * d_cur
        t_prime = t_hat + alpha * h

        # Apply 2nd order correction.
        if solver == 'euler' or i == num_steps - 1:
            x_next = x_hat + h * d_cur
        else:
            assert solver == 'heun'
            denoised = net(x_prime.float() / s(t_prime), sigma(t_prime), class_labels, cfg_scale, feat=feat)['x'].to(
                torch.float64)
            d_prime = (sigma_deriv_n(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)) * x_prime - sigma_deriv_n(
                t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)

    return x_next


def edm_sampler(
        net, latents, class_labels=None, cfg_scale=None, randn_like=torch.randn_like,
        num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
        S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, **kwargs
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

    # Main sampling loop.
    x_next_n = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:])))):  # 0, ..., N-1
        x_cur = x_next_n

        # Increase noise temporarily.
        gamma_n = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma_n * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat.float(), t_hat, class_labels, cfg_scale, **kwargs)['x'].to(torch.float64)
        d_cur_n = (x_hat - denoised) / t_hat
        x_next_n = x_hat + (t_next - t_hat) * d_cur_n

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next_n.float(), t_next, class_labels, cfg_scale, **kwargs)['x'].to(torch.float64)
            d_prime = (x_next_n - denoised) / t_next
            x_next_n = x_hat + (t_next - t_hat) * (0.5 * d_cur_n + 0.5 * d_prime)

    return x_next_n