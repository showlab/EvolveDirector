import sys
sys.path.append("./")
import os
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import warnings
warnings.filterwarnings("ignore")  # ignore warning
import re
import argparse
from datetime import datetime
from tqdm import tqdm
import torch
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL

from diffusion.model.utils import prepare_prompt_ar
from diffusion import IDDPM, DPMS, SASolverSampler
from tools.download import find_model
from diffusion.model.nets import PixArtMS_XL_2, PixArtMSLN_XL_2, PixArt_XL_2, PixArtLN_XL_2
from diffusion.model.t5 import T5Embedder
from diffusion.data.datasets import get_chunks, ASPECT_RATIO_256_TEST, ASPECT_RATIO_512_TEST, ASPECT_RATIO_1024_TEST
import json
import time
import random


def get_formatted_timestamp():
    now = datetime.now()
    formatted_timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    return formatted_timestamp


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sync_root', default='./Synchronizer', type=str)
    parser.add_argument('--task_stamp', default='dynamic_0000', type=str)
    parser.add_argument('--t5_path', default='/path/to/Edgen', type=str)
    parser.add_argument('--tokenizer_path', default='/path/to/Edgen/sd-vae-ft-ema', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--select_ratio', default=0.2, type=float)

    parser.add_argument('--cfg_scale', default=4.5, type=float)
    parser.add_argument('--sampling_algo', default='dpm-solver', type=str, choices=['iddpm', 'dpm-solver', 'sa-solver'])
    parser.add_argument('--step', default=-1, type=int)

    return parser.parse_args()


def set_env(seed, image_size_to_generate):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    for _ in range(30):
        torch.randn(1, 4, image_size_to_generate, image_size_to_generate)


@torch.inference_mode()
def visualize(items, sample_steps, cfg_scale, save_root, image_size_to_generate, bs=1):

    for chunk in tqdm(list(get_chunks(items, bs)), unit='batch'):

        prompts = []
        file_names = []

        inf_item = chunk[0]
        prompt = inf_item['prompt']
        prompt += f' --ar {int(inf_item["height"])}:{int(inf_item["width"])}'
        prompt_clean, _, hw, ar, custom_hw = prepare_prompt_ar(prompt, base_ratios, device=device, show=False)  # ar for aspect ratio
        if image_size_to_generate == 1024:
            latent_size_h, latent_size_w = int(hw[0, 0] // 8), int(hw[0, 1] // 8)
        else:
            hw = torch.tensor([[image_size_to_generate, image_size_to_generate]], dtype=torch.float, device=device).repeat(bs, 1)
            ar = torch.tensor([[1.]], device=device).repeat(bs, 1)
            latent_size_h, latent_size_w = latent_size, latent_size
        prompts.append(prompt_clean.strip())

        file_name = os.path.basename(inf_item['path'])
        file_names.append(file_name)

        null_y = model.y_embedder.y_embedding[None].repeat(len(prompts), 1, 1)[:, None]

        with torch.no_grad():
            caption_embs, emb_masks = t5.get_text_embeddings(prompts)
            caption_embs = caption_embs.float()[:, None]

            if args.sampling_algo == 'iddpm':
                # Create sampling noise:
                n = len(prompts)
                z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device).repeat(2, 1, 1, 1)
                model_kwargs = dict(y=torch.cat([caption_embs, null_y]),
                                    cfg_scale=cfg_scale, data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
                diffusion = IDDPM(str(sample_steps))
                # Sample images:
                samples = diffusion.p_sample_loop(
                    model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
                    device=device
                )
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            elif args.sampling_algo == 'dpm-solver':
                # Create sampling noise:
                n = len(prompts)
                z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device)
                model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
                dpm_solver = DPMS(model.forward_with_dpmsolver,
                                  condition=caption_embs,
                                  uncondition=null_y,
                                  cfg_scale=cfg_scale,
                                  model_kwargs=model_kwargs)
                samples = dpm_solver.sample(
                    z,
                    steps=sample_steps,
                    order=2,
                    skip_type="time_uniform",
                    method="multistep",
                )
            elif args.sampling_algo == 'sa-solver':
                # Create sampling noise:
                n = len(prompts)
                model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
                sa_solver = SASolverSampler(model.forward_with_dpmsolver, device=device)
                samples = sa_solver.sample(
                    S=25,
                    batch_size=n,
                    shape=(4, latent_size_h, latent_size_w),
                    eta=1,
                    conditioning=caption_embs,
                    unconditional_conditioning=null_y,
                    unconditional_guidance_scale=cfg_scale,
                    model_kwargs=model_kwargs,
                )[0]
        samples = vae.decode(samples / 0.18215).sample
        torch.cuda.empty_cache()
        # Save images:
        os.umask(0o000)  # file permission: 666; dir permission: 777
        for i, sample in enumerate(samples):
            save_path = os.path.join(save_root, file_names[i])
            save_image(sample, save_path, nrow=1, normalize=True, value_range=(-1, 1))


if __name__ == '__main__':
    args = get_args()
    # Setup PyTorch:
    print(f"inference with select ratio: {args.select_ratio}")
    print(f"inference with taskstamp: {args.task_stamp}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert args.sampling_algo in ['iddpm', 'dpm-solver', 'sa-solver']

    vae = AutoencoderKL.from_pretrained(args.tokenizer_path).to(device)
    t5 = T5Embedder(device="cuda", local_cache=True, cache_dir=args.t5_path, torch_dtype=torch.float)

    # data setting
    singal_folder = os.path.join(args.sync_root, args.task_stamp)
    gen_singal_path = os.path.join(singal_folder, "gen_img_flag.txt")

    dynamic_json = os.path.join(singal_folder, "dynamic_set.json")

    save_root_folder = os.path.join(singal_folder, "base_imgs_Edgen")
    checkpoint_save_path = os.path.join(singal_folder, "checkpoint_save_path.txt")
    
    while True:
        if os.path.exists(gen_singal_path):
            with open(gen_singal_path, "r") as f:
                if f.read().strip() != "Start":
                    time.sleep(10)
                    continue
        else:
            time.sleep(10)
            continue

        # select text prompts to generate images
        with open(dynamic_json, 'r') as f:
            items = json.load(f)
        
        if len(items)>100:
            selected_items = random.sample(items, int(len(items)*args.select_ratio))
        else:
            selected_items = items

        print(f'{len(selected_items)} images to generate')

        formatted_timestamp = get_formatted_timestamp()

        save_root_folder_withtime = os.path.join(save_root_folder, formatted_timestamp)
        os.makedirs(save_root_folder_withtime, exist_ok=True)
        dynamic_json_selected = os.path.join(save_root_folder_withtime, 'dynamic_subset.json')
        with open(dynamic_json_selected, 'w') as f:
            json.dump(selected_items, f, indent=4)

        # load model
        if int(selected_items[0]["height"]) == 512 and int(selected_items[0]["width"]) == 512:
            image_size_to_generate = 512
        else:
            image_size_to_generate = 1024

        seed = args.seed
        set_env(seed, image_size_to_generate)

        latent_size = image_size_to_generate // 8
        lewei_scale = {256: 1, 512: 1, 1024: 2}     # trick for positional embedding interpolation
        sample_steps_dict = {'iddpm': 100, 'dpm-solver': 20, 'sa-solver': 25}
        sample_steps = args.step if args.step != -1 else sample_steps_dict[args.sampling_algo]
        weight_dtype = torch.float16

        # model setting
        if image_size_to_generate == 512:
            model = PixArtLN_XL_2(input_size=latent_size, lewei_scale=lewei_scale[image_size_to_generate]).to(device)
        else:
            model = PixArtMSLN_XL_2(input_size=latent_size, lewei_scale=lewei_scale[image_size_to_generate]).to(device)

        base_ratios = eval(f'ASPECT_RATIO_{image_size_to_generate}_TEST')

        with open(checkpoint_save_path, "r") as f:
            new_model_path = f.read()

        print(f"Generating sample from ckpt: {new_model_path}")
        state_dict = find_model(new_model_path)
        del state_dict['state_dict']['pos_embed']
        missing, unexpected = model.load_state_dict(state_dict['state_dict'], strict=False)
        model.eval()
        model.to(weight_dtype)

        # generate images
        save_root = os.path.join(save_root_folder_withtime, 'new')
        os.makedirs(save_root, exist_ok=True)
        gen_folder_path = os.path.join(singal_folder, "gen_img_folder.txt")
        with open(gen_folder_path, "w") as f:
            f.write(save_root_folder_withtime)

        with open(gen_singal_path, "w") as f:
            f.write("Advanced Model Start")

        try:
            visualize(selected_items, sample_steps, args.cfg_scale, save_root, image_size_to_generate)
        except AssertionError:
            try:
                visualize(selected_items, sample_steps, args.cfg_scale, save_root, image_size_to_generate, bs=len(selected_items))
            except AssertionError:
                print('last file not processed')

        print('base img gen finished')
        while True:
            with open(gen_singal_path, "r") as f:  # wait for advanced models generation
                if f.read().strip() != "Advanced Model Finished":
                    time.sleep(10)
                    continue
                else:
                    break
        with open(gen_singal_path, "w") as f:
            f.write("Finished")
