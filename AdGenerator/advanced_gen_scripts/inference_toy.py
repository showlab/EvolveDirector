import argparse
from diffusers import DiffusionPipeline
import torch
import os
import warnings
warnings.filterwarnings("ignore")  # ignore warning
import re
import argparse
from datetime import datetime
from tqdm import tqdm
import time
import random
import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_ckpt_path', default='/path/to/playground-v2.5-1024px-aesthetic', type=str)

    return parser.parse_args()




if __name__ == '__main__':
    args = get_args()

    pipe = DiffusionPipeline.from_pretrained(
        args.model_ckpt_path,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")

    input_json = os.path.join('data/evolve_director/training_set_toy_1024.json')
    with open(input_json, 'r') as f:
        items = json.load(f)
    
    save_root = os.path.join('data/imgs')
    os.makedirs(save_root, exist_ok=True)


    if len(items) > 0:

        for item in items:

            prompt = item['prompt']
            file_name = os.path.basename(item['path'])
            save_path = os.path.join(save_root, file_name)

            if os.path.exists(save_path):
                continue

            if int(item["height"])==512 and int(item["width"])==512:               
                image = pipe(prompt=prompt, num_inference_steps=50, guidance_scale=3,
                        width=1024,
                        height=1024).images[0]
            else:
                image = pipe(prompt=prompt, num_inference_steps=50, guidance_scale=3,
                        width=item["width"],
                        height=item["height"]).images[0]

            image.save(save_path)


