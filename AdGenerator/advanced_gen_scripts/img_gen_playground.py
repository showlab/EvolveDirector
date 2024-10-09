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
    parser.add_argument('--sync_root', default='./Synchronizer', type=str)
    parser.add_argument('--task_stamp', default='dynamic_0000', type=str)
    parser.add_argument('--rank', type=int, default=-1)

    return parser.parse_args()




if __name__ == '__main__':
    args = get_args()
    it_index = args.rank
    TASK = 'Playground'

    pipe = DiffusionPipeline.from_pretrained(
        args.model_ckpt_path,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")

    # data setting
    singal_folder = os.path.join(args.sync_root, args.task_stamp)

    local_singal_path = os.path.join(args.sync_root, args.task_stamp, f"advanced_images_{TASK}")

    gen_singal_path = os.path.join(local_singal_path, f"gen_img_flag_{it_index}.txt")

    while True:
        if os.path.exists(gen_singal_path):
            try:
                with open(gen_singal_path, "r") as f:
                    if f.read().strip() != f"Advanced Model Start":
                        time.sleep(10)
                        continue
            except FileNotFoundError:
                continue
        else:
            time.sleep(10)
            continue

        dynamic_json_sub_path = os.path.join(local_singal_path, f'img_to_gen_{it_index}.json')
        with open(dynamic_json_sub_path, 'r') as f:
            selected_items = json.load(f)
        
        filted_items = selected_items
        save_root = os.path.join(singal_folder, f'advanced_images_{TASK}', 'imgs')
        os.makedirs(save_root, exist_ok=True)


        if len(filted_items) > 0:

            for item in filted_items:

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

        
        print(f'{TASK} gen img finished')
        with open(gen_singal_path, "w") as f:
            f.write("Advanced Model Finished")

