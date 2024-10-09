import argparse
import torch
import os
import time
import json
import random
from tqdm import tqdm
import torch.distributed as dist
import shutil
import subprocess


def split_list(lst, num_chunks):
    avg_chunk_size = len(lst) // num_chunks
    remainder = len(lst) % num_chunks
    chunks = []
    start = 0
    for i in range(num_chunks):
        chunk_size = avg_chunk_size + (1 if i < remainder else 0)
        end = start + chunk_size
        chunks.append(lst[start:end])
        start = end
    return chunks


def check_processing_complete(flag_path):
    if os.path.exists(flag_path):
        with open(flag_path, "r") as f:
            return f.read().strip() == "Finished"
    return False


def save_to_json(items, output_file):
    with open(output_file, 'w') as f:
        json.dump(items, f, indent=4)


def load_json(json_path):
    with open(json_path, 'r') as f:
        items = json.load(f)
    return items


def main(args):
    TASKs = ['Playground']  # advanced models
    pros_numbers = args.pros_numbers

    singal_folder = os.path.join(args.sync_root, args.task_stamp)
    gen_singal_path = os.path.join(singal_folder, "gen_img_flag.txt")

    for TASK in TASKs:
        local_singal_path = os.path.join(args.sync_root, args.task_stamp, f'advanced_images_{TASK}')

        os.makedirs(local_singal_path, exist_ok=True)

        for it_index in range(pros_numbers):
            vlm_singal_sub_path = os.path.join(local_singal_path, f"gen_img_flag_{it_index}.txt")
            with open(vlm_singal_sub_path, "w") as f:
                f.write("Advanced Model Finished")

    while True:
        if not os.path.exists(gen_singal_path):
            with open(vlm_singal_sub_path, "w") as f:
                f.write("Initialized")

        with open(gen_singal_path, "r") as f:
            if f.read().strip() != "Advanced Model Start":
                time.sleep(10)
                continue

        print('Gen Advanced Img start')  # Start generate images with advanced models

        gen_folder_path = os.path.join(singal_folder, "gen_img_folder.txt")

        with open(gen_folder_path, 'r') as f:
            save_root_folder_withtime = f.readline().strip() 

        dynamic_json = os.path.join(save_root_folder_withtime, 'dynamic_subset.json')
        new_json_file = os.path.join(save_root_folder_withtime, "new_data.json")

        org_img_gen = False 
        if os.path.exists(new_json_file):
            print('Inference on new text prompts')
            with open(new_json_file, 'r') as f:
                selected_items = json.load(f)
        else:
            org_img_gen = True
            print('Inference on original text prompts')
            with open(dynamic_json, 'r') as f:
                selected_items = json.load(f)

        for TASK in TASKs:
            save_root = os.path.join(singal_folder, f'advanced_images_{TASK}')
            os.makedirs(save_root, exist_ok=True)

        filted_items = []
        if org_img_gen:
            for item in selected_items:
                file_name = os.path.basename(item['path'])
                for TASK in TASKs:
                    save_root = os.path.join(singal_folder, f'advanced_images_{TASK}')
                    file_path = os.path.join(save_root, file_name)
                    if not os.path.exists(file_path) and (item not in filted_items):
                        filted_items.append(item)
        else:
            filted_items = selected_items

        print(f'{len(filted_items)} images to generate')

        if len(filted_items) > 0:

            splited_items = split_list(filted_items, pros_numbers)
            for TASK in TASKs:
                local_singal_path = os.path.join(args.sync_root, args.task_stamp, f'advanced_images_{TASK}')
 
                for it_index, items_i in enumerate(splited_items):
                    dynamic_json_sub_path = os.path.join(local_singal_path, f'img_to_gen_{it_index}.json')
                    if len(TASKs) > 1 and (not org_img_gen):
                        items_i_filted = list(filter(lambda item: item.get("ad_model") == TASK, items_i))
                    else:
                        items_i_filted = items_i
                    save_to_json(items_i_filted, dynamic_json_sub_path)
                    
                    gen_singal_sub_path = os.path.join(local_singal_path, f"gen_img_flag_{it_index}.txt")
                    with open(gen_singal_sub_path, "w") as f:
                        f.write("Advanced Model Start")


            wait_flag_back = 0
            wait_flag = 0
            while wait_flag < pros_numbers*len(TASKs):
                wait_flag = 0
                for it_index in range(pros_numbers):
                    for TASK in TASKs:
                        local_singal_path = os.path.join(args.sync_root, args.task_stamp, f'advanced_images_{TASK}')
                        vlm_singal_sub_path = os.path.join(local_singal_path, f"gen_img_flag_{it_index}.txt")
                        with open(vlm_singal_sub_path, "r") as f:
                            if f.read().strip() == "Advanced Model Finished":
                                wait_flag += 1
                            else:
                                time.sleep(10)

                if wait_flag > wait_flag_back:
                    print(f'Finished Num: {wait_flag}/{pros_numbers*len(TASKs)}')
                    wait_flag_back = wait_flag
        

        with open(gen_singal_path, "w") as f:
            f.write("Advanced Model Finished")
        
        print('Gen Advanced Img Finished')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pros_numbers', type=int, default=1)
    parser.add_argument('--sync_root', default='./Synchronizer', type=str)
    parser.add_argument('--task_stamp', default='dynamic_0000', type=str)
    args = parser.parse_args()
    main(args)
