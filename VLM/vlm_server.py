import argparse
import torch
import os
import time
import json
import random
from tqdm import tqdm
import torch.distributed as dist
import shutil


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
    
    pros_numbers = args.pros_numbers

    singal_folder = os.path.join(args.sync_root, args.task_stamp)
    vlm_singal_path = os.path.join(singal_folder, "vlm_flag.txt")

    local_singal_path = os.path.join(singal_folder, 'llava')
    os.makedirs(local_singal_path, exist_ok=True)

    for it_index in range(pros_numbers):
        vlm_singal_sub_path = os.path.join(local_singal_path, f"vlm_flag_{it_index}.txt")
        with open(vlm_singal_sub_path, "w") as f:
            f.write("Initialized")

    while True:
        if os.path.exists(vlm_singal_path):
            with open(vlm_singal_path, "r") as f:
                if f.read().strip() != "Start":
                    time.sleep(10)
                    continue
        else:
            print('no vlm_singal_path')
            time.sleep(10)
            continue

        print('VLM comparsion start')

        gen_folder_path = os.path.join(singal_folder, "gen_img_folder.txt")
        with open(gen_folder_path, 'r') as f:
            save_root_folder_withtime = f.readline().strip() 

        save_root_folder_withtime = os.path.join(save_root_folder_withtime)
        dynamic_json = os.path.join(save_root_folder_withtime, 'dynamic_subset.json')
        selected_items = load_json(dynamic_json)

        splited_items = split_list(selected_items, pros_numbers)
        for it_index, items_i in enumerate(splited_items):
            dynamic_json_sub_path = os.path.join(local_singal_path, f'imgs_to_compare_{it_index}.json')
            save_to_json(items_i, dynamic_json_sub_path)
            
            vlm_singal_sub_path = os.path.join(local_singal_path, f"vlm_flag_{it_index}.txt")
            with open(vlm_singal_sub_path, "w") as f:
                f.write("Start")


        wait_flag = 0
        while wait_flag < pros_numbers:
            wait_flag = 0
            wait_flag_back = 0
            for it_index in range(pros_numbers):
                vlm_singal_sub_path = os.path.join(local_singal_path, f"vlm_flag_{it_index}.txt")
                with open(vlm_singal_sub_path, "r") as f:
                    if f.read().strip() == "Finished":
                        wait_flag += 1
                    else:
                        time.sleep(10)

            if wait_flag > wait_flag_back:
                print(f'Finished Num: {wait_flag}/{pros_numbers}')
                wait_flag_back = wait_flag
        

        new_items = []
        del_items = []
        for it_index in range(pros_numbers):
            new_json_file_sub = load_json(os.path.join(local_singal_path, f"new_data_{it_index}.json"))
            new_items.extend(new_json_file_sub)

            remove_json_file_sub = load_json(os.path.join(local_singal_path, f"rm_data_{it_index}.json"))
            del_items.extend(remove_json_file_sub)


        new_json_file = os.path.join(singal_folder, "new_data.json")
        remove_json_file = os.path.join(singal_folder, "rm_data.json")
        save_to_json(new_items, new_json_file)
        save_to_json(del_items, remove_json_file)
        print(f"Add new data num: {len(new_items)}")
        print(f"Remove data num: {len(del_items)}")
        new_json_file = os.path.join(save_root_folder_withtime, "new_data.json")
        remove_json_file = os.path.join(save_root_folder_withtime, "rm_data.json")
        save_to_json(new_items, new_json_file)
        save_to_json(del_items, remove_json_file)

        data = {
            "ad_better_count": 0,
            "new_better_count": 0,
            "all_count": 0,
            "ad_better_ratio": 0,
            "new_better_ratio": 0
        }
        for it_index in range(pros_numbers):
            count_data_i = load_json(os.path.join(local_singal_path, f"vlm_count_{it_index}.json"))
            data["ad_better_count"] += count_data_i["ad_better_count"]
            data["new_better_count"] += count_data_i["new_better_count"]
            data["all_count"] += count_data_i["all_count"]

        data["ad_better_ratio"] = round(data["ad_better_count"]  / data["all_count"] * 100, 2)
        data["new_better_ratio"] = round(data["new_better_count"] / data["all_count"] * 100, 2)

        vlm_count_path = os.path.join(singal_folder, "vlm_count.json")
        if os.path.exists(vlm_count_path):
            data_all = load_json(vlm_count_path)
        else:
            data_all = []
        data_all.append(data)
        save_to_json(data_all, vlm_count_path)


        with open(vlm_singal_path, "w") as f:
            f.write("Finished")
        
        print('VLM comparsion finished')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sync_root', default='./Synchronizer', type=str)
    parser.add_argument('--task_stamp', default='dynamic_0000', type=str)
    parser.add_argument('--pros_numbers', type=int, default=1)
    args = parser.parse_args()
    main(args)
