import os
import time
import shutil
import json
import argparse
from datetime import datetime


def check_processing_complete(flag_path):
    if os.path.exists(flag_path):
        with open(flag_path, "r") as f:
            return f.read().strip() == "Finished"
    return False


def main(args):
    
    singal_folder = os.path.join(args.sync_root, args.task_stamp)

    epoch_info_path = os.path.join(singal_folder, "00_epoch_info.txt")

    dynamic_json_path = os.path.join(singal_folder, "dynamic_set.json")

    os.makedirs(singal_folder, exist_ok=True)

    # Initialize single files
    server_singal_path = os.path.join(singal_folder, "00_server_flag.txt") 
    with open(server_singal_path, "w") as f:  
        f.write("Initialized")
    
    gen_singal_path = os.path.join(singal_folder, "gen_img_flag.txt")
    with open(gen_singal_path, "w") as f:
        f.write("Initialized")

    vlm_singal_path = os.path.join(singal_folder, "vlm_flag.txt")
    with open(vlm_singal_path, "w") as f:
        f.write("Initialized")
            
    ex_fea_singal_path = os.path.join(singal_folder, "extract_fea_flag.txt")
    with open(ex_fea_singal_path, "w") as f:
        f.write("Initialized")

    while True:
        if os.path.exists(server_singal_path):
            with open(server_singal_path, "r") as f:
                if f.read().strip() != "Start":
                    time.sleep(10)
                    continue
        else:
            time.sleep(10)
            continue
    
        with open(server_singal_path, "w") as f:
            f.write("Ongoing")

        with open(epoch_info_path, "r") as f:
            epoch_info = f.read() 

        # 01: generate images for comparsion
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S ")
        info = '01: Generate Images with trained model STARTED'
        print(formatted_time+epoch_info+info)

        with open(gen_singal_path, "w") as f:
            f.write("Start")

        img_gen_finished = False
        while not img_gen_finished:
            time.sleep(10)
            img_gen_finished = check_processing_complete(gen_singal_path)

        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S ")
        info = '01: Generate Images with trained model FINISHED'
        print(formatted_time+epoch_info+info)

        # 02: vlm compare
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S ")
        info = '02: VLM Discriminator STARTED'
        print(formatted_time + epoch_info + info)
        vlm_singal_path = os.path.join(singal_folder, "vlm_flag.txt")
        with open(vlm_singal_path, "w") as f:
            f.write("Start")

        vlm_gen_finished = False
        while not vlm_gen_finished:
            time.sleep(10)
            vlm_gen_finished = check_processing_complete(vlm_singal_path)

        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S ")
        info = '02: VLM Discriminator FINISHED'
        print(formatted_time + epoch_info + info)

        # 03: create new training samples and extract features
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S ")
        info = '03: Create new samples and extract features STARTED'
        print(formatted_time + epoch_info + info)

        with open(gen_singal_path, "w") as f:
            f.write("Advanced Model Start")

        img_gen_finished = False
        while not img_gen_finished:
            time.sleep(10)
            with open(gen_singal_path, "r") as f:
                if f.read().strip() == "Advanced Model Finished":
                    img_gen_finished = True

        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S ")
        info = '03: Create new samples FINISHED, and extract features STARTED'
        print(formatted_time + epoch_info + info)

        with open(ex_fea_singal_path, "w") as f:
            f.write("Start")

        extract_fea_finished = False
        while not extract_fea_finished:
            time.sleep(10)
            extract_fea_finished = check_processing_complete(ex_fea_singal_path)

        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S ")

        info = '03: Create new samples and extract features FINISHED'
        print(formatted_time + epoch_info + info)

        # 04: update training data
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S ")
        info = '04: Update training data STARTED'
        print(formatted_time + epoch_info + info)

        new_json_file = os.path.join(singal_folder, "new_data.json")
        remove_json_file = os.path.join(singal_folder, "rm_data.json")


        with open(dynamic_json_path, 'r') as f:
            items_org = json.load(f)

            print(epoch_info+ f"Old data num {len(items_org)}")


        with open(new_json_file, 'r') as f:
            items_new = json.load(f)
            print(epoch_info + f"Add new data {len(items_new)}")

        with open(remove_json_file, 'r') as f:
            items_rm = json.load(f)
            print(epoch_info + f"Remove data {len(items_rm)}")

        items_org += items_new
        items_org = [item for item in items_org if item not in items_rm]
        with open(dynamic_json_path, 'w') as f:
            json.dump(items_org, f, indent=4)
        
        print(epoch_info+ f"Updated data num {len(items_org)}")
        
        with open(server_singal_path, "w") as f:
            f.write("Update")

        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S ")
        info = '04: Update training data FINISHED'
        print(formatted_time + epoch_info + info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sync_root', default='./Synchronizer', type=str)
    parser.add_argument('--task_stamp', default='dynamic_0000', type=str)
    args = parser.parse_args()
    main(args)
