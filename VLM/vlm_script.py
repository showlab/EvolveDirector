import argparse
import torch
import sys
sys.path.append("./")
import warnings
warnings.filterwarnings("ignore")  # ignore warning
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from diffusion.data.datasets import ASPECT_RATIO_1024_Edgen

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
from PIL import Image, ImageDraw, ImageFont
import copy
import os
import time
import json
import random
from tqdm import tqdm
import shutil
import time


def save_to_json(items, output_file):
    while True:
        try:
            with open(output_file, 'w') as f:
                json.dump(items, f, indent=4)
            break
        except TypeError:
            os.remove(output_file)
            print("save jason with TypeError")
            with open(output_file, 'w') as f:
                json.dump([], f, indent=4)
            break


def combine_images(image1_path, image2_path, combined_path, save_img=False):
    # open two images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # make sure two images have same height
    min_height = min(image1.height, image2.height)
    min_width = min(image1.width, image2.width)
    if min_height>512:
        min_width = int(min_width / min_height * 512)
        min_height = 512
    if min_width>512:
        min_height = int(min_height / min_width * 512)
        min_width = 512
    
    if image1.width != min_width or image1.height != min_height:
        image1 = image1.resize((min_width, min_height))

    if image2.width != min_width or image2.height != min_height:
        image2 = image2.resize((min_width, min_height))

    # create white image template
    border_width = min_height//50  # width of the border
    combined_width = image1.width + image2.width + border_width * 3
    combined_height = min_height + border_width * 10
    combined_image = Image.new('RGB', (combined_width, combined_height), color='white')

    # paste image1 and image2 on the white image
    combined_image.paste(image1, (border_width, border_width))
    combined_image.paste(image2, (image1.width + border_width * 2, border_width))

    # add text
    draw = ImageDraw.Draw(combined_image)
    font = ImageFont.load_default()
    text_size = min_height//10
    font = font.font_variant(size=text_size)  # scale of text
    text_a = "(A)"
    text_b = "(B)"
    text_width_a = text_size * len(text_a)
    text_width_b = text_size * len(text_b)
    text_height = text_size
    draw.text((border_width + image1.width//2 - text_width_a//4, min_height + border_width*2), text_a, fill="black", font=font)
    draw.text((image1.width + border_width * 2 + image2.width//2 - text_width_b//4, min_height + border_width*2), text_b, fill="black", font=font)

    if save_img:
        combined_image.save(combined_path)

    return combined_image

def vlm_exe(conv, model, tokenizer, image_tensor, image_size, args):
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            use_cache=True)

    outputs = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    if args.debug:
        print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
    return outputs


def main(args):

    TASKs = ['Playground']  # advanced models

    it_index = args.rank
    # Model
    disable_torch_init()

    extend_num = args.extend_num
    print(f"vlm inference with extend_num: {extend_num}")
    mutation_ratio = args.mutation_ratio
    print(f"vlm inference with mutation_ratio: {mutation_ratio}")

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
    
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode


    singal_folder = os.path.join(args.sync_root, args.task_stamp)
    local_singal_path = os.path.join(singal_folder, 'llava')
    vlm_singal_path = os.path.join(local_singal_path, f"vlm_flag_{it_index}.txt")

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

        gen_folder_path = os.path.join(singal_folder, "gen_img_folder.txt")
        with open(gen_folder_path, 'r') as f:
            save_root_folder_withtime = f.readline().strip() 

        dynamic_json = os.path.join(local_singal_path, f'imgs_to_compare_{it_index}.json')
        with open(dynamic_json, 'r') as f:
            selected_items = json.load(f)

        ad_better_count = 0
        new_better_count = 0
        all_count = len(selected_items)

        del_items = []
        new_items = []

        combined_folder = os.path.join(save_root_folder_withtime, 'combined')
        os.makedirs(combined_folder, exist_ok=True)

        if all_count > 0:
            for item in tqdm(selected_items):

                conv = conv_templates[args.conv_mode].copy()
                if "mpt" in model_name.lower():
                    roles = ('user', 'assistant')
                else:
                    roles = conv.roles

                file_name = os.path.basename(item['path'])

                # prepare text prompts
                text_prompt = item['prompt']
                
                quary_dis = f'In these two images (A) and (B), which one aligns better with the text description "{text_prompt}"? You have two options: <(A) is better>, <(B) is better>. Simply state your choice, no need for a detailed explanation.'                    
                quary_expan = f'Replace the nouns in the text description: "{text_prompt}" with other kinds of objects, characters, backgrounds, names, colors, or styles to generate {extend_num} more diverse text descriptions. Arrange them in the format of a list ["Text description 1", "Text description 2", ...].'
                enhanced_prompts =  ['It should be rough and short.', 'It should contain less than 30 words and be highly detailed.', 'It should contain over 30 words with different granular levels of detail.', 'It should contain over 50 words with a lot of details.']                   
                quary_mut = f'Now, exercise your imagination to generate one new text description for visual content that is completely unrelated to the previous images. It should have a completely different structure from the previous descriptions. {random.choice(enhanced_prompts)} Arrange it in the format of a list just like ["xxxxx"].'                    

                if len(TASKs) > 1:
                    better_ad_index = 0
                    for task_index in range(len(TASKs) - 1):
                        ad_model_1 = TASKs[better_ad_index]
                        ad_model_2 = TASKs[task_index+1]
                        if random.random() < 0.5:
                            img_1 = os.path.join(singal_folder, f'advanced_images_{ad_model_1}', 'imgs', file_name)
                            img_2 = os.path.join(singal_folder, f'advanced_images_{ad_model_2}','imgs', file_name)
                            flag = 'A_ad1_B_ad2'
                        else:
                            img_1 = os.path.join(singal_folder, f'advanced_images_{ad_model_2}','imgs', file_name)
                            img_2 = os.path.join(singal_folder, f'advanced_images_{ad_model_1}', 'imgs', file_name)
                            flag = 'A_ad2_B_ad1'
                        combined_path = os.path.join(combined_folder, file_name)

                        if os.path.exists(img_1) and os.path.exists(img_2):
                            image = combine_images(img_1, img_2, combined_path)
                        else:
                            print('file does not exists')
                            if os.path.exists(os.path.join(singal_folder, f'advanced_images_{ad_model_2}', file_name)):
                                better_ad_index = task_index+1
                            continue
                            
                        image_size = image.size
                        image_tensor = process_images([image], image_processor, model.config)
                        if type(image_tensor) is list:
                            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
                        else:
                            image_tensor = image_tensor.to(model.device, dtype=torch.float16)

                        # discrimination
                        max_re_try_num = 5
                        for re_try_num in range(max_re_try_num):
                            inp = quary_dis
                            if model.config.mm_use_im_start_end:
                                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                            else:
                                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

                            conv_dis = copy.deepcopy(conv)
                            conv_dis.append_message(conv_dis.roles[0], inp)
                            conv_dis.append_message(conv_dis.roles[1], None)
                            outputs = vlm_exe(conv_dis, model, tokenizer, image_tensor, image_size, args)
                            conv_dis.messages[-1][-1] = outputs

                            if (flag == 'A_ad1_B_ad2' and '<(B) is better>' in outputs) or (flag == 'A_ad2_B_ad1' and '<(A) is better>' in outputs):
                                better_ad_index = task_index+1
                                break
                            elif (flag == 'A_ad2_B_ad1' and '<(B) is better>' in outputs) or (flag == 'A_ad1_B_ad2' and '<(A) is better>' in outputs):
                                break
                            else:
                                continue
                            print('reached max retry num, break')
                    ad_model = TASKs[better_ad_index]
                else:
                    ad_model = TASKs[0]

                # prepare image
                if random.random() < 0.5:
                    img_1 = os.path.join(save_root_folder_withtime, 'new', file_name)
                    img_2 = os.path.join(singal_folder, f'advanced_images_{ad_model}','imgs', file_name)
                    flag = 'A_new_B_ad'
                else:
                    img_1 = os.path.join(singal_folder, f'advanced_images_{ad_model}', 'imgs', file_name)
                    img_2 = os.path.join(save_root_folder_withtime, 'new', file_name)
                    flag = 'A_ad_B_new'
                combined_path = os.path.join(combined_folder, file_name)

                if os.path.exists(img_1) and os.path.exists(img_2):
                    image = combine_images(img_1, img_2, combined_path, save_img=True)
                else:
                    print('file does not exists')
                    continue
                image_size = image.size
                image_tensor = process_images([image], image_processor, model.config)
                if type(image_tensor) is list:
                    image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
                else:
                    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

                # discrimination
                max_re_try_num = 5
                for re_try_num in range(max_re_try_num):
                    inp = quary_dis
                    if model.config.mm_use_im_start_end:
                        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                    else:
                        inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                    conv_dis = copy.deepcopy(conv)
                    conv_dis.append_message(conv_dis.roles[0], inp)
                    conv_dis.append_message(conv_dis.roles[1], None)
                    outputs = vlm_exe(conv_dis, model, tokenizer, image_tensor, image_size, args)
                    conv_dis.messages[-1][-1] = outputs
                    
                    expen_flag = False
                    mut_flag = False

                    if (flag == 'A_new_B_ad' and '<(B) is better>' in outputs) or (flag == 'A_ad_B_new' and '<(A) is better>' in outputs):
                        expen_flag = True
                        ad_better_count += 1
                        break
                    elif (flag == 'A_ad_B_new' and '<(B) is better>' in outputs) or (flag == 'A_new_B_ad' and '<(A) is better>' in outputs):
                        new_better_count += 1
                        del_items.append(item)  # deletion
                        if random.random() < mutation_ratio:
                            mut_flag = True
                        break
                    else:
                        continue
                    
                    print('reached max retry num, break')
                
                conv = conv_dis

                # expansion
                if expen_flag:
                    for re_try_num in range(max_re_try_num):
                        inp = quary_expan
                        conv_expan = copy.deepcopy(conv)
                        conv_expan.append_message(conv_expan.roles[0], inp)
                        conv_expan.append_message(conv_expan.roles[1], None)
                        outputs = vlm_exe(conv_expan, model, tokenizer, image_tensor, image_size, args)
                        conv_expan.messages[-1][-1] = outputs
                        
                        try:
                            text_prompts = eval(outputs)
                        except SyntaxError:
                            try:
                                text_prompts = eval(outputs.replace('" ', '\\" ').replace(' "',' \\"'))
                            except SyntaxError:
                                continue
                        
                        if len(text_prompts) != extend_num:
                            continue
                        elif text_prompts[0] == text_prompts[1] or text_prompts[1] == text_prompts[2] or text_prompts[0] == text_prompts[2]:
                            continue
                        else:
                            timestamp = time.strftime("%m%d%H%M", time.localtime())
                            for text_index, text_pro_i in enumerate(text_prompts):
                                item_new = copy.deepcopy(item)
                                item_new['prompt'] = text_pro_i
                                item_new['path'] = item_new['path'].replace('.png', f'_T{timestamp}_{text_index}.png').replace('.jpg', f'_T{timestamp}_{text_index}.jpg')
                                item_new["ad_model"] = ad_model
                                if args.multi_scale:
                                    random_ratio = random.choice(list(ASPECT_RATIO_1024_Edgen.keys()))
                                    random_h_w = ASPECT_RATIO_1024_Edgen[random_ratio]
                                    item_new["ratio"] = random_ratio
                                    item_new["height"] = int(random_h_w[0])
                                    item_new["width"] = int(random_h_w[1])
                                new_items.append(item_new)
                            break
                            
                if mut_flag:
                    for re_try_num in range(max_re_try_num):
                        inp = quary_mut
                        conv_mut = copy.deepcopy(conv)
                        conv_mut.append_message(conv_mut.roles[0], inp)
                        conv_mut.append_message(conv_mut.roles[1], None)
                        outputs = vlm_exe(conv_mut, model, tokenizer, image_tensor, image_size, args)
                        conv_mut.messages[-1][-1] = outputs
                        try:
                            text_prompts = eval(outputs)
                        except SyntaxError:
                            try:
                                text_prompts = eval(outputs.replace('" ', '\\" ').replace(' "',' \\"'))
                            except SyntaxError:
                                continue
                        if len(text_prompts) == 0:
                            continue
                        text_pro_i = text_prompts[0]
                        item_new = copy.deepcopy(item)
                        item_new['prompt'] = text_pro_i
                        timestamp = time.strftime("%m%d%H%M", time.localtime())
                        item_new['path'] = item_new['path'].replace('.png', f'_T{timestamp}_m.png').replace('.jpg', f'_T{timestamp}_m.jpg')
                        item_new["ad_model"] = ad_model
                        new_items.append(item_new)
                        break
    
            
        new_json_file = os.path.join(local_singal_path, f"new_data_{it_index}.json")
        remove_json_file = os.path.join(local_singal_path, f"rm_data_{it_index}.json")
        save_to_json(new_items, new_json_file)
        save_to_json(del_items, remove_json_file)
        print(f"Add new data num: {len(new_items)}")
        print(f"Remove data num: {len(del_items)}")

        data = {
            "ad_better_count": ad_better_count,
            "new_better_count": new_better_count,
            "all_count": all_count
        }

        save_to_json(data, os.path.join(local_singal_path, f"vlm_count_{it_index}.json"))


        with open(vlm_singal_path, "w") as f:
            f.write("Finished")
        
        print('VLM compare finished')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--sync_root', default='./Synchronizer', type=str)
    parser.add_argument('--task_stamp', default='dynamic_0000', type=str)

    parser.add_argument("--model-path", type=str, default="/path/to/llava-v1.6-34b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", default=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--multi_scale', action="store_true")

    parser.add_argument('--extend_num', type=int, default=3)
    parser.add_argument('--mutation_ratio', type=float, default=0.1)
    args = parser.parse_args()
    main(args)
