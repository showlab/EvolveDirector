# -*- coding: utf-8 -*-
import os
import re
import html
import urllib.parse as ul

import ftfy
import torch
from bs4 import BeautifulSoup
from transformers import T5EncoderModel, AutoTokenizer
from huggingface_hub import hf_hub_download

class T5Embedder:

    available_models = ['t5-v1_1-xxl']
    bad_punct_regex = re.compile(r'['+'#®•©™&@·º½¾¿¡§~'+'\)'+'\('+'\]'+'\['+'\}'+'\{'+'\|'+'\\'+'\/'+'\*' + r']{1,}')  # noqa

    def __init__(self, device, dir_or_name='t5-v1_1-xxl', *, local_cache=False, cache_dir=None, hf_token=None, use_text_preprocessing=True,
                 t5_model_kwargs=None, torch_dtype=None, use_offload_folder=None, model_max_length=120):
        self.device_new = torch.device(device)
        self.torch_dtype = torch_dtype or torch.bfloat16
        if t5_model_kwargs is None:
            t5_model_kwargs = {'low_cpu_mem_usage': True, 'torch_dtype': self.torch_dtype}
            if use_offload_folder is not None:
                t5_model_kwargs['offload_folder'] = use_offload_folder
                t5_model_kwargs['device_map'] = {
                    'shared': self.device_new,
                    'encoder.embed_tokens': self.device_new,
                    'encoder.block.0': self.device_new,
                    'encoder.block.1': self.device_new,
                    'encoder.block.2': self.device_new,
                    'encoder.block.3': self.device_new,
                    'encoder.block.4': self.device_new,
                    'encoder.block.5': self.device_new,
                    'encoder.block.6': self.device_new,
                    'encoder.block.7': self.device_new,
                    'encoder.block.8': self.device_new,
                    'encoder.block.9': self.device_new,
                    'encoder.block.10': self.device_new,
                    'encoder.block.11': self.device_new,
                    'encoder.block.12': 'disk',
                    'encoder.block.13': 'disk',
                    'encoder.block.14': 'disk',
                    'encoder.block.15': 'disk',
                    'encoder.block.16': 'disk',
                    'encoder.block.17': 'disk',
                    'encoder.block.18': 'disk',
                    'encoder.block.19': 'disk',
                    'encoder.block.20': 'disk',
                    'encoder.block.21': 'disk',
                    'encoder.block.22': 'disk',
                    'encoder.block.23': 'disk',
                    'encoder.final_layer_norm': 'disk',
                    'encoder.dropout': 'disk',
                }
            else:
                t5_model_kwargs['device_map'] = {'shared': self.device_new, 'encoder': self.device_new}

        self.use_text_preprocessing = use_text_preprocessing
        self.hf_token = hf_token
        self.cache_dir = cache_dir or os.path.expanduser('~/.cache/IF_')
        self.dir_or_name = dir_or_name
        tokenizer_path, path = dir_or_name, dir_or_name
        if local_cache:
            cache_dir = os.path.join(self.cache_dir, dir_or_name)
            tokenizer_path, path = cache_dir, cache_dir
        elif dir_or_name in self.available_models:
            cache_dir = os.path.join(self.cache_dir, dir_or_name)
            for filename in [
                'config.json', 'special_tokens_map.json', 'spiece.model', 'tokenizer_config.json',
                'pytorch_model.bin.index.json', 'pytorch_model-00001-of-00002.bin', 'pytorch_model-00002-of-00002.bin'
            ]:
                hf_hub_download(repo_id=f'DeepFloyd/{dir_or_name}', filename=filename, cache_dir=cache_dir,
                                force_filename=filename, token=self.hf_token)
            tokenizer_path, path = cache_dir, cache_dir
        else:
            cache_dir = os.path.join(self.cache_dir, 't5-v1_1-xxl')
            for filename in [
                'config.json', 'special_tokens_map.json', 'spiece.model', 'tokenizer_config.json',
            ]:
                hf_hub_download(repo_id='DeepFloyd/t5-v1_1-xxl', filename=filename, cache_dir=cache_dir,
                                force_filename=filename, token=self.hf_token)
            tokenizer_path = cache_dir

        print(tokenizer_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = T5EncoderModel.from_pretrained(path, **t5_model_kwargs).eval()
        self.model_max_length = model_max_length

    def get_text_embeddings(self, texts):
        texts = [self.text_preprocessing(text) for text in texts]

        text_tokens_and_mask = self.tokenizer(
            texts,
            max_length=self.model_max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        text_tokens_and_mask['input_ids'] = text_tokens_and_mask['input_ids']
        text_tokens_and_mask['attention_mask'] = text_tokens_and_mask['attention_mask']

        with torch.no_grad():
            text_encoder_embs = self.model(
                input_ids=text_tokens_and_mask['input_ids'].to(self.device_new),
                attention_mask=text_tokens_and_mask['attention_mask'].to(self.device_new),
            )['last_hidden_state'].detach()
        return text_encoder_embs, text_tokens_and_mask['attention_mask'].to(self.device_new)

    def text_preprocessing(self, text):
        if self.use_text_preprocessing:
            # The exact text cleaning as was in the training stage:
            text = self.clean_caption(text)
            text = self.clean_caption(text)
            return text
        else:
            return text.lower().strip()

    @staticmethod
    def basic_clean(text):
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()

    # Clean caption
    def clean_caption(self, caption_to_clean):
        caption_to_clean = str(caption_to_clean)
        caption_to_clean = ul.unquote_plus(caption_to_clean)
        caption_to_clean = caption_to_clean.strip().lower()
        caption_to_clean = re.sub('<person>', 'person', caption_to_clean)
        # urls:
        caption_to_clean = re.sub(
            r'\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))',  # noqa
            '', caption_to_clean)  # regex for urls
        caption_to_clean = re.sub(
            r'\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))',  # noqa
            '', caption_to_clean)  # regex for urls
        # html:
        caption_to_clean = BeautifulSoup(caption_to_clean, features='html.parser').text

        caption_to_clean = re.sub(r'@[\w\d]+\b', '', caption_to_clean)

        caption_to_clean = re.sub(r'[\u31c0-\u31ef]+', '', caption_to_clean)
        caption_to_clean = re.sub(r'[\u31f0-\u31ff]+', '', caption_to_clean)
        caption_to_clean = re.sub(r'[\u3200-\u32ff]+', '', caption_to_clean)
        caption_to_clean = re.sub(r'[\u3300-\u33ff]+', '', caption_to_clean)
        caption_to_clean = re.sub(r'[\u3400-\u4dbf]+', '', caption_to_clean)
        caption_to_clean = re.sub(r'[\u4dc0-\u4dff]+', '', caption_to_clean)
        caption_to_clean = re.sub(r'[\u4e00-\u9fff]+', '', caption_to_clean)
     
        caption_to_clean = re.sub(
            r'[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+',  # noqa
            '-', caption_to_clean)
   
        caption_to_clean = re.sub(r'[`´«»“”¨]', '"', caption_to_clean)
        caption_to_clean = re.sub(r'[‘’]', "'", caption_to_clean)
     
        caption_to_clean = re.sub(r'&quot;?', '', caption_to_clean)
       
        caption_to_clean = re.sub(r'&amp', '', caption_to_clean)

        caption_to_clean = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' ', caption_to_clean)

        caption_to_clean = re.sub(r'\d:\d\d\s+$', '', caption_to_clean)
        
        caption_to_clean = re.sub(r'\\n', ' ', caption_to_clean)

        caption_to_clean = re.sub(r'#\d{1,3}\b', '', caption_to_clean)
       
        caption_to_clean = re.sub(r'#\d{5,}\b', '', caption_to_clean)
        
        caption_to_clean = re.sub(r'\b\d{6,}\b', '', caption_to_clean)
        
        caption_to_clean = re.sub(r'[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)', '', caption_to_clean)
       
        caption_to_clean = re.sub(r'[\"\']{2,}', r'"', caption_to_clean)  # """AUSVERKAUFT"""
        caption_to_clean = re.sub(r'[\.]{2,}', r' ', caption_to_clean)  # """AUSVERKAUFT"""

        caption_to_clean = re.sub(self.bad_punct_regex, r' ', caption_to_clean)  # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption_to_clean = re.sub(r'\s+\.\s+', r' ', caption_to_clean)  # " . "

        regex2 = re.compile(r'(?:\-|\_)')
        if len(re.findall(regex2, caption_to_clean)) > 3:
            caption_to_clean = re.sub(regex2, ' ', caption_to_clean)

        caption_to_clean = self.basic_clean(caption_to_clean)

        caption_to_clean = re.sub(r'\b[a-zA-Z]{1,3}\d{3,15}\b', '', caption_to_clean)  # jc6640
        caption_to_clean = re.sub(r'\b[a-zA-Z]+\d+[a-zA-Z]+\b', '', caption_to_clean)  # jc6640vc
        caption_to_clean = re.sub(r'\b\d+[a-zA-Z]+\d+\b', '', caption_to_clean)  # 6640vc231

        caption_to_clean = re.sub(r'(worldwide\s+)?(free\s+)?shipping', '', caption_to_clean)
        caption_to_clean = re.sub(r'(free\s)?download(\sfree)?', '', caption_to_clean)
        caption_to_clean = re.sub(r'\bclick\b\s(?:for|on)\s\w+', '', caption_to_clean)
        caption_to_clean = re.sub(r'\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?', '', caption_to_clean)
        caption_to_clean = re.sub(r'\bpage\s+\d+\b', '', caption_to_clean)

        caption_to_clean = re.sub(r'\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b', r' ', caption_to_clean)  # j2d1a2a...

        caption_to_clean = re.sub(r'\b\d+\.?\d*[xх×]\d+\.?\d*\b', '', caption_to_clean)

        caption_to_clean = re.sub(r'\b\s+\:\s+', r': ', caption_to_clean)
        caption_to_clean = re.sub(r'(\D[,\./])\b', r'\1 ', caption_to_clean)
        caption_to_clean = re.sub(r'\s+', ' ', caption_to_clean)

        caption_to_clean.strip()

        caption_to_clean = re.sub(r'^[\"\']([\w\W]+)[\"\']$', r'\1', caption_to_clean)
        caption_to_clean = re.sub(r'^[\'\_,\-\:;]', r'', caption_to_clean)
        caption_to_clean = re.sub(r'[\'\_,\-\:\-\+]$', r'', caption_to_clean)
        caption_to_clean = re.sub(r'^\.\S+$', '', caption_to_clean)

        return caption_to_clean.strip()
