import argparse
import os
import yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
from ruamel.yaml import YAML
import torch
import gc
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import openai
from gen_model.blip import blip_decoder
from data import create_dataset, create_sampler, create_loader
import utils
from openai.error import APIConnectionError
import matplotlib
matplotlib.use('Agg')
import dashscope

def gen_text(generating_model, pre_loader, gen_config):

    device = "cuda"
    male_keywords = ["male", "man", "boy", "he", "guy", "him", "his"]
    female_keywords = ["female", "woman", "girl", "she", "lady", "her", "hers"]

    data_list = []
    for n_iter, (image, captions, image_path, ids) in enumerate(pre_loader):

        if n_iter<150:
            continue
        print(n_iter)

        image = image.to(device, non_blocking=True)

        gender_ids = []

        with torch.no_grad():
            gen_base_captions = generating_model.generate(gender_ids, image, sample=True, num_beams=gen_config['num_beams'],
                                                 max_length=gen_config['max_length'],
                                                 min_length=gen_config['min_length'])
        gender_ids = []

        for t, caption in enumerate(gen_base_captions):

            if any(keyword in caption for keyword in female_keywords):
                gender_id = 1
            elif any(keyword in caption for keyword in male_keywords):
                gender_id = 0
            else:
                gender_id = 2
            gender_ids.append(gender_id)

        with torch.no_grad():
            gen_captions = generating_model.generate2(gender_ids, image, sample=True, num_beams=gen_config['num_beams'],
                                                 max_length=gen_config['max_length'],
                                                 min_length=gen_config['min_length'])

        for t, gen_caption in enumerate(gen_captions):

            gpt_promt = f"Extract the attributes of this text:{gen_caption},divided into upper attribute vocabulary and lower attribute vocabulary, and output them according to a fixed format, e.g., Upper: black long-sleeved sweater black backpack Lower: blue jeans red. " \
                        f"Use only the output results as required."

            for attempt in range(3):

                try:

                    messages = [
                        {'role': 'system', 'content': 'You are a helpful assistant.'},
                        {'role': 'user', 'content': gpt_promt}
                    ]
                    response1 = dashscope.Generation.call(
                        # If no environment variables are configured, replace the following line with the Baolian API Key: api_key="sk-xxx",
                        api_key="sk-dd88f2a221ea43e49799d743f92c209f",
                        model="qwen-plus",
                        messages=messages,
                        result_format='message'
                    )
                    output = response1.output.choices[0].message.content

                    break
                except APIConnectionError as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(2)
            else:
                print("Multiple attempts to connect to OpenAI failed.")
                break

            data_list.append({
                "image_path": image_path[t],
                "gen_base_caption": gen_base_captions[t],
                "gen_caption": gen_caption,
                "caption": captions[t],
                "gpt_gen_caption": output,
                "id": ids[t].item()
            })

        # with open(f"gen_data2/gen_data_{n_iter}.json", "w", encoding="utf-8") as f:
        #     json.dump(data_list, f, ensure_ascii=False, indent=4)
        # # with open(f"gen_data31/gen_data_{n_iter}.json", "w", encoding="utf-8") as f:
        # #     json.dump(data_list1, f, ensure_ascii=False, indent=4)
        # data_list = []




    return data_list


def create_and_load_datasets(config):

    cuhk_pre_dataset, rstp_pre_dataset, icfg_pre_dataset = create_dataset(config['gen_dataset'], config)
    sampler = [None, None, None]
    cuhk_pre_loader, rstp_pre_loader, icfg_pre_loader = create_loader(
        [cuhk_pre_dataset, rstp_pre_dataset, icfg_pre_dataset ], sampler,
        batch_size=[100, 100, 100],
        num_workers=[4, 4, 4],
        is_trains=[False, False, False],
        collate_fns=[None, None, None]
    )

    del cuhk_pre_dataset, rstp_pre_dataset, icfg_pre_dataset
    gc.collect()
    torch.cuda.empty_cache()

    return cuhk_pre_loader, rstp_pre_loader, icfg_pre_loader


def main(config):

    device = 'cuda'
    img_size = config['image_size']
    generating_model = blip_decoder(pretrained=config['pretrained'], image_size=img_size, vit=config['vit'],
                                    vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'],
                                    prompt=config['prompt'])
    generating_model = generating_model.to(device)

    cuhk_pre_loader, rstp_pre_loader, icfg_pre_loader = create_and_load_datasets(config)

    model_save_path = './checkpoint/cuhk_best_model_epoch.pth'
    generating_model.load_state_dict(torch.load(model_save_path))
    cuhk_gen_caption = gen_text(generating_model, cuhk_pre_loader, config)

    with open(f"./gen_data/cuhk_gen_caption.json", "w", encoding="utf-8") as f:
        json.dump(cuhk_gen_caption, f, ensure_ascii=False, indent=4)

    model_save_path = './checkpoint/icfg_best_model_epoch.pth'
    generating_model.load_state_dict(torch.load(model_save_path))

    icfg_gen_caption = gen_text(generating_model, icfg_pre_loader, config)
    with open(f"./gen_data/icfg_gen_caption.json", "w", encoding="utf-8") as f:
        json.dump(icfg_gen_caption, f, ensure_ascii=False, indent=4)

    model_save_path = './checkpoint/rstp_best_model_epoch.pth'
    generating_model.load_state_dict(torch.load(model_save_path))

    rstp_gen_caption = gen_text(generating_model, rstp_pre_loader, config)
    with open(f"./gen_data/rstp_gen_caption.json", "w", encoding="utf-8") as f:
        json.dump(rstp_gen_caption, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', default='./configs/caption.yaml')

    args = parser.parse_args()

    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    yaml = YAML(typ='safe')

    with open(args.configs, 'r') as file:
        configs = yaml.load(file)

    main(configs)