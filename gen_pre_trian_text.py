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

import os
import dashscope

def gen_text( pre_loader):


    male_keywords = ["male", "man", "boy", "he", "guy", "him", "his"]
    female_keywords = ["female", "woman", "girl", "she", "lady", "her", "hers"]

    data_list = []

    for n_iter, (_, captions, image_paths, ids) in enumerate(pre_loader):

        print(n_iter)
        promts ="Please generate a description of the person’s clothing and appearance based on the image, first determining their gender and then starting with 'The man is wearing' or 'The woman is wearing' without adding subjective evaluation."

        gender_promts = [
            [
                "Please generate a description starting with 'He is wearing', focusing only on his clothing and appearance, without mentioning the surroundings.",
                "Please generate a description starting with 'The man wears', focusing only on his clothing and appearance, without mentioning the surroundings.",
                "Please generate a description starting with 'A man wears', focusing only on his clothing and appearance, without mentioning the surroundings.",
                "Please generate a description starting with 'The boy has', focusing only on his clothing and appearance, without mentioning the surroundings.",
                "Please generate a description starting with 'A boy talking on', focusing only on his clothing and appearance, without mentioning the surroundings.",
            ],
            [
                "Please generate a description starting with 'She is wearing', focusing only on his clothing and appearance, without mentioning the surroundings.",
                "Please generate a description starting with 'The woman wears', focusing only on his clothing and appearance, without mentioning the surroundings.",
                "Please generate a description starting with 'A woman wears', focusing only on his clothing and appearance, without mentioning the surroundings.",
                "Please generate a description starting with 'The girl has', focusing only on his clothing and appearance, without mentioning the surroundings.",
                "Please generate a description starting with 'A girl talking on', focusing only on his clothing and appearance, without mentioning the surroundings.",

            ],
            [
                "Please generate a description starting with 'The person wears', focusing only on his clothing and appearance, without mentioning the surroundings.",
                "Please generate a description starting with 'This person is wearing', focusing only on his clothing and appearance, without mentioning the surroundings.",
                "Please generate a description starting with 'The man wearing', focusing only on his clothing and appearance, without mentioning the surroundings.",
                "Please generate a description starting with 'A person in', focusing only on his clothing and appearance, without mentioning the surroundings.",
                "Please generate a description starting with 'The person has', focusing only on his clothing and appearance, without mentioning the surroundings.",

            ]
        ]

        m = 0
        w = 0
        o = 0

        for s, img_path in enumerate(image_paths):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"image": img_path},
                        {"text": promts}
                    ]
                }
            ]
            response = dashscope.MultiModalConversation.call(
                api_key="sk-dd88f2a221ea43e49799d743f92c209f",
                model='qwen-vl-max',
                messages=messages
            )
            if response.output is None:
                print(1)
            else:
                gen_base_caption = response.output.choices[0].message.content[0]["text"]
                # print(gen_base_caption)
                length = len(gen_base_caption)
                if length > 50:

                    qwen_text =f"Simplify the text given below to make the description simpler and more straightforward, removing any contextual description, any subjective description and possible depictions, focusing only on appearance, and outputting the final description without changing the text style." \
                               f"{gen_base_caption}"
                    messages1 = [
                        {'role': 'system', 'content': 'You are a helpful assistant.'},
                        {'role': 'user', 'content': qwen_text}
                    ]
                    response1 = dashscope.Generation.call(
                        api_key="sk-dd88f2a221ea43e49799d743f92c209f",
                        model="qwen-plus",
                        messages=messages1,
                        result_format='message'
                    )
                    gen_base_caption1 = response1.output.choices[0].message.content
                else:
                    gen_base_caption1 = gen_base_caption


                if any(keyword in gen_base_caption for keyword in female_keywords):
                    gender_promt = gender_promts[1][w % 5]
                    w += 1
                elif any(keyword in gen_base_caption for keyword in male_keywords):
                    gender_promt = gender_promts[0][m % 5]
                    m += 1
                else:
                    gender_promt = gender_promts[2][o % 5]
                    o += 1
                messages2 = [
                    {
                        "role": "user",
                        "content": [
                            {"image": img_path},
                            {"text": gender_promt}
                        ]
                    }
                ]

                response2 = dashscope.MultiModalConversation.call(
                    api_key="sk-dd88f2a221ea43e49799d743f92c209f",
                    model='qwen-vl-max',
                    messages=messages2
                )
                if response2.output is None:
                    print(1)
                else:
                    gen_caption = response2.output.choices[0].message.content[0]["text"]

                    length = len(gen_caption)
                    if length > 50:
                        qwen_text = f"Simplify the text given below to make the description simpler and more straightforward, removing any contextual description, any subjective description and possible depictions, focusing only on appearance, and outputting the final description without changing the text style." \
                                    f"{gen_caption}"
                        messages3 = [
                            {'role': 'system', 'content': 'You are a helpful assistant.'},
                            {'role': 'user', 'content': qwen_text}
                        ]
                        response3 = dashscope.Generation.call(
                            api_key="sk-dd88f2a221ea43e49799d743f92c209f",
                            model="qwen-plus",
                            messages=messages3,
                            result_format='message'
                        )
                        gen_caption1 = response3.output.choices[0].message.content
                    else:
                        gen_caption1 = gen_caption

                    data_list.append({

                        "image_path": image_paths[s],
                        "gen_base_caption": gen_base_caption1,
                        "gen_caption": gen_caption1,
                        "caption": captions[s],
                        "id": ids[s].item()
                    })


    return data_list


def create_and_load_datasets(config):

    cuhk_pre_dataset, rstp_pre_dataset, icfg_pre_dataset = create_dataset(config['pre_dataset'], config)
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
    cuhk_pre_loader, rstp_pre_loader, icfg_pre_loader = create_and_load_datasets(config)
    cuhk_gen_caption = gen_text(cuhk_pre_loader)
    rstp_gen_caption = gen_text(rstp_pre_loader)
    icfg_gen_caption = gen_text(icfg_pre_loader)

    with open(f"./pre_gen_data/cuhk_gen_pre_caption.json", "w", encoding="utf-8") as f:
        json.dump(cuhk_gen_caption, f, ensure_ascii=False, indent=4)
    with open(f"./pre_gen_data/icfg_gen_pre_caption.json", "w", encoding="utf-8") as f:
        json.dump(icfg_gen_caption, f, ensure_ascii=False, indent=4)
    with open(f"./pre_gen_data/rstp_gen_pre_caption.json", "w", encoding="utf-8") as f:
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