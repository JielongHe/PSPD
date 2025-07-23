
import shutil
import torch
import gc

from processor.finetune import generate_cluster
from processor.get_multi_target_features import get_distances

import torch.nn.functional as F


eps0_all = 1.004


@torch.no_grad()
def pre_train( i, train_loader, source_train_loader, pre_model, min_samples, config, eps0_all = None):
    device = "cuda"

    source_features, target_features = [], []
    gpt_gen_captions,img_paths, captions, multi_captions, labels, upper_captions, lower_captions = [], [], [], [],[], [],[]
    text_embeds, image_embeds = [], []

    pre_model.eval()

    for n_iter, (image, multi_caption, caption, img_pid, img_path, gpt_gen_caption) in enumerate(train_loader):
        image = image.to(device, non_blocking=True)
        img_pid = img_pid.to(device, non_blocking=True)
        gpt_gen_captions.extend(gpt_gen_caption)
        labels.append(img_pid)
        img_paths.extend(img_path)

        multi_captions.extend(multi_caption)
        captions.extend(caption)
        text_input = pre_model.tokenizer(multi_caption, padding='max_length', truncation=True, max_length=73,
                                         return_tensors="pt").to(device)

        text_output = pre_model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask,
                                             mode='text')
        text_embed = F.normalize(pre_model.text_proj(text_output.last_hidden_state[:, 0, :]))
        text_embeds.append(text_embed)

        image_feat = pre_model.visual_encoder(image)
        image_embed = F.normalize(pre_model.vision_proj(image_feat[:, 0, :]), dim=-1)
        image_embeds.append(image_embed)


        fused_output = torch.cat((image_embed, text_embed), dim=1)
        target_features.append(fused_output)

        del text_output, image_feat, image, img_pid, image_embed, text_embed, text_input
        gc.collect()
        torch.cuda.empty_cache()


    for t, (image1, caption1) in enumerate(source_train_loader):

        image1 = image1.to(device, non_blocking=True)
        text_input1 = pre_model.tokenizer(caption1, padding='max_length', truncation=True, max_length=73,
                                          return_tensors="pt").to(device)

        text_output1 = pre_model.text_encoder(text_input1.input_ids, attention_mask=text_input1.attention_mask,
                                              mode='text')
        text_embed1 = F.normalize(pre_model.text_proj(text_output1.last_hidden_state[:, 0, :]))

        image_feat1 = pre_model.visual_encoder(image1)
        image_embed1 = F.normalize(pre_model.vision_proj(image_feat1[:, 0, :]), dim=-1)

        fused_output1 = torch.cat((image_embed1, text_embed1), dim=1)
        source_features.append(fused_output1)

        del image1, text_input1, text_output1, text_embed1, image_feat1, image_embed1, fused_output1
        gc.collect()
        torch.cuda.empty_cache()
        # 分批次拼接特征，避免一次性占用大量内存
    target_features = torch.cat(target_features, dim=0).half().cpu()  # 将数据转移到 CPU 以减少 GPU 内存
    source_features = torch.cat(source_features, dim=0).half().cpu()
    image_embeds = torch.cat(image_embeds, dim=0)
    text_embeds = torch.cat(text_embeds, dim=0)

    del source_train_loader
    gc.collect()
    torch.cuda.empty_cache()

    # 拼接标签
    target_labels = torch.cat(labels, dim=0)
    #
    if i == 0:
        dist_all, eps0_all = get_distances(source_features, target_features, i, ratio=0.051, domain=0,
                                           flag='all')
    else:
        dist_all, _ = get_distances(source_features, target_features, i, ratio=0.051, domain=0, flag='all')
    print(eps0_all)

    del source_features
    gc.collect()
    torch.cuda.empty_cache()

    # 生成聚类数据集
    datasets, n_clusters = generate_cluster(target_features, target_labels, img_paths, captions, multi_captions, gpt_gen_captions,
                                            pre_model, train_loader, config, image_embeds, text_embeds,
                                            dist=dist_all, eps=eps0_all, min_samples=min_samples, flag='all')

    # 删除不再需要的变量，清理内存
    del target_labels, target_features, dist_all, img_paths, captions, pre_model, image_embeds, text_embeds, train_loader
    gc.collect()
    torch.cuda.empty_cache()

    return datasets, n_clusters, eps0_all

