import logging
import scipy.io
import numpy as np
import os
import shutil
import numpy as np
from torch.cuda.amp import autocast
# import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.io import savemat
import time
import math
from sklearn.cluster import DBSCAN, KMeans
from sklearn import metrics
# from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import torch
import gc
from collections import Counter
import scipy.io as sio
import utils
import datetime
import random

def analysis_features():
    m = loadmat('duke' + '_pytorch_target_result.mat')
    print('type(m) = %s' % type(m))
    print(m.keys())
    target_features = m['train_f']
    data_num = len(target_features)

    np.random.seed(100)

    print('len(target_features) = %s' % len(target_features))
    print('target_features[0].size = %d' % target_features[0].size)
    # savemat(os.path.join(dst_path, 'target_features.mat'), {'features': target_features})

    target_labels = m['train_label'][0]
    # print('target_labels = %s' % (target_labels))
    # print('unique lable = %s' % np.unique(np.sort(target_labels)))
    print('real class_num = %s' % len(np.unique(np.sort(target_labels))))
    print('len(target_labels) = %s' % len(target_labels))

    target_names = m['train_name']
    print(sorted(Counter(target_labels).values())[:10])
    print(sorted(Counter(target_labels).values())[-10:])

    indices = np.random.permutation(len(target_labels))
    target_labels = target_labels[indices]
    target_features = target_features[indices]

    same_dist = []
    diff_dist = []
    same_max = []
    diff_min = []
    same_avg = []
    diff_avg = []

    for i in np.arange(len(target_features)):
        dist = ((target_features[i] - target_features) * (target_features[i] - target_features)).sum(1)
        same_sub = [dist[j] for j in np.arange(len(target_features)) if target_labels[j] == target_labels[i]]
        diff_sub = [dist[j] for j in np.arange(len(target_features)) if target_labels[j] != target_labels[i]]
        same_dist.append(same_sub)
        diff_dist.append(diff_sub[: len(same_sub)])
        same_max.append(np.max(same_sub))
        diff_min.append(np.min(diff_sub))
        same_avg.append(np.sum(same_sub) / (len(same_sub) - 1 + 1e-6))
        diff_avg.append(np.mean(diff_sub))
        if i % 200 == 0:
            print('i = %3d' % i)
        if i > 1000:
            break

    cnt = np.sum(np.array(same_max) < np.array(diff_min))
    ratio = cnt / len(same_max)
    print(ratio)
    print('avg same_max = %.3f    avg diff_min = %.3f' % (np.mean(same_max), np.mean(diff_min)))
    print('avg same = %.3f    avg diff = %.3f' % (np.mean(same_avg), np.mean(diff_avg)))

def chunked_matrix_multiplication(image_embeds, text_embeds, chunk_size=1024):
    num_rows = image_embeds.size(0)
    num_cols = text_embeds.size(0)
    sims_matrix = torch.zeros((num_rows, num_cols), device=image_embeds.device)

    for i in range(0, num_rows, chunk_size):
        end_i = min(i + chunk_size, num_rows)
        sims_matrix[i:end_i] = image_embeds[i:end_i] @ text_embeds.t()

    return sims_matrix

@torch.no_grad()
def evaluation(model, image_embeds, train_loader, text_embeds, config, captions):
    device = 'cuda'
    start_time = time.time()

    k_test = config['k_tests']
    k = config['k']  # 最终选择前 10 个
    batch_size = config.get('batch_size', 4)  # 批量大小

    text_ids = []
    text_atts = []
    image_feats = []

    with torch.no_grad():
        for n_iter, (image, caption,_, img_pid, img_path,_) in enumerate(train_loader):
            image = image.to(device, non_blocking=True)
            with torch.no_grad():
                image_feat = model.visual_encoder(image)
                image_feats.append(image_feat.cpu())
    del train_loader, image_feat, image, caption, img_pid, img_path
    gc.collect()
    torch.cuda.empty_cache()

    i2t_topk_indices = []
    t2i_topk_indices = []
    num_rows = image_embeds.size(0)

    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = num_rows // num_tasks + 1
    start = rank * step
    end = min(num_rows, start + step)

    with torch.no_grad():
        for i in range(start, end, batch_size):
            batch_end = min(i + batch_size, end)
            caption = captions[i:batch_end]
            text_input = model.tokenizer(caption, padding='max_length', truncation=True, max_length=73,
                                         return_tensors="pt").to(device)
            text_ids.append(text_input.input_ids)
            text_atts.append(text_input.attention_mask)

    num_cols = text_embeds.size(0)
    del captions, text_input, caption
    gc.collect()
    torch.cuda.empty_cache()

    image_feats = torch.cat(image_feats, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    print(1)

    for i in range(start, end, batch_size):
        batch_end = min(i + batch_size, end)
        # sims_batch = sims_matrix[i:batch_end]

        sims_batch = image_embeds[i:batch_end] @ text_embeds.t()

        # 获取 top-k_test 相似度
        topk_sim, topk_idx = sims_batch.topk(k=k_test, dim=1)
        # image = images[i:batch_end].to(device)
        # image_feats = model.visual_encoder(image)
        encoder_output = image_feats[i:batch_end].unsqueeze(1).repeat(1, k_test, 1, 1).to(device)
        encoder_output = encoder_output.view(-1, encoder_output.size(-2), encoder_output.size(-1))
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)

        # 混合精度计算
        with autocast():
            output = model.text_encoder(text_ids[topk_idx.view(-1)],
                                        attention_mask=text_atts[topk_idx.view(-1)],
                                        encoder_hidden_states=encoder_output,
                                        encoder_attention_mask=encoder_att,
                                        return_dict=True,
                                        )
            score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]

        # 获取 top-k 中得分最高的前 10 个
        score = score.view(batch_end - i, k_test)
        final_topk_sim, final_topk_idx = (score + topk_sim).topk(k=k, dim=1)

        # 存储 top-k 索引和分数
        i2t_topk_indices.append(topk_idx.gather(1, final_topk_idx).cpu().numpy())
        # i2t_topk_scores.append(final_topk_sim.cpu().numpy())

    i2t_topk_indices = [item for sublist in i2t_topk_indices for item in sublist]

    del encoder_output, encoder_att, sims_batch, topk_sim, topk_idx, final_topk_sim, final_topk_idx  # 释放内存
    torch.cuda.empty_cache()  # 清空缓存
    step = num_cols // num_tasks + 1
    start = rank * step
    end = min(num_cols, start + step)

    # for i, text_embed in enumerate(metric_logger.log_every(text_embeds[start:end], 50, header)):
    #     # 获取 top-k_test 相似度
    #     sims = text_embed @ image_embeds.t()
    #
    #     topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
    #     # image = images[topk_idx.cpu()].to(device)
    #     encoder_output = image_feats[topk_idx.cpu()].to(device)
    #     encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
    #
    #     # 混合精度计算
    #     with autocast():
    #         output = model.text_encoder(text_ids[start + i].repeat(k_test, 1),
    #                                     attention_mask=text_atts[start + i].repeat(k_test, 1),
    #                                     encoder_hidden_states=encoder_output,
    #                                     encoder_attention_mask=encoder_att,
    #                                     return_dict=True,
    #                                     )
    #         score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
    #
    #     final_topk_sim, final_topk_idx = (score + topk_sim).topk(k=k, dim=0)
    #     t2i_topk_indices.append(topk_idx[final_topk_idx].cpu().numpy())
    # 文本到图像
    batch_size1 = 2
    for i in range(start, end, batch_size1):
        batch_end = min(i + batch_size1, end)

        # 获取文本嵌入的小批次
        text_embed_batch = text_embeds[i:batch_end]

        # 获取 top-k_test 相似度
        sims_batch = text_embed_batch @ image_embeds.t()
        topk_sim, topk_idx = sims_batch.topk(k=k_test, dim=1)

        # 提取相应的图像特征批次
        encoder_output = image_feats[topk_idx.cpu().view(-1)].to(device)
        encoder_output = encoder_output.view(batch_end - i, k_test, encoder_output.size(-2), encoder_output.size(-1))
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)

        # 混合精度计算
        with autocast():
            output = model.text_encoder(
                text_ids[i:batch_end].repeat_interleave(k_test, dim=0),
                attention_mask=text_atts[i:batch_end].repeat_interleave(k_test, dim=0),
                encoder_hidden_states=encoder_output.view(-1, encoder_output.size(-2), encoder_output.size(-1)),
                encoder_attention_mask=encoder_att.view(-1, encoder_output.size(2)),
                return_dict=True,
            )
            score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]

        score = score.view(batch_end - i, k_test)
        final_topk_sim, final_topk_idx = (score + topk_sim).topk(k=k, dim=1)

        # 存储 top-k 索引和分数
        t2i_topk_indices.append(topk_idx.gather(1, final_topk_idx).cpu().numpy())
    t2i_topk_indices = [item for sublist in t2i_topk_indices for item in sublist]

    del encoder_output, encoder_att, sims_batch, topk_sim, topk_idx, final_topk_sim, final_topk_idx, image_feats, text_ids, text_atts
    torch.cuda.empty_cache()

    # 记录运行时间
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    # 返回 top-k 的索引和分数
    return i2t_topk_indices, t2i_topk_indices


def update_labels_based_on_matching(cluster_centers, target_features, i2t_topk_indices, t2i_topk_indices, labels):

    num_texts = len(t2i_topk_indices)
    memory = [None] * num_texts
    final_texts_for_images = []
    for i_index in range(num_texts):
        nearest_text = i2t_topk_indices[i_index]
        valid_texts = []
        # valid_texts.append(i_index)
        for text_index in nearest_text:
            nearest_images_for_text = t2i_topk_indices[text_index]
            if text_index in nearest_images_for_text:
                valid_texts.append(text_index)
        final_texts_for_images.append(valid_texts)

    final_images_for_texts = []
    for t_index in range(num_texts):
        nearest_images = t2i_topk_indices[t_index]
        valid_images = []
        # valid_images.append(t_index)
        for image_index in nearest_images:
            nearest_texts_for_image = i2t_topk_indices[image_index]
            if image_index in nearest_texts_for_image:
                valid_images.append(image_index)
        final_images_for_texts.append(valid_images)

    final_texts_images = []
    for idx in range(num_texts):
        if final_texts_for_images[idx] == None:
            continue
        if final_images_for_texts[idx] == None:
            continue
        final_texts = final_texts_for_images[idx]
        final_text_image = []
        for final_text in final_texts:
            if final_text in final_images_for_texts[idx]:
                final_text_image.append(final_text)
        final_texts_images.append(final_text_image)

    new_labels = labels.copy()
    sum = 0

    for index, value in enumerate(final_texts_images):
        if len(value) == 0:
            continue
        matched_labels = [labels[i] for i in value]
        if len(labels) > index:
            matched_labels.append(labels[index])
            value.append(index)
        label_counter = Counter(matched_labels)
        if -1 in label_counter:
            del label_counter[-1]
        if label_counter:
            max_count = max(label_counter.values())
            most_common_labels = [label for label, count in label_counter.items() if count == max_count]
            if len(most_common_labels) > 1:
                if labels[index] in most_common_labels:
                    most_common_label = labels[index]
                else:
                    most_common_label = random.choice(most_common_labels)
            else:
                most_common_label = random.choice(most_common_labels)

        else:
            continue

        for i in value:
            if memory[i] == None:
                new_labels[i] = most_common_label
                if labels[i] == -1:
                    sum += 1
                    labels[i] = most_common_label
                memory[i] = cluster_centers[most_common_labels[0]]
            else:
                old_distance = torch.norm(memory[i] - target_features[i], p=2)
                new_distance = torch.norm(cluster_centers[most_common_labels[0]] - target_features[i], p=2)
                if new_distance > old_distance:
                    new_labels[i] = most_common_label
                    memory[i] = most_common_label

    print(f'add{sum}')
    return new_labels


def generate_cluster(target_features, target_labels, img_paths, captions, multi_captions, gpt_gen_captions,
                     model, train_loader, config,image_embeds,text_embeds,
                     dist=None, eps=0.8, min_samples=10, data_dir=None, flag='did'):
    # data_num = len(target_features)
    process_num = target_features.shape[0]
    if dist is None:
        X = target_features[:process_num]
    else:
        X = dist[:process_num]
    #
    labels_true = target_labels[:process_num]
    labels_true = labels_true.cpu().numpy()

    print('DBSCAN starting ......')

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    lab = db.labels_
    del dist
    torch.cuda.empty_cache()
    labels = lab.copy()
    # sio.savemat('labels.mat', {'labels': labels})
    i2t_topk_indices, t2i_topk_indices = evaluation(model, image_embeds, train_loader, text_embeds, config, captions)
    unique_labels = set(labels)
    cluster_centers = {}

    for label in unique_labels:
        if label != -1:  # 排除噪声点
            # 找到属于该簇的核心样本点
            class_member_mask = (labels == label) & core_samples_mask
            cluster_center = target_features[class_member_mask].mean(axis=0)
            cluster_centers[label] = cluster_center
    #
    updated_labels = update_labels_based_on_matching(cluster_centers, target_features, i2t_topk_indices,
                                                     t2i_topk_indices, labels)
    labels = updated_labels


    print(sorted(Counter(labels).values())[:10])
    print(sorted(Counter(labels).values())[-10:])


    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters_)

    n_node = np.sum(labels != -1)
    print('Estimated number of nodes: %d' % n_node)

    dir_cnt = 0
    image_cnt = 0
    datasets = []


    for itr, label in enumerate(labels):
        # if label != -1 and label not in values_gt_100:
        if label != -1:

            dataset = [int(label), img_paths[itr], captions[itr], gpt_gen_captions[itr]]
            dataset = tuple(dataset)
            datasets.append(dataset)

            dataset1 = [int(label), img_paths[itr], multi_captions[itr], gpt_gen_captions[itr]]
            dataset1 = tuple(dataset1)
            datasets.append(dataset1)


    print('valid cluster number: %d    file number:%d' % (dir_cnt, image_cnt))

    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))

    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))


    return datasets, n_clusters_






def generate_cluster_kmeans(cluster_result_path, dist=None, eps=0.8, min_samples=10, data_dir=None, flag='did'):
    m = loadmat(str(0) + '_' + flag + '_' + data_dir + '_pytorch_target_result.mat')
    target_features = m['train_f']
    data_num = len(target_features)
    print('len(target_features) = %s' % len(target_features))
    print('target_features[0].size = %d' % target_features[0].size)
    target_labels = m['train_label'][0]
    print('real class_num = %s' % len(np.unique(np.sort(target_labels))))
    print('len(target_labels) = %s' % len(target_labels))
    target_names = m['train_name']

    if os.path.exists(cluster_result_path):
        shutil.rmtree(cluster_result_path)
    os.mkdir(cluster_result_path)
    process_num = m['train_label'][0].shape[0]
    if dist is None:
        X = target_features[:process_num]
    else:
        X = dist[:process_num]
    labels_true = target_labels[:process_num]
    names = target_names[:process_num]
    print('KMeans starting ......')
    db = KMeans(n_clusters=702).fit(X)
    labels = db.labels_
    print(sorted(Counter(labels).values())[:10])
    print(sorted(Counter(labels).values())[-10:])

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters_)
    n_node = np.sum(labels != -1)
    print('Estimated number of nodes: %d' % n_node)
    dir_cnt = 0
    image_cnt = 0
    for i in np.arange(n_clusters_):
        files = names[np.where(labels == i)]
        if len(files) > np.max((2, min_samples - 1)):
            dir_path = os.path.join(cluster_result_path, str(i).zfill(4))
            os.mkdir(dir_path)
            for file in files:
                file = file.strip()
                shutil.copy(
                    os.path.join(os.path.split(os.path.split(cluster_result_path)[0])[0], 'bounding_box_train', file),
                    os.path.join(dir_path, file))
        dir_cnt += 1
        image_cnt += len(files)

    print('valid cluster number: %d    file number:%d' % (dir_cnt, image_cnt))
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    # print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
    # print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))

def generate_cluster_with_semantic_clean(cluster_result_path, dist=None, eps=0.8, min_samples=10, domain_num=5, data_dir=None, flag='all'):
    info_mat = []
    target_features_all = []
    target_labels_all = []
    target_names_all = []
    target_cluster_result_all = []
    target_cluster_num_all = []
    target_nodes_all = []
    if 'did' in flag:
        clean_list = ['did']
    elif 'sid' in flag:
        clean_list = ['sid']
    else:
        clean_list = ['all', 'did', 'sid']
    for cl in clean_list:
        m = loadmat(str(0) + '_' + cl + '_' + data_dir + '_pytorch_target_result.mat')
        target_features = m['train_f']
        target_labels = m['train_label'][0]
        target_names = m['train_name']
        process_num = m['train_label'][0].shape[0]
        if dist is None:
            X = target_features[:process_num]
        else:
            X = dist[cl][:process_num]
        labels_true = target_labels[:process_num]
        names = target_names[:process_num]
        print('Semantic %s  DBSCAN starting ......' % cl)
        db = DBSCAN(eps=eps[cl], min_samples=min_samples).fit(X)
        labels = db.labels_
        print(sorted(Counter(labels).values())[:10])
        print(sorted(Counter(labels).values())[-10:])
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print('Estimated number of clusters: %d' % n_clusters_)
        n_node = np.sum(labels != -1)
        print('Estimated number of nodes: %d' % n_node)
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))

        target_features_all.append(target_features)
        target_labels_all.append(labels_true)
        target_names_all.append(target_names)
        target_cluster_result_all.append(labels)
        target_cluster_num_all.append(n_clusters_)
        sub_nodes = []
        for j in np.arange(n_clusters_):
            nodes = names[np.where(labels == j)]
            sub_nodes.append(nodes)
        target_nodes_all.append(sub_nodes)

    print('Semantic Cluster Data Cleaning ......')
    cnt = 0
    for i in np.arange(1, len(clean_list)):
        for j in np.arange(target_cluster_num_all[0]):
            iou_max = 0.0
            index = -1
            for k in np.arange(target_cluster_num_all[i]):
                iou = len(np.intersect1d(target_nodes_all[0][j], target_nodes_all[i][k])) / len(
                    np.union1d(target_nodes_all[0][j], target_nodes_all[i][k]))
                if iou > 1e-6:
                    # print('cnt = %3d  iou = %.3f' % (cnt, iou))
                    cnt += 1
                if iou > iou_max:
                    iou_max = iou
                    index = k
            if index != -1:
                disabled_nodes = np.setdiff1d(target_nodes_all[0][j], target_nodes_all[i][index])
                ind = [np.where(target_names_all[0] == x)[0][0] for x in disabled_nodes]
                for _ind in ind:
                    target_cluster_result_all[0][_ind] = -1
                target_nodes_all[0][j] = np.intersect1d(target_nodes_all[0][j], target_nodes_all[i][index])
            else:
                disabled_nodes = target_nodes_all[0][j]
                ind = [np.where(target_names_all[0] == x)[0][0] for x in disabled_nodes]
                for _ind in ind:
                    target_cluster_result_all[0][_ind] = -1
                target_nodes_all[0][j] = []

    if os.path.exists(cluster_result_path):
        shutil.rmtree(cluster_result_path)
    os.mkdir(cluster_result_path)
    dir_cnt = 0
    image_cnt = 0
    for i in np.arange(len(target_nodes_all[0])):
        dir_path = os.path.join(cluster_result_path, str(i).zfill(4))
        if len(target_nodes_all[0][i]) >= 4:
            os.mkdir(dir_path)
            files = target_nodes_all[0][i]
            for file in files:
                file = file.strip()
                shutil.copy(
                    os.path.join(os.path.split(os.path.split(cluster_result_path)[0])[0], 'bounding_box_train', file),
                    os.path.join(dir_path, file))
            dir_cnt += 1
            image_cnt += len(files)
    print('valid cluster number: %d    file number:%d' % (dir_cnt, image_cnt))
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(target_labels_all[0], target_cluster_result_all[0]))
    print("Completeness: %0.3f" % metrics.completeness_score(target_labels_all[0], target_cluster_result_all[0]))

def generate_cluster_with_semantic_kmeans_clean(cluster_result_path, dist=None, eps=0.8, min_samples=10, domain_num=5, data_dir=None, flag='all'):
    info_mat = []
    target_features_all = []
    target_labels_all = []
    target_names_all = []
    target_cluster_result_all = []
    target_cluster_num_all = []
    target_nodes_all = []
    if 'did' in flag:
        clean_list = ['did']
    elif 'sid' in flag:
        clean_list = ['sid']
    else:
        clean_list = ['all', 'did', 'sid']
    cluster_num = 0
    indices = 0
    for cl in clean_list:
        if 'all' in cl:
            m = loadmat(str(0) + '_' + cl + '_' + data_dir + '_pytorch_target_result.mat')
            target_features = m['train_f']
            target_labels = m['train_label'][0]
            target_names = m['train_name']
            process_num = m['train_label'][0].shape[0]
            if dist is None:
                X = target_features[:process_num]
            else:
                X = dist[cl][:process_num]
            labels_true = target_labels[:process_num]
            names = target_names[:process_num]
            print('Semantic  %s  DBSCAN  cluster starting ......' % cl)
            db = DBSCAN(eps=eps[cl], min_samples=min_samples).fit(X)
            labels = db.labels_
            db_labels = labels
            indices = np.where(labels != -1)[0]
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            cluster_num = n_clusters_
        else:
            m = loadmat(str(0) + '_' + cl + '_' + data_dir + '_pytorch_target_result.mat')
            target_features = m['train_f']
            target_labels = m['train_label'][0]
            target_names = m['train_name']
            process_num = m['train_label'][0].shape[0]
            if dist is None:
                X = target_features[:process_num]
            else:
                X = dist[cl][:process_num]
            labels_true = target_labels[:process_num]
            names = target_names[:process_num]
            print('Semantic  %s  KMeans  cluster starting ......' % cl)
            cluster_center = np.zeros((cluster_num, X.shape[1]))
            for i in np.arange(cluster_num):
                cluster_center[i] = np.average(X[np.where(db_labels==i)[0]], 0)
            db = KMeans(n_clusters=cluster_num, init=cluster_center).fit(X)
            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print(sorted(Counter(labels).values())[:10])
        print(sorted(Counter(labels).values())[-10:])
        print('Estimated number of clusters: %d' % n_clusters_)
        n_node = np.sum(labels != -1)
        print('Estimated number of nodes: %d' % n_node)
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
        target_features_all.append(target_features)
        target_labels_all.append(labels_true)
        target_names_all.append(target_names)
        target_cluster_result_all.append(labels)
        target_cluster_num_all.append(n_clusters_)
        sub_nodes = []
        for j in np.arange(n_clusters_):
            nodes = names[np.where(labels == j)]
            sub_nodes.append(nodes)
        target_nodes_all.append(sub_nodes)

    print('Semantic Cluster Data Cleaning ......')
    cnt = 0
    for i in np.arange(1, len(clean_list)):
        for j in np.arange(target_cluster_num_all[0]):
            iou_max = 0.0
            index = -1
            for k in np.arange(target_cluster_num_all[i]):
                iou = len(np.intersect1d(target_nodes_all[0][j], target_nodes_all[i][k])) / len(
                    np.union1d(target_nodes_all[0][j], target_nodes_all[i][k]))
                if iou > 1e-6:
                    # print('cnt = %3d  iou = %.3f' % (cnt, iou))
                    cnt += 1
                if iou > iou_max:
                    iou_max = iou
                    index = k
            if index != -1:
                disabled_nodes = np.setdiff1d(target_nodes_all[0][j], target_nodes_all[i][index])
                ind = [np.where(target_names_all[0] == x)[0][0] for x in disabled_nodes]
                for _ind in ind:
                    target_cluster_result_all[0][_ind] = -1
                target_nodes_all[0][j] = np.intersect1d(target_nodes_all[0][j], target_nodes_all[i][index])
            else:
                disabled_nodes = target_nodes_all[0][j]
                ind = [np.where(target_names_all[0] == x)[0][0] for x in disabled_nodes]
                for _ind in ind:
                    target_cluster_result_all[0][_ind] = -1
                target_nodes_all[0][j] = []

    if os.path.exists(cluster_result_path):
        shutil.rmtree(cluster_result_path)
    os.mkdir(cluster_result_path)
    dir_cnt = 0
    image_cnt = 0
    for i in np.arange(len(target_nodes_all[0])):
        dir_path = os.path.join(cluster_result_path, str(i).zfill(4))
        if len(target_nodes_all[0][i]) >= 4:
            os.mkdir(dir_path)
            files = target_nodes_all[0][i]
            for file in files:
                file = file.strip()
                shutil.copy(
                    os.path.join(os.path.split(os.path.split(cluster_result_path)[0])[0], 'bounding_box_train', file),
                    os.path.join(dir_path, file))
            dir_cnt += 1
            image_cnt += len(files)
    print('valid cluster number: %d    file number:%d' % (dir_cnt, image_cnt))
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(target_labels_all[0], target_cluster_result_all[0]))
    print("Completeness: %0.3f" % metrics.completeness_score(target_labels_all[0], target_cluster_result_all[0]))

def generate_cluster_with_domain_clean(cluster_result_path, dist=None, eps=0.8, min_samples=10, domain_num=5, data_dir=None, flag='did'):
    info_mat = []
    target_features_all = []
    target_labels_all = []
    target_names_all = []
    target_cluster_result_all = []
    target_cluster_num_all = []
    target_nodes_all = []
    for i in np.arange(domain_num):
        m = loadmat(str(i) + '_' + flag + '_' + data_dir + '_pytorch_target_result.mat')
        target_features = m['train_f']
        target_labels = m['train_label'][0]
        target_names = m['train_name']

        process_num = m['train_label'][0].shape[0]
        if dist is None:
            X = target_features[:process_num]
        else:
            X = dist[i][:process_num]
        labels_true = target_labels[:process_num]
        names = target_names[:process_num]
        print('Domain %d  DBSCAN starting ......' % i)
        db = DBSCAN(eps=eps[i], min_samples=min_samples).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        print(sorted(Counter(labels).values())[:10])
        print(sorted(Counter(labels).values())[-10:])
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print('Estimated number of clusters: %d' % n_clusters_)
        n_node = np.sum(labels != -1)
        print('Estimated number of nodes: %d' % n_node)
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))

        target_features_all.append(target_features)
        target_labels_all.append(labels_true)
        target_names_all.append(target_names)
        target_cluster_result_all.append(labels)
        target_cluster_num_all.append(n_clusters_)
        sub_nodes = []
        for j in np.arange(n_clusters_):
            nodes = names[np.where(labels == j)]
            sub_nodes.append(nodes)
        target_nodes_all.append(sub_nodes)

    cnt = 0
    for i in np.arange(1, domain_num):
        for j in np.arange(target_cluster_num_all[0]):
            iou_max = 0.0
            index = -1
            for k in np.arange(target_cluster_num_all[i]):
                iou = len(np.intersect1d(target_nodes_all[0][j], target_nodes_all[i][k])) / len(
                    np.union1d(target_nodes_all[0][j], target_nodes_all[i][k]))
                if iou > 1e-6:
                    # print('cnt = %3d  iou = %.3f' % (cnt, iou))
                    cnt += 1
                if iou > iou_max:
                    iou_max = iou
                    index = k
            if index != -1:
                disabled_nodes = np.setdiff1d(target_nodes_all[0][j], target_nodes_all[i][index])
                ind = [np.where(target_names_all[0] == x)[0][0] for x in disabled_nodes]
                for _ind in ind:
                    target_cluster_result_all[0][_ind] = -1
                target_nodes_all[0][j] = np.intersect1d(target_nodes_all[0][j], target_nodes_all[i][index])
            else:
                disabled_nodes = target_nodes_all[0][j]
                ind = [np.where(target_names_all[0] == x)[0][0] for x in disabled_nodes]
                for _ind in ind:
                    target_cluster_result_all[0][_ind] = -1
                target_nodes_all[0][j] = []

    if os.path.exists(cluster_result_path):
        shutil.rmtree(cluster_result_path)
    os.mkdir(cluster_result_path)
    dir_cnt = 0
    image_cnt = 0
    for i in np.arange(len(target_nodes_all[0])):
        dir_path = os.path.join(cluster_result_path, str(i).zfill(4))
        if len(target_nodes_all[0][i]) >= np.max((3, min_samples - 1)):
            os.mkdir(dir_path)
            files = target_nodes_all[0][i]
            for file in files:
                file = file.strip()
                shutil.copy(
                    os.path.join(os.path.split(os.path.split(cluster_result_path)[0])[0], 'bounding_box_train', file),
                    os.path.join(dir_path, file))
            dir_cnt += 1
            image_cnt += len(files)
    print('valid cluster number: %d    file number:%d' % (dir_cnt, image_cnt))
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(target_labels_all[0], target_cluster_result_all[0]))
    print("Completeness: %0.3f" % metrics.completeness_score(target_labels_all[0], target_cluster_result_all[0]))


def generate_cluster_with_clean_union(cluster_result_path, eps=0.5, min_samples=5, domain_num=2):
    info_mat = []
    target_features_all = []
    target_labels_all = []
    target_names_all = []
    target_cluster_result_all = []
    target_cluster_num_all = []
    target_nodes_all = []
    for i in np.arange(domain_num):
        m = loadmat(str(i) + '_duke' + '_pytorch_target_result.mat')
        target_features = m['train_f']
        target_labels = m['train_label'][0]
        target_names = m['train_name']

        process_num = m['train_label'][0].shape[0]
        X = target_features[:process_num]
        labels_true = target_labels[:process_num]
        names = target_names[:process_num]
        print('Domain %d  DBSCAN starting ......' % i)
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        print(sorted(Counter(labels).values())[:10])
        print(sorted(Counter(labels).values())[-10:])
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print('Estimated number of clusters: %d' % n_clusters_)
        n_node = np.sum(labels != -1)
        print('Estimated number of nodes: %d' % n_node)
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))

        target_features_all.append(target_features)
        target_labels_all.append(labels_true)
        target_names_all.append(target_names)
        target_cluster_result_all.append(labels)
        target_cluster_num_all.append(n_clusters_)
        sub_nodes = []
        for j in np.arange(n_clusters_):
            nodes = names[np.where(labels == j)]
            sub_nodes.append(nodes)
        target_nodes_all.append(sub_nodes)

    cnt = 0
    for i in np.arange(1, domain_num):
        for j in np.arange(target_cluster_num_all[0]):
            iou_max = 0.0
            index = -1
            for k in np.arange(target_cluster_num_all[i]):
                iou = len(np.intersect1d(target_nodes_all[0][j], target_nodes_all[i][k])) / len(
                    np.union1d(target_nodes_all[0][j], target_nodes_all[i][k]))
                if iou > 1e-6:
                    # print('cnt = %3d  iou = %.3f' % (cnt, iou))
                    cnt += 1
                if iou > iou_max:
                    iou_max = iou
                    index = k
            if index != -1:
                enabled_nodes = np.setdiff1d(target_nodes_all[i][index], target_nodes_all[0][j])
                ind = [np.where(target_names_all[0] == x)[0][0] for x in enabled_nodes]
                for _ind in ind:
                    target_cluster_result_all[0][_ind] = target_cluster_result_all[0][j]
                target_nodes_all[0][j] = np.union1d(target_nodes_all[0][j], target_nodes_all[i][index])

    if os.path.exists(cluster_result_path):
        shutil.rmtree(cluster_result_path)
    os.mkdir(cluster_result_path)
    dir_cnt = 0
    image_cnt = 0
    for i in np.arange(len(target_nodes_all[0])):
        dir_path = os.path.join(cluster_result_path, str(i).zfill(4))
        if len(target_nodes_all[0][i]) >= np.max((2, min_samples - 1)):
            os.mkdir(dir_path)
            files = target_nodes_all[0][i]
            for file in files:
                shutil.copy(
                    os.path.join(os.path.split(os.path.split(cluster_result_path)[0])[0], 'bounding_box_train', file),
                    os.path.join(dir_path, file))
            dir_cnt += 1
            image_cnt += len(files)
    print('valid cluster number: %d    file number:%d' % (dir_cnt, image_cnt))
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(target_labels_all[0], target_cluster_result_all[0]))
    print("Completeness: %0.3f" % metrics.completeness_score(target_labels_all[0], target_cluster_result_all[0]))


if __name__ == '__main__':
    # analysis_features()
    # cluster()
    # generate_cluster_data('data/duke/pytorch/train_all_cluster')
    generate_cluster_data_with_clean('data/duke/pytorch/train_all_cluster')
