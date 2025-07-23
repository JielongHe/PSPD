#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:46:56 2017
@author: luohao
"""

"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

"""
Modified by L.Song and C.Wang
"""

import numpy as np
import torch
from scipy.spatial.distance import cdist

import numpy as np
from scipy.spatial.distance import cdist
import scipy.sparse as sp
#
# def re_ranking(input_feature_source, input_feature, k1=20, k2=6, lambda_value=0.1, batch_size=500):
#
#     all_num = input_feature.shape[0]
#     feat = input_feature.astype(np.float16)
#
#     if lambda_value != 0:
#         print('Computing source distance in batches...')
#         all_num_source = input_feature_source.shape[0]
#         source_dist_vec = np.zeros(all_num, dtype=np.float16)
#
#         # 分块计算source距离
#         for start in range(0, all_num, batch_size):
#             end = min(start + batch_size, all_num)
#             sour_tar_dist = np.power(
#                 cdist(input_feature[start:end], input_feature_source), 2).astype(np.float16)
#             sour_tar_dist = 1 - np.exp(-sour_tar_dist)
#             source_dist_vec[start:end] = np.min(sour_tar_dist, axis=1)
#             del sour_tar_dist
#
#         source_dist_vec = source_dist_vec / np.max(source_dist_vec)
#         source_dist = np.zeros([all_num, all_num], dtype=np.float16)
#         for i in range(all_num):
#             source_dist[i, :] = source_dist_vec + source_dist_vec[i]
#
#         del source_dist_vec
#
#     print('Computing original distance in batches...')
#     euclidean_dist_sparse = sp.lil_matrix((all_num, all_num), dtype=np.float16)
#
#     # 分块计算欧氏距离
#     for start in range(0, all_num, batch_size):
#         end = min(start + batch_size, all_num)
#         original_dist = np.power(cdist(feat[start:end], feat), 2).astype(np.float16)
#         # 仅保存最近的k1个距离
#         for i, row in enumerate(original_dist):
#             nearest_indices = np.argsort(row)[:k1 + 1]
#             euclidean_dist_sparse[start + i, nearest_indices] = row[nearest_indices]
#         del original_dist
#     # 分块计算欧氏距离
#     # for start in range(0, all_num, batch_size):
#     #     end = min(start + batch_size, all_num)
#     #     original_dist = np.power(cdist(feat[start:end], feat), 2).astype(np.float16)
#     #     euclidean_dist[start:end, :] = original_dist
#     #     del original_dist
#
#     del feat
#     original_dist = sp.csr_matrix(euclidean_dist_sparse / euclidean_dist_sparse.max(axis=0).toarray())
#     del euclidean_dist_sparse
#
#     V = sp.lil_matrix(original_dist.shape, dtype=np.float16)
#     initial_rank = np.argsort(original_dist.toarray()).astype(np.int32)
#
#     print('Starting re-ranking...')
#     for i in range(all_num):
#         forward_k_neigh_index = initial_rank[i, :k1 + 1]
#         backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
#         fi = np.where(backward_k_neigh_index == i)[0]
#         k_reciprocal_index = forward_k_neigh_index[fi]
#         k_reciprocal_expansion_index = k_reciprocal_index
#         for j in range(len(k_reciprocal_index)):
#             candidate = k_reciprocal_index[j]
#             candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
#             candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index, :int(np.around(k1 / 2)) + 1]
#             fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
#             candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
#             if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(candidate_k_reciprocal_index):
#                 k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)
#
#         k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
#         weight = np.exp(-original_dist[i, k_reciprocal_expansion_index].toarray())
#         V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
#
#     if k2 != 1:
#         V_qe = sp.lil_matrix(V.shape, dtype=np.float16)
#         for i in range(all_num):
#             V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :].toarray(), axis=0)
#         V = V_qe
#         del V_qe
#
#     del initial_rank
#     invIndex = []
#     for i in range(all_num):
#         invIndex.append(np.where(V[:, i].toarray() != 0)[0])
#
#     jaccard_dist = sp.lil_matrix(original_dist.shape, dtype=np.float16)
#
#     for i in range(all_num):
#         temp_min = np.zeros(shape=[1, all_num], dtype=np.float16)
#         indNonZero = np.where(V[i, :].toarray() != 0)[0]
#         indImages = [invIndex[ind] for ind in indNonZero]
#         for j in range(len(indNonZero)):
#             temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]].toarray(), V[indImages[j], indNonZero[j]].toarray())
#         jaccard_dist[i] = 1 - temp_min / (2 - temp_min)
#
#     pos_bool = (jaccard_dist < 0).toarray()
#     jaccard_dist[pos_bool] = 0.0
#
#     if lambda_value == 0:
#         return jaccard_dist
#     else:
#         final_dist = jaccard_dist * (1 - lambda_value) + source_dist * lambda_value
#         return final_dist

def re_ranking(input_feature_source, input_feature, k1=20, k2=6, lambda_value=0.1, batch_size=500):

    all_num = input_feature.shape[0]
    feat = input_feature.astype(np.float16)

    if lambda_value != 0:
        print('Computing source distance in batches...')
        all_num_source = input_feature_source.shape[0]
        source_dist_vec = np.zeros(all_num, dtype=np.float16)

        # 分块计算source距离
        for start in range(0, all_num, batch_size):
            end = min(start + batch_size, all_num)
            sour_tar_dist = np.power(
                cdist(input_feature[start:end], input_feature_source), 2).astype(np.float16)
            sour_tar_dist = 1 - np.exp(-sour_tar_dist)
            source_dist_vec[start:end] = np.min(sour_tar_dist, axis=1)
            del sour_tar_dist

        source_dist_vec = source_dist_vec / np.max(source_dist_vec)
        source_dist = np.zeros([all_num, all_num], dtype=np.float16)
        for i in range(all_num):
            source_dist[i, :] = source_dist_vec + source_dist_vec[i]

        del source_dist_vec

    print('Computing original distance in batches...')
    euclidean_dist = np.zeros((all_num, all_num), dtype=np.float16)

    # 分块计算欧氏距离
    for start in range(0, all_num, batch_size):
        end = min(start + batch_size, all_num)
        original_dist = np.power(cdist(feat[start:end], feat), 2).astype(np.float16)
        euclidean_dist[start:end, :] = original_dist
        del original_dist


    del feat
    original_dist = np.transpose(euclidean_dist / np.max(euclidean_dist, axis=0))
    del euclidean_dist

    V = np.zeros_like(original_dist).astype(np.float16)
    # initial_rank = np.argsort(original_dist).astype(np.int32)
    # 获取前 k1+1 个最近邻的索引
    k1_plus_1 = k1 + 1
    initial_rank = np.argpartition(original_dist, k1_plus_1, axis=1)[:, :k1_plus_1]

    # 对前 k1+1 个最近邻的距离进行排序
    for i in range(initial_rank.shape[0]):
        local_indices = initial_rank[i]
        initial_rank[i, :k1_plus_1] = local_indices[np.argsort(original_dist[i, local_indices])]
        del local_indices

    initial_rank = initial_rank.astype(np.int32)

    print('Starting re-ranking...')
    for i in range(all_num):
        forward_k_neigh_index = initial_rank[i, :k1+1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1+1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1/2))+1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index, :int(np.around(k1/2))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2/3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)

    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe

    del initial_rank
    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(all_num):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0

    if lambda_value == 0:
        return jaccard_dist
    else:
        final_dist = jaccard_dist * (1 - lambda_value) + source_dist * lambda_value
        return final_dist

def re_ranking_gpu(input_feature_source, input_feature, k1=20, k2=6, lambda_value=0.1):
    # 将输入特征转换为PyTorch张量，并迁移到GPU
    input_feature = torch.tensor(input_feature).cuda().float()
    input_feature_source = torch.tensor(input_feature_source).cuda().float()

    all_num = input_feature.shape[0]

    if lambda_value != 0:
        print('Computing source distance on GPU...')
        all_num_source = input_feature_source.shape[0]
        sour_tar_dist = torch.cdist(input_feature, input_feature_source, p=2) ** 2
        sour_tar_dist = 1 - torch.exp(-sour_tar_dist)
        source_dist_vec = torch.min(sour_tar_dist, dim=1)[0]
        source_dist_vec /= torch.max(source_dist_vec)
        source_dist = source_dist_vec.unsqueeze(1) + source_dist_vec.unsqueeze(0)
        del sour_tar_dist, source_dist_vec

    print('Computing original distance on GPU...')
    original_dist = torch.cdist(input_feature, input_feature, p=2) ** 2
    euclidean_dist = original_dist.clone()
    original_dist = original_dist / torch.max(original_dist, dim=0)[0].unsqueeze(0)

    V = torch.zeros_like(original_dist).cuda()
    initial_rank = torch.argsort(original_dist, dim=-1)

    print('Starting re_ranking on GPU...')
    for i in range(all_num):
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = torch.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index

        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :k1 // 2 + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index, :k1 // 2 + 1]
            fi_candidate = torch.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(torch.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = torch.cat((k_reciprocal_expansion_index, candidate_k_reciprocal_index))

        k_reciprocal_expansion_index = torch.unique(k_reciprocal_expansion_index)
        weight = torch.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / torch.sum(weight)

    if k2 != 1:
        V_qe = torch.zeros_like(V).cuda()
        for i in range(all_num):
            V_qe[i, :] = torch.mean(V[initial_rank[i, :k2], :], dim=0)
        V = V_qe
        del V_qe

    invIndex = [torch.where(V[:, i] != 0)[0] for i in range(all_num)]

    jaccard_dist = torch.zeros_like(original_dist).cuda()
    for i in range(all_num):
        temp_min = torch.zeros(1, all_num).cuda()
        indNonZero = torch.where(V[i, :] != 0)[0]
        for j in range(len(indNonZero)):
            temp_min[0, invIndex[indNonZero[j].item()]] += torch.min(V[i, indNonZero[j]], V[invIndex[indNonZero[j].item()], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    pos_bool = jaccard_dist < 0
    jaccard_dist[pos_bool] = 0.0

    if lambda_value == 0:
        return jaccard_dist.cpu().numpy()
    else:
        final_dist = jaccard_dist * (1 - lambda_value) + source_dist * lambda_value
        return final_dist.cpu().numpy()
