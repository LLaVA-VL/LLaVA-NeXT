#    Copyright 2024 Flash-VStream Authors 
#
#    Licensed under the Apache License, Version 2.0 (the "License"); 
#    you may not use this file except in compliance with the License. 
#    You may obtain a copy of the License at 
#
#        http://www.apache.org/licenses/LICENSE-2.0 
#
#    Unless required by applicable law or agreed to in writing, software 
#    distributed under the License is distributed on an "AS IS" BASIS, 
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
#    See the License for the specific language governing permissions and 
#    limitations under the License. 

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

def drop_feature(img_feature, video_max_frames, img_similarity=None):
    T, P, D = img_feature.shape
    indices = [[i] for i in range(T)]
    T0 = video_max_frames
    if T <= T0:
        return img_feature, img_similarity, [indices]
    cur_feature = img_feature[:T0]  # [T0, P, D]
    if img_similarity is not None:
        cur_sim = img_similarity[:T0 - 1]
    else:
        cur_sim = F.cosine_similarity(cur_feature[:-1].view(T0 - 1, P * D), cur_feature[1:].view(T0 - 1, P * D))  # [T0 - 1]
    cur_indices = indices[:T0]
    step_indices = [cur_indices]
    for i in range(T0, T):
        new_feature = img_feature[i]
        new_sim = F.cosine_similarity(cur_feature[-1].view(-1), new_feature.view(-1), dim=0)
        all_feature = torch.cat([cur_feature, new_feature.unsqueeze(0)], dim=0)
        all_indices = cur_indices + [[i]]
        all_sim = torch.cat([cur_sim, new_sim.unsqueeze(0)], dim=0)
        idx = torch.argmax(all_sim)
        if random.randint(0, 1) > 0:
            idx = idx + 1
        cur_feature = torch.cat([all_feature[:idx], all_feature[idx + 1:]])
        if idx + 1 == T0 + 1:
            cur_sim = all_sim[:T0 - 1]
            cur_indices = all_indices[:-1] 
        elif idx == 0:
            cur_sim = all_sim[1:]
            cur_indices = all_indices[1:] 
        else:
            cur_sim = torch.cat([all_sim[:idx], all_sim[idx + 1:]])
            cur_sim[idx - 1] = F.cosine_similarity(all_feature[idx - 1].view(-1), all_feature[idx + 1].view(-1), dim=0)
            cur_indices = all_indices[:idx] + all_indices[idx + 1:]
        step_indices.append(cur_indices)
    # print(f'Note: perform drop feature {img_feature.shape} to {cur_feature.shape}')
    return cur_feature, cur_sim, step_indices


def merge_feature(img_feature, video_max_frames, img_similarity=None):
    T, P, D = img_feature.shape
    indices = [[i] for i in range(T)]
    T0 = video_max_frames
    if T <= T0:
        return img_feature, img_similarity, [indices]
    cur_feature = img_feature[:T0]  # [T0, P, D]
    cur_indices = indices[:T0]
    step_indices = [cur_indices]
    if img_similarity is not None:
        cur_sim = img_similarity[:T0 - 1]
    else:
        cur_sim = F.cosine_similarity(cur_feature[:-1].view(T0 - 1, P * D), cur_feature[1:].view(T0 - 1, P * D))  # [T0 - 1]
    for i in range(T0, T):
        new_feature = img_feature[i]
        new_sim = F.cosine_similarity(cur_feature[-1].view(-1), new_feature.view(-1), dim=0)
        all_feature = torch.cat([cur_feature, new_feature.unsqueeze(0)], dim=0)
        all_sim = torch.cat([cur_sim, new_sim.unsqueeze(0)], dim=0)
        all_indices = cur_indices + [[i]]
        idx = torch.argmax(all_sim)
        all_feature[idx + 1] = (all_feature[idx] + all_feature[idx + 1]) / 2.0
        all_indices[idx + 1] = all_indices[idx] + all_indices[idx + 1]
        cur_feature = torch.cat([all_feature[:idx], all_feature[idx + 1:]])
        cur_sim = torch.cat([all_sim[:idx], all_sim[idx + 1:]])
        cur_indices = all_indices[:idx] + all_indices[idx + 1:]
        if idx > 0:
            cur_sim[idx - 1] = F.cosine_similarity(all_feature[idx - 1].view(-1), all_feature[idx + 1].view(-1), dim=0)
        if idx + 1 < T0:
            cur_sim[idx] = F.cosine_similarity(all_feature[idx + 1].view(-1), all_feature[idx + 2].view(-1), dim=0)
        step_indices.append(cur_indices)
    # print(f'Note: perform merge feature {img_feature.shape} to {cur_feature.shape}')
    return cur_feature, cur_sim, step_indices


def kmeans_feature(img_feature, video_max_frames, img_similarity=None):
    def kmeans_torch(X, num_clusters, distance='euclidean', tol=1e-4, max_iter=10):
        indices = torch.randperm(X.size(0))[:num_clusters]
        centroids = X[indices]
        for i in range(max_iter):
            if distance == 'euclidean':
                dists = torch.cdist(X, centroids, p=2)
            else:
                raise NotImplementedError("Only Euclidean distance is supported yet")
            labels = torch.argmin(dists, dim=1)
            new_centroids = []
            for j in range(num_clusters):
                cluster_points = X[labels == j]
                if len(cluster_points) > 0:
                    new_centroid = cluster_points.mean(0)
                else:  # fix nan centroids
                    new_centroid = X[random.randint(0, X.size(0) - 1)]
                new_centroids.append(new_centroid)
            new_centroids = torch.stack(new_centroids)
            diff = torch.norm(centroids - new_centroids, dim=1).sum()
            if diff < tol:
                break
            centroids = new_centroids
        return centroids, labels, i
    T, P, D = img_feature.shape
    T0 = video_max_frames
    if T <= T0:
        return img_feature, img_similarity, [[[i] for i in range(T)]]
    X = img_feature.view(T, -1)  # [T, P, D]
    centroids, labels, exit_step = kmeans_torch(X, T0)
    reduced_feature = centroids.view(T0, P, D)
    # print(f'Note: perform kmeans feature {img_feature.shape} to {reduced_feature.shape}, exit at step={exit_step}')  # actually, K=T0
    step_indices = [[] for _ in range(T0)]
    for i in range(T0):
        step_indices[i] = [j for j in range(T) if labels[j] == i]
    return reduced_feature, img_similarity, [step_indices]


def weighted_kmeans_feature(img_feature, video_max_frames, weights=None):
    if weights is None:
        weights = torch.ones(img_feature.size(0), dtype=img_feature.dtype, device=img_feature.device)
    def weighted_kmeans_torch(X, num_clusters, weights=None, distance='euclidean', tol=1e-4, max_iter=10):
        indices = torch.randperm(X.size(0), device=X.device)[:num_clusters]
        centroids = X[indices]
        for i in range(max_iter):
            if distance == 'euclidean':
                dists = ((X.unsqueeze(1) - centroids.unsqueeze(0)) ** 2).sum(dim=2).sqrt()
            else:
                raise NotImplementedError("Only Euclidean distance is supported yet")
            labels = torch.argmin(dists, dim=1)
            weighted_sum = torch.zeros_like(centroids)
            weights_sum = torch.zeros(num_clusters, dtype=X.dtype, device=X.device)
            for j in range(num_clusters):
                cluster_mask = labels == j
                weighted_sum[j] = torch.sum(weights[cluster_mask, None] * X[cluster_mask], dim=0)
                weights_sum[j] = torch.sum(weights[cluster_mask])
            mask = weights_sum > 0
            new_centroids = torch.zeros_like(weighted_sum)
            new_centroids[mask] = weighted_sum[mask] / weights_sum[mask, None]
            if mask.sum() < num_clusters:  # fix nan centroids
                new_centroids[~mask] = torch.stack([X[random.randint(0, X.size(0) - 1)] for _ in range(num_clusters - mask.sum())])
            diff = torch.norm(centroids - new_centroids, dim=1).sum()
            if diff < tol:
                break
            centroids = new_centroids
        return centroids, labels, weights_sum, i
    T, P, D = img_feature.shape
    T0 = video_max_frames
    if T <= T0:
        return img_feature, weights, [[[i] for i in range(T)]]
    X = img_feature.view(T, -1)  # [T, P, D]
    centroids, labels, weights, exit_step = weighted_kmeans_torch(X, T0, weights)
    reduced_feature = centroids.view(T0, P, D)
    # print(f'Note: perform weighted kmeans feature {img_feature.shape} to {reduced_feature.shape}, exit at step={exit_step}')  # actually, K=T0
    step_indices = [[] for _ in range(T0)]
    for i in range(T0):
        step_indices[i] = [j for j in range(T) if labels[j] == i]
    return reduced_feature, weights, [step_indices]


def k_drop_feature(img_feature, video_max_frames, img_similarity=None):
    T, P, D = img_feature.shape
    indices = [[i] for i in range(T)]
    T0 = video_max_frames
    if T <= T0:
        return img_feature, img_similarity, [indices]
    cur_feature = img_feature[:T0]  # [T0, P, D]
    normed_cur_features = F.normalize(cur_feature.view(T0, P * D), p=2, dim=1)
    cur_sim = torch.mm(normed_cur_features, normed_cur_features.T)  # [T0, T0]
    cur_sim.fill_diagonal_(-100.0)
    cur_indices = indices[:T0]
    step_indices = [cur_indices]
    for i in range(T0, T):
        # get new feature
        new_feature = img_feature[i]
        normed_new_feature = F.normalize(new_feature.view(1, P * D), p=2, dim=1)
        new_sim = torch.mm(normed_cur_features, normed_new_feature.T)  # [T0, 1]
        all_feature = torch.cat([cur_feature, new_feature.unsqueeze(0)], dim=0)
        normed_all_features = torch.cat([normed_cur_features, normed_new_feature], dim=0)
        all_indices = cur_indices + [[i]]
        # get new similarity
        all_sim_1 = torch.cat([cur_sim, new_sim], dim=1)  # [T0, T0 + 1]
        all_sim = torch.cat([all_sim_1, torch.ones_like(all_sim_1[-1:]) * -100.0], dim=0)  # [T0 + 1, T0 + 1]
        all_sim[-1, :-1] = new_sim.T
        # choose compression position
        idx = torch.argmax(all_sim)
        left, right = idx // (T0 + 1), idx % (T0 + 1)
        if random.randint(0, 1) > 0:
            idx = left
        else:
            idx = right
        assert all_sim[left, right] == torch.max(all_sim)
        # get compressed feature and similarity
        cur_feature = torch.cat([all_feature[:idx], all_feature[idx + 1:]])
        normed_cur_features = torch.cat([normed_all_features[:idx], normed_all_features[idx + 1:]])
        cur_indices = all_indices[:idx] + all_indices[idx + 1:]
        cur_sim_1 = torch.cat([all_sim[:idx], all_sim[idx + 1:]], dim=0)  # [T0, T0 + 1]
        cur_sim = torch.cat([cur_sim_1[:, :idx], cur_sim_1[:, idx + 1:]], dim=1)  # [T0, T0]
        step_indices.append(cur_indices)
    # print(f'Note: perform k-drop feature {img_feature.shape} to {cur_feature.shape}')
    return cur_feature, None, step_indices


def k_merge_feature(img_feature, video_max_frames, img_similarity=None):
    T, P, D = img_feature.shape
    indices = [[i] for i in range(T)]
    T0 = video_max_frames
    if T <= T0:
        return img_feature, img_similarity, [indices]
    cur_feature = img_feature[:T0]  # [T0, P, D]
    normed_cur_features = F.normalize(cur_feature.view(T0, P * D), p=2, dim=1)
    cur_sim = torch.mm(normed_cur_features, normed_cur_features.T)  # [T0, T0]
    cur_sim.fill_diagonal_(-100.0)
    cur_indices = indices[:T0]
    step_indices = [cur_indices]
    for i in range(T0, T):
        # get new feature
        new_feature = img_feature[i]
        normed_new_feature = F.normalize(new_feature.view(1, P * D), p=2, dim=1)
        new_sim = torch.mm(normed_cur_features, normed_new_feature.T)  # [T0, 1]
        all_feature = torch.cat([cur_feature, new_feature.unsqueeze(0)], dim=0)
        normed_all_features = torch.cat([normed_cur_features, normed_new_feature], dim=0)
        all_indices = cur_indices + [[i]]
        # get new similarity
        all_sim_1 = torch.cat([cur_sim, new_sim], dim=1)  # [T0, T0 + 1]
        all_sim = torch.cat([all_sim_1, torch.ones_like(all_sim_1[-1:]) * -100.0], dim=0)  # [T0 + 1, T0 + 1]
        all_sim[-1, :-1] = new_sim.T
        # choose compression position
        idx = torch.argmax(all_sim)
        left, right = idx // (T0 + 1), idx % (T0 + 1)
        assert all_sim[left, right] == torch.max(all_sim)
        # update feature
        all_feature[right] = (all_feature[left] + all_feature[right]) / 2.0
        normed_all_features[right] = F.normalize(all_feature[right].view(1, P * D), p=2, dim=1)
        all_indices[right] = all_indices[left] + all_indices[right]
        # update similarity
        new_sim = torch.mm(normed_all_features, normed_all_features[right:right+1].T)  # [T0 + 1, 1]
        all_sim[right, :] = new_sim.T
        all_sim[:, right:right+1] = new_sim
        all_sim[right, right] = -100.0
        # get compressed feature and similarity
        cur_feature = torch.cat([all_feature[:left], all_feature[left + 1:]])
        normed_cur_features = torch.cat([normed_all_features[:left], normed_all_features[left + 1:]])
        cur_indices = all_indices[:left] + all_indices[left + 1:]
        cur_sim_1 = torch.cat([all_sim[:left], all_sim[left + 1:]], dim=0)  # [T0, T0 + 1]
        cur_sim = torch.cat([cur_sim_1[:, :left], cur_sim_1[:, left + 1:]], dim=1)  # [T0, T0]
        step_indices.append(cur_indices)
    # print(f'Note: perform k-merge feature {img_feature.shape} to {cur_feature.shape}')
    return cur_feature, cur_sim, step_indices


def attention_feature(img_feature, video_max_frames, attention_fn=None, update_ratio=0.2):
    T, P, D = img_feature.shape
    T0 = video_max_frames
    if T <= T0:
        return img_feature, None
    cur_feature = img_feature[:T0]  # [T0, P, D]
    turing_memory = cur_feature.reshape(T0*P, D)  # [T0*P, D]
    for i in range(T0, T, T0):
        j = min(i + T0, T)
        new_feature = img_feature[i:j]  # [P, D]
        new_feature = new_feature.reshape(-1, D)  # [n*P, D]
        turing_memory = attention_fn(turing_memory, new_feature, update_ratio=update_ratio)  # [T0*P, n*P]
    cur_feature = turing_memory.reshape(T0, P, D)
    # print(f'Note: perform {attention_fn.__name__} feature {img_feature.shape} to {cur_feature.shape}')
    return cur_feature, None



