# import numpy as np
# import torch

# float_array = np.random.uniform(low=-5.2, high=16.0, size=(50,))

# min_val = float_array.min()
# max_val = float_array.max()

# max_int = len(float_array) - 1
# normalized_array = ((float_array - min_val) / (max_val - min_val)) * max_int

# int_array = normalized_array.astype(int)

# embedding_tensor = torch.tensor(int_array)

# print("float:", float_array)
# print("int:", int_array)
# print("long:", embedding_tensor)
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

def farthest_point_sampling(xyz, npoint):
    N, C = xyz.shape
    centroids = torch.zeros(npoint, dtype=torch.long)
    distance = torch.ones(N) * 1e10
    farthest = torch.randint(0, N, (1,), dtype=torch.long)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :].view(1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.argmax(distance)
    return centroids

def knn(xyz, k):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(xyz)
    distances, indices = nbrs.kneighbors(xyz)
    return indices

class PointNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PointNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        npoint, k, input_dim = x.shape
        x = x.view(npoint * k, input_dim)
        x = self.mlp(x)
        x = x.view(npoint, k, -1)
        return x

class SpatialNeighborEmbedding(nn.Module):
    def __init__(self, npoint, k, input_dim, output_dim):
        super(SpatialNeighborEmbedding, self).__init__()
        self.npoint = npoint
        self.k = k
        self.pointnet = PointNet(input_dim, output_dim)

    def forward(self, xyz, features):
        B, N, _ = xyz.shape

        new_xyz = []
        new_features = []

        for b in range(B):
            fps_idx = farthest_point_sampling(xyz[b], self.npoint)
            sampled_xyz = xyz[b][fps_idx]
            knn_idx = knn(xyz[b].detach().cpu().numpy(), self.k)

            group_xyz = xyz[b][knn_idx]
            group_features = features[b][knn_idx]

            group_features = torch.cat([group_xyz, group_features], dim=-1)
            # print(group_features.shape)
            # group_features = group_features.permute(1024, 16, 2, 5)  # (npoint, k, B, input_dim)
            new_feature = self.pointnet(group_features)
            new_feature = F.max_pool2d(new_feature.unsqueeze(0), kernel_size=[1, new_feature.size(-1)]).squeeze(-1).squeeze(0)

            new_features.append(new_feature)
            new_xyz.append(sampled_xyz)

        new_xyz = torch.stack(new_xyz)
        new_features = torch.stack(new_features)

        return new_xyz, new_features

# 測試模塊
B = 256  # batch size
N = 64  # number of points
C = 3  # XYZ dimensions
F_ = 2  # feature dimensions
npoint = 32  # number of sampled points 16, 8
k = 32  # number of neighbors 16, 8

xyz = torch.rand(B, N, C)
features = torch.rand(B, N, F_)
print("Points Shape:", xyz.shape)
print("Features Shape:", features.shape)

spatial_neighbor_embedding = SpatialNeighborEmbedding(npoint, k, F_ + C, 256)
new_xyz, new_features = spatial_neighbor_embedding(xyz, features)

print("Sampled Points Shape:", new_xyz.shape)
print("New Features Shape:", new_features.shape)
