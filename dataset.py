import torch
from torch.utils.data import Dataset
from sklearn.metrics.pairwise import euclidean_distances
from dtw import dtw
import numpy as np

def channel_standardization(local_features):
    # 沿着通道维度计算均值和标准差
    mean = np.mean(local_features, axis=0)
    std = np.std(local_features, axis=0)
    # 对每个局部特征进行标准化
    local_features = (local_features - mean) / std
    return local_features

# 3. 使用DTW进行时间上的扭曲和对齐
def warp_and_align(local_features1, local_features2):
    # 计算两个局部特征序列之间的距离矩阵
    dist_matrix = euclidean_distances(local_features1, local_features2)
    # 进行DTW对齐
    dist_fun = lambda x, y: np.linalg.norm(x - y)
    dist, cost, acc, path = dtw(dist_matrix, dist_fun)
    # 根据DTW对齐路径得到对齐后的局部特征序列
    warped_local_features1 = local_features1[path[0]]
    warped_local_features2 = local_features2[path[1]]
    return warped_local_features1, warped_local_features2


def resize_and_standardization(warped_local_features1, warped_local_features2, length=1024):
    # 将扭曲的特征序列调整到指定长度
    warped_local_features1 = np.resize(warped_local_features1, (length, warped_local_features1.shape[1]))
    warped_local_features2 = np.resize(warped_local_features2, (length, warped_local_features2.shape[1]))
    # 进行另一次标准化
    warped_local_features1 = channel_standardization(warped_local_features1)
    warped_local_features2 = channel_standardization(warped_local_features2)
    return warped_local_features1, warped_local_features2

class SignatureDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        # 获取一个签名的笔坐标
        x = self.signatures[idx][0]#size(7,24)
        y = self.signatures[idx][1]#size(7,24)

        # 2. 通道标准化
        x = channel_standardization(x)
        y = channel_standardization(y)
        # 3. 使用DTW进行时间上的扭曲和对齐
        x, y = warp_and_align(x, y)
        # 4. 将扭曲的特征序列调整到预定义长度，并进行另一次标准化
        x, y = resize_and_standardization(x, y)

        return x, y

    def __len__(self):
        return len(self.data)

#定义collate_fn加载到dataloader
def collate_fn(batch):
    x_batch, y_batch = [], []
    for x, y in batch:
        x_batch.append(x)
        y_batch.append(y)


    # 转换为PyTorch tensor并移动到GPU
    x_batch = torch.from_numpy(x_batch).float().cuda()
    y_batch = torch.from_numpy(y_batch).float().cuda()

    return x_batch, y_batch
