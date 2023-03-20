import numpy as np
from scipy.spatial.distance import cdist
import torch
import torch.nn as nn
import torch.nn.functional as F
from dtw import dtw


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(7, 64, kernel_size=7, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = F.normalize(x, p=2, dim=1)
        return x

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.extracor = FeatureExtractor()
    def forward(self, x, y):

        x = self.extracor(x)
        y = self.extracor(y)
        dist = F.pairwise_distance(x, y)
        return dist
