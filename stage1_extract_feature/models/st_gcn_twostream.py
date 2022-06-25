import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from stage1_extract_feature.models.utils.tgcn import ConvTemporalGraphical
from stage1_extract_feature.models.utils.graph import Graph

from .st_gcn import Model as ST_GCN


class Model(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.origin_stream = ST_GCN(*args, **kwargs)
        self.motion_stream = ST_GCN(*args, **kwargs)

    def forward(self, x):
        N, C, T, V, M = x.size()
        m = torch.cat((torch.cuda.FloatTensor(N, C, 1, V, M).zero_(),
                       x[:, :, 1:-1] - 0.5 * x[:, :, 2:] - 0.5 * x[:, :, :-2],
                       torch.cuda.FloatTensor(N, C, 1, V, M).zero_()), 2)

        res = self.origin_stream(x) + self.motion_stream(m)
        return res

    def extract_feature(self, x):
        N, C, T, V, M = x.size()
        m = torch.cat((torch.cuda.FloatTensor(N, C, 1, V, M).zero_(),
                       x[:, :, 1:-1] - 0.5 * x[:, :, 2:] - 0.5 * x[:, :, :-2],
                       torch.cuda.FloatTensor(N, C, 1, V, M).zero_()), 2)

        logits_x, feats_x = self.origin_stream.extract_feature(x)
        logits_m, feats_m = self.origin_stream.extract_feature(m)
        return logits_x + logits_m, feats_x + feats_m
