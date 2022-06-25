import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from stage1_extract_feature.models.utils.tgcn import ConvTemporalGraphical
from stage1_extract_feature.models.utils.graph import Graph
from stage1_extract_feature.models.utils.graph import Graph
from stage1_extract_feature.models.st_gcn import Model
from stage1_extract_feature.models.BiLSTM import BiLSTMLayer


class SeqSTGCN(nn.Module):
    def __init__(self, in_channels, num_classes, graph_args, edge_importance_weighting,
                 temporal_kernel_size=9, stride=4, duration=8, **kwargs):
        super().__init__()
        self.stride = stride
        self.duration = duration
        self.st_gcn = Model(
            in_channels, num_classes, graph_args, edge_importance_weighting,
            temporal_kernel_size, **kwargs
        )

        self.lstm = BiLSTMLayer(256, hidden_size=512, num_layers=2, dropout=0.3)

        self.tconv = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=(5,), padding=2),
            # nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=(5,), padding=2),
            # nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=512, out_channels=num_classes, kernel_size=(1,), padding=0),
        )
        self.fc1 = nn.Linear(256, num_classes)
        # self.fc2 = nn.Linear(512, num_class)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, video=False):
        if video:
            return self.forward_video(x)
        # gcn_logits = self.st_gcn(x)[:, None]
        _, feats = self.st_gcn.extract_feature(x)
        feats = F.avg_pool2d(feats.squeeze(-1), (1, 57)).squeeze(-1).permute(2, 0, 1)
        feats = feats.mean(dim=0)[None]
        gcn_logits = self.fc1(feats)
        return {
            'conv_predictions': gcn_logits.permute(1, 0, 2).contiguous(),
            'conv_feats': feats.permute(1, 2, 0),
            'lstm_predictions': gcn_logits.permute(1, 0, 2).contiguous(),
        }

    def forward_video(self, x):
        # inputs = inputs[:, :inputs.shape[1] // 32 * 32]
        b, t, c, h, w = x.shape
        left = self.duration // 2
        right = t // self.stride * self.stride + self.duration - t
        x_pad = torch.zeros(left + t + right, c, h, w, device=x.device)
        x_pad[left:left + t] = x[0]
        split_list = []
        len_list = list(range(left + t + right))
        for t in range(left, left + t, self.stride):
            l, r = t - self.duration // 2, t + self.duration // 2
            split_list.append(len_list[l:r])
        x_list = x_pad[split_list]  # [temporal//stride+1, duration, c, h, w]
        x_list = x_list.transpose(1, 2)
        temporal = x_list.size(0)
        batch_size = 64
        pred = []
        feats = []
        for i in range(temporal // batch_size + 1):
            if len(x_list[i * batch_size:(i + 1) * batch_size]) > 0:
                ret = self.forward(x_list[i * batch_size:(i + 1) * batch_size])
                pred.append(ret['conv_predictions'][:, 0])
                feats.append(ret['conv_feats'])
        return torch.cat(pred, dim=0), torch.cat(feats, dim=0)  # [:, 0]
