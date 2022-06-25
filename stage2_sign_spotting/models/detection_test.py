import pdb
from tkinter.tix import InputOnly
import torch
import torch.nn as nn
from .rnn import RNN
from .cnn import Visual
import torch.nn.functional as F
from .transformer import transformer_model


class LayerNorm(nn.Module):
    """
    LayerNorm that supports inputs of size B, C, T
    """

    def __init__(
            self,
            num_channels,
            eps=1e-5,
            affine=True,
            device=None,
            dtype=None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones([1, num_channels, 1], **factory_kwargs))
            self.bias = nn.Parameter(
                torch.zeros([1, num_channels, 1], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        assert x.dim() == 3
        assert x.shape[1] == self.num_channels

        # normalization along C channels
        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x ** 2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)

        # apply weight and bias
        if self.affine:
            out *= self.weight
            out += self.bias

        return out


class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs


class TemporalModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super(TemporalModule, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.bilstm = RNN(self.input_dim, self.hidden_dim, 2, self.dropout, True, 'LSTM')
        self.transformer = transformer_model(self.input_dim, self.dropout)
        self.cnn = nn.Sequential(
            nn.Conv1d(self.input_dim, self.hidden_dim, 3, 1, 1),
            LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, 3, 1, 1),
            LayerNorm(self.hidden_dim),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, lgt):
        b, t, c = x.shape
        backbone_feat1 = self.bilstm(x.permute(1, 0, 2), lgt)['contextual_feat'].permute(1, 0, 2)
        backbone_feat2 = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)  # [b,t,c]
        backbone_feat = self.fusion(torch.cat((backbone_feat1, backbone_feat2), dim=-1).
                                    view(b * t, 1024)).view(b, t, -1)
        return backbone_feat


class Head(nn.Module):
    def __init__(self, input_dim, output_dim, relu_flag):
        super(Head, self).__init__()
        self.relu_flag = relu_flag
        self.cnn_head = nn.Sequential(
            nn.Conv1d(input_dim, 256, 3, 1, 1),
            LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, 3, 1, 1),
            LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, 3, 1, 1),
            LayerNorm(256),
            nn.ReLU(inplace=True)
        )
        self.classifer = nn.Conv1d(256, output_dim, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.cnn_head(x)
        x = self.classifer(x)[0].T
        if self.relu_flag:
            x = self.relu(x)
        return x


class Detector(nn.Module):
    def __init__(self, modality=(0, 1, 2, 3), dropout_ratio=0.5, input_dim=512, hidden_dim=512, num_classes=61,
                 dropout=0.3, level=2):
        super(Detector, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_class = num_classes
        self.dropout = dropout
        self.level = level
        feat_dims = [2048, 2048, 2048, 256]
        self.modality_index = []
        print(modality)
        for mod in modality:
            self.modality_index += [i + sum(feat_dims[:mod]) for i in range(feat_dims[mod])]
        self.dim_decrease = nn.Sequential(
            nn.Linear(len(self.modality_index), 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        self.temporal_module = TemporalModule(512, self.hidden_dim, self.dropout)
        self.cls_head = Head(self.hidden_dim, self.num_class, relu_flag=False)
        self.reg_head = Head(self.hidden_dim, 3, relu_flag=True)
        self.dp = nn.Dropout(dropout_ratio)

    def forward(self, vid, len_x):
        # x [b,t,c] len_x [b]
        batch, max_len, dim = vid.size()
        feats = self.dim_decrease(vid.view(batch * max_len, dim)[:, self.modality_index]).view(batch, max_len, -1)
        fused_feats = self.temporal_module(feats, len_x)
        fused_feats = self.dp(fused_feats)
        cls_logits = []
        reg_results = []
        for i in range(batch):
            feat_single = fused_feats[i, :len_x[i]][None].permute(0, 2, 1)  # [1,t,c]
            cls_logits.append(self.cls_head(feat_single))
            reg_results.append(self.reg_head(feat_single))
        return torch.cat(cls_logits, dim=0), torch.cat(reg_results, dim=0)


if __name__ == '__main__':
    layer = Detector()
    vid = torch.rand((2, 60, 2048))
    skeleton = torch.rand((2, 60, 256))
    len_x = torch.tensor([60, 40])
    a, b = layer(vid, skeleton, len_x)
    print(a[0], b[0])
