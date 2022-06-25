import pdb
from tkinter.tix import InputOnly
import torch
import torch.nn as nn
from .rnn import RNN
from .cnn import Visual
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


class Detector(nn.Module):
    def __init__(self, modality=(0, 1, 2, 3), dropout_ratio=0.5, input_dim=512, hidden_dim=512, num_classes=61,
                 dropout=0.3, level=2):
        super().__init__()
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

        # self.dim_decrease = nn.Sequential(
        #     nn.Linear(6400, 2048),
        #     LayerNorm(2048),
        #     nn.ReLU(),
        #     nn.Linear(2048, 512),
        #     LayerNorm(512),
        #     nn.ReLU(inplace=True)
        # )
        # print('layer_norm')
        self.fusion = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )

        self.bilstm = RNN(
            self.input_dim,
            self.hidden_dim,
            2,
            self.dropout,
            True,
            'LSTM'
        )

        self.transformer = transformer_model(
            self.input_dim,
            self.dropout
        )

        # self.cnn = Visual(
        #     self.input_dim,
        #     self.hidden_dim
        # )
        self.cnn = nn.Sequential(
            nn.Conv1d(self.input_dim, self.hidden_dim, 3, 1, 1),
            LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, 3, 1, 1),
            LayerNorm(self.hidden_dim),
            nn.ReLU(),
            # nn.Conv1d(self.hidden_dim, self.hidden_dim, 3, 1, 1),
            # LayerNorm(self.hidden_dim),
            # nn.ReLU()
        )

        # self.branch = nn.ModuleList()
        # for i in range(self.level):
        #     self.branch.append(
        #         nn.Sequential(
        #             nn.Conv1d(self.hidden_dim, self.hidden_dim, 3, 1, 1),
        #             LayerNorm(),
        #             nn.GELU(),
        #             nn.MaxPool1d(3, 2, 1)
        #         )
        #     )

        # self.fpn = FPN_1d(self.hidden_dim, 2, self.level)

        self.cls_head = nn.Sequential(
            nn.Conv1d(self.hidden_dim, 256, 3, 1, 1),
            LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, 3, 1, 1),
            LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, 3, 1, 1),
            LayerNorm(256),
            nn.ReLU(inplace=True)
        )

        self.classifer = nn.Conv1d(256, self.num_class, 3, 1, 1)

        self.reg_head = nn.Sequential(
            nn.Conv1d(self.hidden_dim, 256, 3, 1, 1),
            LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, 3, 1, 1),
            LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, 3, 1, 1),
            LayerNorm(256),
            nn.ReLU(inplace=True)
        )

        self.reg_predictor = nn.Sequential(
            nn.Conv1d(256, 2, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.dp = nn.Dropout(dropout_ratio)
        # self.reg_predictor = nn.Sequential(
        #     nn.Linear(256, 2),
        #     nn.ReLU()
        # )

    def forward(self, vid, len_x):
        # x [b,t,c] len_x [b]
        batch, max_len, dim = vid.size()
        # vid = self.dim_decrease(vid.view(batch*max_len, dim)).view(batch,max_len,-1)
        # x = self.dim_decrease(vid.view(batch * max_len, dim)[:, self.modality_index]).view(batch, max_len, -1)
        x = self.dim_decrease(vid.view(batch * max_len, dim)[:, self.modality_index]).view(batch, max_len, -1)
        # x = self.dim_decrease(vid)
        # x = torch.cat((vid, skeleton), dim=-1)
        # x = self.fusion(x.view(batch*max_len, -1)).view(batch, max_len, -1)

        backbone_feat1 = self.bilstm(x.permute(1, 0, 2), len_x)['contextual_feat'].permute(1, 0, 2)  # [b,t,c]
        backbone_feat2 = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)  # [b,t,c]
        backbone_feat = self.fusion(
            torch.cat((backbone_feat1, backbone_feat2), dim=-1).view(batch * max_len, 1024)).view(batch, max_len, -1)
        # backbone_feat3 = self.transformer(x, len_x, 'feat')['transformer_feat']  # [b,t,c]
        # backbone_feat = self.fusion(
        #     torch.cat((backbone_feat1, backbone_feat2, backbone_feat3), dim=-1).view(batch * max_len, 512 * 3)).view(
        #     batch, max_len, -1)
        backbone_feat = self.dp(backbone_feat)
        # batch, max_len, dim = backbone_feat.size()
        cls_logits = []
        reg_results = []
        for i in range(batch):
            feat_single = backbone_feat[i, :len_x[i]][None].permute(0, 2, 1)  # [1,t,c]
            cls_result = self.cls_head(feat_single)  # [1,c,t]
            cls_logits.append(self.classifer(cls_result)[0].permute(1, 0))
            reg_result = self.reg_head(feat_single)
            reg_results.append(self.reg_predictor(reg_result)[0].permute(1, 0))
        return torch.cat(cls_logits, dim=0), torch.cat(reg_results, dim=0)

        # for i in range(batch):
        #     feat_single = backbone_feat[i,:len_x[i]][None].permute(0,2,1) # [1,t,c]
        #     cls_result = self.cls_head(feat_single) # [1,c,t]
        #     cls_logits.append(self.classifer(cls_result[0].permute(1,0)))
        #     reg_result = self.reg_head(feat_single)
        #     reg_results.append(self.reg_predictor(reg_result[0].permute(1,0)))
        # return torch.cat(cls_logits, dim=0), torch.cat(reg_results,dim=0)


if __name__ == '__main__':
    layer = Detector()
    vid = torch.rand((2, 60, 2048))
    skeleton = torch.rand((2, 60, 256))
    len_x = torch.tensor([60, 40])
    a, b = layer(vid, skeleton, len_x)
    print(a[0], b[0])
