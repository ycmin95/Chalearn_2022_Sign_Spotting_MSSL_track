import pdb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from stage1_extract_feature.models import resnet
from stage1_extract_feature.models import r2plus1d
from stage1_extract_feature.models import p3d_model
from stage1_extract_feature.models import x3d


# from lib.baseline.extract_feature.models import BiLSTM


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class TModel(nn.Module):
    def __init__(self, stride=4, duration=8, backbone="i3d", hidden_size=512, *args, **kargs):
        self.stride = stride
        self.duration = duration
        self.backbone = backbone
        self.num_classes = kargs['num_classes']
        super(TModel, self).__init__()
        if backbone == "i3d":
            self.feature_extractor = resnet.i3_res50(**kargs)
            weights = torch.load("/data/ycmin/share/i3d_r50_kinetics.pth")
            weights.pop('fc.weight')
            weights.pop('fc.bias')
            self.feature_extractor.load_state_dict(weights, strict=False)
            feat_dim = self.feature_extractor.fc.in_features
        elif backbone == "i3d_nl":
            self.feature_extractor = resnet.i3_res50_nl(**kargs)
            weights = torch.load("/data/ycmin/share/i3d_r50_nl_kinetics.pth")
            weights.pop('fc.weight')
            weights.pop('fc.bias')
            self.feature_extractor.load_state_dict(weights, strict=False)
            feat_dim = self.feature_extractor.fc.in_features
        elif backbone == "p3d":
            self.feature_extractor = p3d_model.P3D199(**kargs)
            weights = torch.load("/data/ycmin/share/p3d_rgb_199.checkpoint.pth.tar")['state_dict']
            weights.pop('fc.weight')
            weights.pop('fc.bias')
            self.feature_extractor.load_state_dict(weights, strict=False)
            feat_dim = self.feature_extractor.fc.in_features
        elif backbone == "r21d":
            self.feature_extractor = r2plus1d.r2plus1d_r50(model_num_class=self.num_classes)
            feat_dim = self.feature_extractor.blocks[5].proj.in_features
        elif backbone == "x3d":
            self.feature_extractor = x3d.x3d_m(model_num_class=self.num_classes)
            feat_dim = self.feature_extractor.blocks[5].proj.in_features

        self.conv = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs, video=False):
        if video:
            return self.forward_video(inputs)
        inputs_shape = inputs.shape
        temporal_lgt = inputs.shape[2]
        reshaped_inputs = inputs.view(inputs_shape[0], inputs_shape[1], temporal_lgt // self.duration,
                                      self.duration, inputs.shape[3], inputs.shape[4])
        reshaped_inputs_shape = reshaped_inputs.shape

        reshaped_inputs = reshaped_inputs.permute(0, 2, 1, 3, 4, 5).reshape(
            (-1, reshaped_inputs_shape[1], reshaped_inputs_shape[3], reshaped_inputs_shape[4],
             reshaped_inputs_shape[5])
        )
        ret_dict = self.feature_extractor(reshaped_inputs)
        feats = ret_dict['sequence_features']. \
            view(reshaped_inputs_shape[0], reshaped_inputs_shape[2], -1).permute(0, 2, 1)
        # logits = self.conv(feats).permute(0, 2, 1)[:, 0]

        if self.backbone in ["r21d", "x3d"]:
            logits = self.feature_extractor.blocks[5].proj(self.conv(feats).permute(0, 2, 1))
        else:
            logits = self.feature_extractor.fc(self.conv(feats).permute(0, 2, 1))
        # lstm_ret = self.lstm(feats.permute(1, 0, 2), [feats.shape[1]] * feats.shape[0])
        return {
            'conv_predictions': ret_dict['classify_logits'].view(reshaped_inputs_shape[0], -1, self.num_classes),
            'conv_feats': feats,
            'lstm_predictions': logits.contiguous(),
        }

    def forward_video(self, inputs):
        # inputs = inputs[:, :inputs.shape[1] // 32 * 32]
        b, t, c, h, w = inputs.shape
        left = self.duration // 2
        right = t // self.stride * self.stride + self.duration // 2 - t
        x_pad = torch.zeros(left + t + right, c, h, w, device=inputs.device)
        x_pad[left:left + t] = inputs[0]
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
                pred.append(ret['conv_predictions'])
                feats.append(ret['conv_feats'])
        return torch.cat(pred, dim=0), torch.cat(feats, dim=0)
        # pdb.set_trace()
        # reshaped_inputs = inputs.view(b, t // self.duration, self.duration, c, h, w)
        # reshaped_inputs = reshaped_inputs.reshape((-1, self.duration, c, h, w)).permute(0, 2, 1, 3, 4)
        # ret_dict = self.feature_extractor(reshaped_inputs)
        # feats = ret_dict['sequence_features'][None].permute(0, 2, 1)
        # logits = self.conv(feats)
        # # lstm_pred = []
        # # for i in range(inputs_shape[0] // 60):
        # #     lstm_ret = self.lstm(feats[i * 15:(i + 1) * 15, None], [15])
        # #     lstm_pred.append(self.lstm_fc(lstm_ret['predictions']).permute(1, 0, 2))
        # # lstm_pred = torch.cat(lstm_pred, dim=1)
        # # return lstm_pred
        # return ret_dict['classify_logits']
        # # return logits[0].T
