from cProfile import label
from turtle import forward
import pdb
import torch
import torch.nn as nn

from .losses import LabelSmoothingSegmentLoss, LabelSmoothingCrossEntropy, FocalLoss, sigmoidFocalLoss


def unpack_padded_seqence(x, len_x):
    b, t, c = x.size()
    logits = []
    for i in range(b):
        logits.append(x[i, 0:len_x[i], :])
    logits = torch.cat(logits, dim=0)
    return logits


class crossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.celoss = nn.CrossEntropyLoss()
        self.nllloss = nn.NLLLoss(reduction='mean')
        self.scale = 64

    def forward(self, x, label):
        # x: b * t * c
        b, t, c = x.size()
        x = x.contiguous().view(b * t, -1)
        label = label.view(b * t)
        return self.celoss(x, label)


class iou_loss(nn.Module):
    def __init__(self, reduction):
        super().__init__()
        self.reduction = reduction
        self.eps = 1e-8

    def forward(self, input_offsets, target_offsets):
        input_offsets = input_offsets.float()
        target_offsets = target_offsets.float()
        # check all 1D events are valid
        assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
        assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

        lp, rp = input_offsets[:, 0], input_offsets[:, 1]
        lg, rg = target_offsets[:, 0], target_offsets[:, 1]

        # intersection key points
        lkis = torch.min(lp, lg)
        rkis = torch.min(rp, rg)

        # iou
        intsctk = rkis + lkis
        unionk = (lp + rp) + (lg + rg) - intsctk
        iouk = intsctk / unionk.clamp(min=self.eps)

        # smallest enclosing box
        lc = torch.max(lp, lg)
        rc = torch.max(rp, rg)
        len_c = lc + rc

        # offset between centers
        rho = 0.5 * (rp - lp - rg + lg)

        # diou
        loss = 1.0 - iouk + torch.square(rho / len_c.clamp(min=self.eps))

        if self.reduction == "mean":
            loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class g_iou_loss(nn.Module):
    def __init__(self, reduction):
        super().__init__()
        self.reduction = reduction
        self.eps = 1e-8

    def forward(self, input_offsets, target_offsets):
        input_offsets = input_offsets.float()
        target_offsets = target_offsets.float()
        # check all 1D events are valid
        assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
        assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

        lp, rp = input_offsets[:, 0], input_offsets[:, 1]
        lg, rg = target_offsets[:, 0], target_offsets[:, 1]

        # intersection key points
        lkis = torch.min(lp, lg)
        rkis = torch.min(rp, rg)

        # iou
        intsctk = rkis + lkis
        unionk = (lp + rp) + (lg + rg) - intsctk
        iouk = intsctk / unionk.clamp(min=self.eps)

        # giou is reduced to iou in our setting, skip unnecessary steps
        loss = 1.0 - iouk

        if self.reduction == "mean":
            loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class detector_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='sum')
        # self.ce_loss = LabelSmoothingCrossEntropy()
        self.iou_loss = iou_loss(reduction='sum')
        self.lamb = 1

    def forward(self, cls_logits, reg_result, label, offset, len_x):
        cls_loss = self.ce_loss(cls_logits, label)
        positive_sample_index = torch.nonzero(label).squeeze()
        reg_loss = self.iou_loss(reg_result[positive_sample_index, :], offset[positive_sample_index, :])
        return cls_loss / len(label) + self.lamb * reg_loss / max(len(positive_sample_index), 1)


def get_criterion():
    criterion = dict()
    criterion['segment_loss'] = LabelSmoothingCrossEntropy().cuda()
    criterion['ce_loss'] = crossEntropyLoss().cuda()
    # criterion['iou_loss'] = iou_loss().cuda()
    criterion['detector_loss'] = detector_loss().cuda()
    return criterion


if __name__ == '__main__':
    # loss = crossEntropyLoss_sequence('sum')
    loss = iou_loss('sum')
    a = torch.zeros((10, 2))
    b = torch.zeros((10, 2))
    print(loss(a, b))
