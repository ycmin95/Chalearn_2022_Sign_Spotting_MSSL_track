import torch
import torch.nn as nn

from .losses import FocalLoss
from .losses import CrossEntropyLabelSmooth


def get_criterion():
    criterion = dict()
    criterion['ce_loss'] = nn.CrossEntropyLoss().cuda()
    criterion['focal_loss'] = FocalLoss(gamma=2, alpha=0.25, size_average=True).cuda()
    criterion['label_smooth'] = CrossEntropyLabelSmooth().cuda()
    return criterion
