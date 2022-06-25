import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def single_batch_forward(self, x, target):
        """
        :param x: [T, C]
        :param target: [T]
        :return:
        """
        target = target.type(torch.int64)
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.sum()
    
    def forward(self, logits, label, len_logits):
        loss, batch_size = 0, logits.size(0)
        # criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
        for batch_idx in range(logits.size(0)):
            label_single = label[batch_idx, :len_logits[batch_idx]]
            logit = logits[batch_idx, :len_logits[batch_idx], :]
            loss += self.single_batch_forward(logit, label_single)
        loss /= batch_size
        return loss


def LabelSmoothingSegmentLoss(logits, label, len_logits, smoothing=0.0):
    loss, batch_size = 0, logits.size(0)
    criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
    for batch_idx in range(logits.size(1)):
        label_single = label[batch_idx, :len_logits[batch_idx]]
        logit = logits[batch_idx, :len_logits[batch_idx], :]
        loss += criterion(logit, label_single)
    loss /= batch_size
    return loss
