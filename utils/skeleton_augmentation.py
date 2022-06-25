# ----------------------------------------
# Written by Yuecong Min
# ----------------------------------------
import cv2
import pdb
import PIL
import copy
import scipy.misc
import torch
import random
import numbers
import numpy as np


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, skeleton):
        for t in self.transforms:
            skeleton = t(skeleton)
        return skeleton


class ToTensor(object):
    def __call__(self, skeleton):
        if isinstance(skeleton, np.ndarray):
            skeleton = torch.from_numpy(skeleton)
        return skeleton


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, skeleton):
        # B, H, W, 3
        flag = random.random() < self.prob
        if flag:
            skeleton[:, :, 0] *= -1
            swap_ind = [0, 4, 5, 6, 1, 2, 3, 8, 7, 10, 9, 12, 11, 14, 13] + \
                       [i + 36 for i in range(21)] + [i + 15 for i in range(21)]
            skeleton = skeleton[:, swap_ind]
            # pdb.set_trace()
        return skeleton


class RandomResize(object):
    def __init__(self, rate):
        self.rate = rate

    def __call__(self, skeleton):
        scale = random.uniform(1 - self.rate, 1 + self.rate)
        # B, H, W, 3
        return skeleton * scale


class RandomShift(object):
    def __init__(self, rate):
        self.rate = rate

    def __call__(self, skeleton):
        x_scale = random.uniform(-self.rate, self.rate)
        y_scale = random.uniform(-self.rate, self.rate)
        z_scale = random.uniform(-self.rate, self.rate)
        skeleton[:, :, 0] += x_scale
        skeleton[:, :, 1] += y_scale
        skeleton[:, :, 2] += z_scale
        return skeleton
