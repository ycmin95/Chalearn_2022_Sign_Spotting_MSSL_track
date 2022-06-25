import os
import pdb
import glob
import tqdm
import random
import pickle
import numpy as np
from PIL import Image
from turtle import right
from collections import Counter
from collections import defaultdict

import torch
import torch.utils.data as data


class MSSLFeatureClipFeeder(data.Dataset):
    def __init__(self, data_type='video', mode='train', remove_bg=False, bg_class=0, final_flag=False,
                 stride=4, clip_duration=8, seq_duration=8, feature_dir="features", transform_mode='train'):
        self.prefix = './dataset/MSSL_dataset'
        self.data = dict()
        self.data_type = data_type
        self.mode = mode
        self.labels = []
        self.stride = stride
        self.remove_bg = remove_bg
        self.feature_dir = feature_dir
        self.bg_class = bg_class
        self.clip_duration = clip_duration
        self.seq_duration = seq_duration
        if final_flag and self.mode == 'train':
            file_path = 'final_train_input.txt'
        else:
            file_path = f'{self.mode}_input.txt'
        self._read_file(os.path.join(self.prefix, file_path))

    def _read_file(self, path):
        idx = 0
        lines = open(path, "r").readlines()
        for line in lines:
            video_file_name, _, start_frame, end_frame = line[:-1].split(' | ')
            video_file_name = video_file_name.rsplit("/")[-1]
            start_frame = int(start_frame)
            end_frame = int(end_frame)
            mode = "train" if "train" in line else "valid"
            self.data[idx] = (video_file_name, mode, start_frame, end_frame)
            idx += 1

    def __getitem__(self, idx):
        video_file_name, mode, start, end = self.data[idx]
        feature, label = self.read_fusion_feature(video_file_name, mode)
        if self.mode == "test":
            offset = 1
        else:
            offset = generate_offset(label)
        return feature, label, offset, self.data[idx]

    def __len__(self):
        return len(list(self.data.keys()))

    def read_fusion_feature(self, video_file_name, mode):
        features = []
        if self.mode == "test":
            labels = 1
        else:
            labels = torch.from_numpy(
                np.load(f"{self.prefix}/processed/{mode}/clipwise_label/{video_file_name}.npy")
            ).long().squeeze()
        for feature_type in ['video', 'mask_video', 'flow', 'skeleton']:
            feature_path = f"{self.prefix}/processed/{self.feature_dir}/{feature_type}/{video_file_name}.npy"
            features.append(torch.from_numpy(np.load(feature_path)))
        if len(features[2]) < len(features[0]):
            features[2] = torch.cat((features[2],
                                     torch.zeros((len(features[0]) - len(features[2]),) + features[2].size()[1:],
                                                 dtype=features[2].dtype)), dim=0)
        return torch.cat(features, dim=-1), labels

    @staticmethod
    def collate_fn(batch):
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        feature, label, offset, info = list(zip(*batch))

        max_len = len(feature[0])
        padding_size = feature[0].shape[1:]
        video_length = torch.LongTensor([len(vid) for vid in feature])
        padded_video = [torch.cat(
            (
                vid,
                torch.zeros((max_len - len(vid),) + padding_size),
            )
            , dim=0)
            for vid in feature]
        padded_video = torch.stack(padded_video)

        padded_label = torch.cat(label, dim=0)
        padded_offset = torch.cat(offset, dim=0)

        if max(video_length) == 0:
            return padded_video, video_length, [], [], info
        else:
            return padded_video, video_length, padded_label, padded_offset, info


def generate_offset(label):
    # label [t]
    length = len(label)
    offset = np.zeros((length, 2))
    action_label, action_start, action_end = 0, 0, 0
    for i in range(length):
        if label[i] == 0:
            if action_label == 0:
                continue
            else:
                action_end = i - 1
                offset[action_start:i, 1] = action_end - np.array(list(range(action_start, i)))
                action_label = label[i]
        else:
            if action_label == 0:
                action_label = label[i]
                action_start = i
            else:
                if label[i] == action_label:
                    offset[i, 0] = i - action_start
                else:
                    action_end = i - 1
                    offset[action_start:i, 1] = action_end - np.array(list(range(action_start, i)))
                    action_label = label[i]
                    action_start = i
    if not action_label == 0:
        action_end = length - 1
        offset[action_start:i, 1] = action_end - np.array(list(range(action_start, i)))
    return torch.from_numpy(offset).clamp_(min=0, max=4)


if __name__ == '__main__':
    batch_size = 2
    train_flag = True

    dataloader = torch.utils.data.DataLoader(
        dataset=MSSLFeatureClipFeeder(mode='valid', data_type='fusion_feature', seq_duration=32),
        batch_size=batch_size,
        shuffle=False,
        drop_last=train_flag,
        num_workers=0,
        collate_fn=MSSLFeatureClipFeeder.collate_fn,
        pin_memory=False
    )

    for batch_idx, data in enumerate(dataloader):
        feature, lgt, label, offset, info = data
        pdb.set_trace()
    print(batch_idx)
