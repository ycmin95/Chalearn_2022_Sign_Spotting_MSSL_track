import os
import pdb
import glob
import tqdm
import copy
import pickle
import numpy as np
from PIL import Image
from collections import Counter
from collections import defaultdict

import torch
import torch.utils.data as data

from utils import video_augmentation
from utils import skeleton_augmentation


# import video_augmentation
# from sampler_mssl_clip import ImbalancedDatasetSampler


class MSSLClipFeeder(data.Dataset):
    def __init__(self, data_type='video', mode='train', transform_mode=True, remove_bg=False, bg_class=0,
                 mask=False, stride=4, clip_duration=8, seq_duration=8, sampling_rate=1):
        # create soft link in ./dataset
        self.prefix = './dataset/MSSL_dataset'
        self.data = {}
        self.mode = mode
        self.mask = mask
        self.stride = stride
        self.bg_class = bg_class
        self.data_type = data_type
        self.remove_bg = remove_bg
        self.sampling_rate = sampling_rate
        self.seq_duration = seq_duration
        self.clip_duration = clip_duration
        self.labels = []
        self.transform_mode = 'train' if transform_mode else 'test'
        file_path = f"{mode}_input.txt"
        print(file_path)
        self.data_aug = self.transform()
        self.skeleton_aug = self.skeleton_transform()
        # self.class_mapping = self._class_mapping()
        if self.data_type == 'skeleton' or self.mask:
            self.skeleton_data = pickle.load(open(f"{self.prefix}/processed/{mode}_pose.pkl", "rb"))
        self._read_file(os.path.join(self.prefix, file_path))

    def _class_mapping(self):
        class_cnt = 0
        class_dict = dict()
        for i in range(len(self.data)):
            if self.data[i][1] == 0:
                class_dict[i] = 61 + class_cnt
                class_cnt += 1
            else:
                class_dict[i] = self.data[i][1]
        return class_dict

    def _read_file(self, path):
        idx = 0
        for line in open(path, 'r'):
            video_file_name, _, start_frame, end_frame = line[:-1].split(' | ')
            video_file_name = video_file_name.rsplit("/")[-1]
            framewise_labels = np.load(
                f"{self.prefix}/processed/{self.mode}/framewise_label/{video_file_name}.npy"
            ).astype(np.int)
            start_frame = int(start_frame)
            end_frame = int(end_frame)
            if self.data_type == 'skeleton' or self.mask:
                self.interpolate_skeleton_data(video_file_name, framewise_labels)
            if self.data_type == 'flow':
                end_frame = end_frame - 1
            for mid in range(start_frame, end_frame + 1, self.stride):
                start = max(mid - self.seq_duration // 2, start_frame)
                end = min(mid + self.seq_duration // 2, end_frame + 1)
                # remove partial clip
                if end - start < self.seq_duration:
                    continue
                label = [
                    Counter(
                        framewise_labels[start:end][self.clip_duration * i:self.clip_duration * (i + 1)]
                    ).most_common(1)[0][0] for i in range(self.seq_duration // self.clip_duration)]
                if self.remove_bg and label[0] == self.bg_class:
                    continue
                self.labels.append(label)
                self.data[idx] = (video_file_name, label, start, end)
                idx += 1

    def __getitem__(self, idx):
        video_file_name, label, start_frame, end_frame = self.data[idx]
        # label = self.class_mapping[idx]
        info = (video_file_name, start_frame, end_frame)
        if self.data_type == 'video':
            input_data = self.read_video(video_file_name, start_frame, end_frame)
            input_data = self.normalize(input_data)
        elif self.data_type == 'flow':
            input_data = self.read_flow(video_file_name, start_frame, end_frame)
            input_data = self.normalize(input_data)
        elif self.data_type == 'skeleton':
            input_data = self.read_skeleton(video_file_name, start_frame, end_frame)
            input_data = self.skeleton_aug(input_data).permute(0, 2, 1).unsqueeze(-1)
            # raise NotImplementedError("ERROR: func not implemented!")
        return input_data, torch.LongTensor(label), info

    def read_flow(self, video_file_name, start_frame, end_frame):
        img_root = f"{self.prefix}/processed/{self.mode}/flow/{video_file_name}_256x256"
        img_data_u = [np.array(Image.open(f"{img_root}/{str(i).zfill(5)}_u.jpg"))[None] for i in
                      range(start_frame, end_frame)]
        img_data_u = np.concatenate(img_data_u, axis=0)
        img_data_v = [np.array(Image.open(f"{img_root}/{str(i).zfill(5)}_v.jpg"))[None] for i in
                      range(start_frame, end_frame)]
        img_data_v = np.concatenate(img_data_v, axis=0)
        return np.concatenate([img_data_u[:, :, :, None], img_data_v[:, :, :, None]], axis=3)

    def read_video(self, video_file_name, start_frame, end_frame):
        img_root = f"{self.prefix}/processed/{self.mode}/video/{video_file_name}_256x256"
        img_data = [np.array(Image.open(f"{img_root}/{str(i).zfill(5)}.jpg"))[None] for i in
                    range(start_frame, end_frame)]
        if self.sampling_rate > 1:
            img_data = [img[:, ::self.sampling_rate, ::self.sampling_rate] for img in img_data]
        if self.mask:
            skeleton_data = self.skeleton_data[video_file_name]['skeleton'][start_frame:end_frame]
            for i in range(len(skeleton_data)):
                mask = np.zeros_like(img_data[i])
                _, h, w, c = mask.shape
                left_hand_center = skeleton_data[i, 15:36].mean(axis=0).astype(int)
                right_hand_center = skeleton_data[i, 36:57].mean(axis=0).astype(int)
                window_size = 32
                mask[0, max(left_hand_center[1] - window_size, 0): left_hand_center[1] + window_size,
                max(left_hand_center[0] - window_size, 0): left_hand_center[0] + window_size] = 1
                mask[0, max(right_hand_center[1] - window_size, 0):right_hand_center[1] + window_size,
                max(right_hand_center[0] - window_size, 0):right_hand_center[0] + window_size] = 1
                img_data[i] = img_data[i] * mask
        # import matplotlib.pyplot as plt
        # plt.imsave("mask.jpg", img_data[0][0])
        # print([f"{img_root}/{str(i).zfill(5)}.jpg" for i in range(start_frame, end_frame)])
        return np.concatenate(img_data, axis=0)

    def read_skeleton(self, video_file_name, start_frame, end_frame):
        input_data = self.skeleton_data[video_file_name]['interpolated_skeleton']
        return input_data[start_frame:end_frame]

    def interpolate_skeleton_data(self, fname, flabel):
        skeleton_seq = self.skeleton_data[fname]['skeleton']
        skeleton_idx = self.skeleton_data[fname]['frame_idx']
        total_lgt = len(flabel)
        interpolated_skeleton = np.zeros((total_lgt, skeleton_seq.shape[1], skeleton_seq.shape[2]))
        interpolated_skeleton[skeleton_idx] = skeleton_seq
        empty_frames = [i for i in range(total_lgt) if i not in skeleton_idx]
        if len(empty_frames) > 0:
            start_end_pair = []
            start_idx = empty_frames[0]
            end_idx = empty_frames[0] + 1
            for idx, empty_idx in enumerate(empty_frames[1:]):
                if empty_idx == empty_frames[idx] + 1:
                    end_idx += 1
                else:
                    start_end_pair.append([start_idx, end_idx])
                    start_idx = empty_idx
                    end_idx = empty_idx + 1
            if len(start_end_pair) == 0 or [start_idx, end_idx] != start_end_pair[-1]:
                start_end_pair.append([start_idx, end_idx])
            for pair in start_end_pair:
                if pair[0] == 0:
                    interpolated_skeleton[:pair[1]] = interpolated_skeleton[pair[1]]
                elif pair[1] == total_lgt:
                    interpolated_skeleton[pair[0]:] = interpolated_skeleton[pair[0] - 1]
                else:
                    for i in range(pair[0], pair[1]):
                        alpha = 1.0 * (i - pair[0]) / (pair[1] - pair[0])
                        interpolated_skeleton[i] = \
                            (1 - alpha) * interpolated_skeleton[pair[0] - 1] + alpha * interpolated_skeleton[pair[1]]
        self.skeleton_data[fname]['skeleton'] = interpolated_skeleton
        normed_interpolated_skeleton = self.skeleton_normalize(copy.deepcopy(interpolated_skeleton))
        self.skeleton_data[fname]['interpolated_skeleton'] = normed_interpolated_skeleton

    def skeleton_normalize(self, skeleton):
        skeleton[:, :, 0] = skeleton[:, :, 0] - skeleton[:, 0].mean(axis=0)[0]
        skeleton[:, :, 1] = skeleton[:, :, 1] - skeleton[:, 9].mean(axis=0)[1]
        norm_x = abs(skeleton[:, 9, 0] - skeleton[:, 10, 0]).mean() * 2
        skeleton[:, :, 0] = skeleton[:, :, 0] / norm_x
        skeleton[:, :, 1] = skeleton[:, :, 1] / norm_x
        skeleton[:, :, 2] = (skeleton[:, :, 2] - skeleton[:, :, 2].min()) / (
                skeleton[:, :, 2].max() - skeleton[:, :, 2].min()) * 2 - 1
        return skeleton

    def normalize(self, video):
        mean = [114.75, 114.75, 114.75]
        std = [57.375, 57.375, 57.375]
        video = self.data_aug(video).float()
        # video = video - torch.FloatTensor(mean)[None, :, None, None]
        # video = video / torch.FloatTensor(std)[None, :, None, None]
        # pdb.set_trace()
        video = video / 127.5 - 1
        return video

    def transform(self):
        if self.transform_mode == "train":
            print("Apply training transform.")
            return video_augmentation.Compose([
                video_augmentation.RandomCrop(224 // self.sampling_rate),
                video_augmentation.RandomHorizontalFlip(0.5),
                video_augmentation.ToTensor(),
            ])
        else:
            print("Apply testing transform.")
            return video_augmentation.Compose([
                video_augmentation.CenterCrop(224 // self.sampling_rate),
                video_augmentation.ToTensor(),
            ])

    def skeleton_transform(self):
        if self.transform_mode == "train":
            print("Apply training transform.")
            return skeleton_augmentation.Compose([
                skeleton_augmentation.RandomHorizontalFlip(0.5),
                # skeleton_augmentation.RandomResize(0.1),
                # skeleton_augmentation.RandomShift(0.2),
                skeleton_augmentation.ToTensor(),
            ])
        else:
            print("Apply testing transform.")
            return skeleton_augmentation.Compose([
                skeleton_augmentation.RandomHorizontalFlip(0.5),
                skeleton_augmentation.ToTensor(),
            ])

    def get_labels(self):
        return self.labels

    def __len__(self):
        return len(list(self.data.keys()))


if __name__ == '__main__':
    batch_size = 10
    train_flag = False
    train_dataset = MSSLClipFeeder(mode='EVALUATION', transform_mode=train_flag)

    dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        sampler=ImbalancedDatasetSampler(train_dataset),
        batch_size=batch_size,
        shuffle=False,
        drop_last=train_flag,
        num_workers=5,
        pin_memory=False
    )

    progress_bar = tqdm.tqdm(
        desc='Count iter', ncols=80,
        total=len(dataloader),
        initial=0
    )

    cls_dict = defaultdict(int)
    for batch_idx, data in enumerate(dataloader):
        pdb.set_trace()
        for idx in range(batch_size):
            cls_dict[data[1][idx].item()] += 1
        progress_bar.update(1)

    print(cls_dict)
