import os
import pdb
import cv2
import numpy as np
from tqdm import tqdm
from collections import Counter


def read_file(path):
    labels, starts, ends = [], [], []
    for line in open(path, "r"):
        label, start, end = line[:-1].split(',')
        labels.append(int(label))
        starts.append(int(start) // 40)
        ends.append(int(end) // 40)
    return labels, starts, ends


def generate_framewise_labels(gt_file_path, video_file_path, save_file_path):
    labels, starts, ends = read_file(gt_file_path)
    video_file = cv2.VideoCapture(video_file_path)
    video_length = int(video_file.get(cv2.CAP_PROP_FRAME_COUNT))
    framewise_labels = np.zeros(video_length)
    for idx in range(len(labels)):
        framewise_labels[starts[idx]:ends[idx] + 1] = labels[idx] + 1
    np.save(save_file_path, framewise_labels)


def generate_clip_labels(framewise_labels_path, start_frame, end_frame):
    seq_duration, clip_duration, stride = 8, 8, 4
    labels = []
    framewise_labels = np.load(framewise_labels_path).astype(np.int)
    start_frame = int(start_frame)
    end_frame = int(end_frame)
    framewise_labels = framewise_labels[start_frame:end_frame + 1]
    t = framewise_labels.shape[0]

    left = seq_duration // 2
    right = t // stride * stride + seq_duration // 2 - t
    x_pad = np.zeros(left + t + right)
    x_pad[left:left + t] = framewise_labels
    for t in range(left, left + t, stride):
        l, r = t - seq_duration // 2, t + seq_duration // 2
        # print(x_pad[l:r]);;p
        label = [
            Counter(
                x_pad[l:r][clip_duration * i:clip_duration * (i + 1)]
            ).most_common(1)[0][0] for i in range(seq_duration // clip_duration)]
        labels.append(label)  # [temporal//stride+1, duration, c, h, w]
    return labels


def generate_labels_start_end_time(framewise_labels, bg_class=['background']):
    labels = []
    starts = []
    ends = []
    last_label = framewise_labels[0]
    if framewise_labels[0] not in bg_class:
        labels.append(framewise_labels[0])
        starts.append(0)
    for i in range(len(framewise_labels)):
        if framewise_labels[i] != last_label:
            if framewise_labels[i] not in bg_class:
                labels.append(framewise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i - 1)
            last_label = framewise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends


def main():
    GT_root = {
        "train": "./MSSL_dataset/TRAIN/MSSL_TRAIN_SET_GT_TXT/",
        "valid": "./MSSL_dataset/VALIDATION/MSSL_VAL_SET_GT_TXT",
    }
    VID_root = {
        "train": "./MSSL_dataset/TRAIN/MSSL_TRAIN_SET_VIDEOS_ELAN/",
        "valid": "./MSSL_dataset/VALIDATION/MSSL_VAL_SET_VIDEOS",
    }
    for mode in ["train", "valid"]:
        video_file_root = VID_root[mode]
        gt_file_root = GT_root[mode]
        # generate frame-wise labels
        save_file_root = f"./MSSL_dataset/processed/{mode}/framewise_label"
        if not os.path.exists(save_file_root):
            os.makedirs(save_file_root)
        gt_file_list = sorted(os.listdir(gt_file_root))
        for gt_file_name in tqdm(gt_file_list):
            gt_file_name_prev = gt_file_name[:-4]
            gt_file_path = os.path.join(gt_file_root, gt_file_name)
            video_file_path = os.path.join(video_file_root, gt_file_name_prev + '.mp4')
            save_file_path = os.path.join(save_file_root, gt_file_name_prev)
            generate_framewise_labels(gt_file_path, video_file_path, save_file_path)

        # generate clip-wise labels
        save_file_root = f"./MSSL_dataset/processed/{mode}/clipwise_label"
        if not os.path.exists(save_file_root):
            os.makedirs(save_file_root)
        inputs_list = open(f"./MSSL_dataset/{mode}_input.txt", 'r').readlines()
        for line in tqdm(inputs_list):
            video_file_name, _, start_frame, end_frame = line[:-1].split(' | ')
            video_file_name = video_file_name.rsplit("/")[-1]
            start_frame = int(start_frame)
            end_frame = int(end_frame)
            fname = f"./MSSL_dataset/processed/{mode}/framewise_label/{video_file_name}.npy"
            label = generate_clip_labels(fname, start_frame, end_frame)
            label = np.array(label)
            np.save(os.path.join(save_file_root, video_file_name + '.npy'), label)


if __name__ == '__main__':
    main()
