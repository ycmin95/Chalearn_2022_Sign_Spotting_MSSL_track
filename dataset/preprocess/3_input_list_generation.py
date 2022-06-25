import os
import pdb
import cv2
import glob
import pickle
import mediapipe
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool


def load_boundary(pickle_path):
    pose_info = pickle.load(open(pickle_path, "rb"))
    upper_kp = [0, 11, 12, 23, 24]
    mean_x = []
    for pose in pose_info:
        value = pose['holistic_pose_landmarks']
        if value is not None:
            x = (np.array([lm.x for lm in value.landmark]) * 1280).astype(np.int32)
            mean_x.append(x[upper_kp].mean())
    return int(np.mean(mean_x))


def run_mp_cmd(processes, process_func, process_args):
    with Pool(processes) as p:
        outputs = list(tqdm(p.imap(process_func, process_args), total=len(process_args)))
    return outputs


def resize_frames(idx, info):
    img_list = sorted(glob.glob(f"{info[idx]}/*.jpg"))
    boundary = load_boundary(f"{info[idx].replace('original_video','pose')}_pose.pkl")
    save_dir = info[idx].replace('original_video', 'video')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for img_path in img_list:
        if ".jpg" not in img_path:
            continue
        img = cv2.imread(img_path)
        saved_img = cv2.resize(img[:, boundary - 360:boundary + 360], dsize=(256, 256))
        cv2.imwrite(img_path.replace("original_video", "video"), saved_img)


for mode in ["train", "valid", "test"]:
    dir_path = f"./MSSL_dataset/processed/{mode}/original_video/*"
    vid_dir_list = sorted(glob.glob(dir_path))
    vid_dir_list = [line for line in vid_dir_list if "pose" not in line and '256x256' not in line]
    print(mode, len(vid_dir_list))
    # if len(vid_dir_list) != len(glob.glob(dir_path.replace("original_video", "video"))):
    #     run_mp_cmd(10, partial(resize_frames, info=vid_dir_list), np.arange(len(vid_dir_list)))

    with open(f"./MSSL_dataset/{mode}_input.txt", "w") as f:
        for vpath in tqdm(vid_dir_list):
            img = glob.glob(f"{vpath}/*.jpg")
            pickle_path = f"{vpath.replace('original_video','pose')}_pose.pkl"
            pose_info = pickle.load(open(pickle_path, "rb"))
            for start_idx, pose in enumerate(pose_info):
                if pose['holistic_pose_landmarks'] is not None:
                    break
            for end_idx, pose in enumerate(pose_info[::-1]):
                if pose['holistic_pose_landmarks'] is not None:
                    break
            if start_idx != 0:
                f.writelines(
                    f"{vpath} | {vpath.replace('original_video','pose')}_pose.pkl | {start_idx+3} | {len(img)-end_idx-1}\n")
            else:
                f.writelines(
                    f"{vpath} | {vpath.replace('original_video','pose')}_pose.pkl | {start_idx} | {len(img)-end_idx-1}\n")

with open(f"./MSSL_dataset/final_train_input.txt", "w") as f:
    vid_dir_list = []
    for mode in ["train", "valid"]:
        dir_path = f"./MSSL_dataset/processed/{mode}/original_video/*"
        curr_vid_dir_list = sorted(glob.glob(dir_path))
        curr_vid_dir_list = [line for line in curr_vid_dir_list if "pose" not in line and '256x256' not in line]
        vid_dir_list += curr_vid_dir_list
        print(mode, len(vid_dir_list))
    for vpath in tqdm(vid_dir_list):
        img = glob.glob(f"{vpath}/*.jpg")
        pickle_path = f"{vpath.replace('original_video','pose')}_pose.pkl"
        pose_info = pickle.load(open(pickle_path, "rb"))
        for start_idx, pose in enumerate(pose_info):
            if pose['holistic_pose_landmarks'] is not None:
                break
        for end_idx, pose in enumerate(pose_info[::-1]):
            if pose['holistic_pose_landmarks'] is not None:
                break
        if start_idx != 0:
            f.writelines(
                f"{vpath} | {vpath.replace('original_video','pose')}_pose.pkl | {start_idx+3} | {len(img)-end_idx-1}\n")
        else:
            f.writelines(
                f"{vpath} | {vpath.replace('original_video','pose')}_pose.pkl | {start_idx} | {len(img)-end_idx-1}\n")
