import os
import cv2
import pdb
import glob
import pickle
import numpy as np
from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
from mediapipe_layer import MediaPipe


def estimate_pose(vid, info):
    mp = MediaPipe()
    img_list = sorted(glob.glob(f"{info[vid]}/*.jpg"))
    pose_estimation = []
    for img_idx, img_path in enumerate(tqdm(img_list)):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        holistic_results = mp.holistic_model.process(img)
        hand_results = mp.hand_detector.process(img)
        pose_estimation.append({
            'holistic_face_landmarks': holistic_results.face_landmarks,
            'holistic_left_hand_landmarks': holistic_results.left_hand_landmarks,
            'holistic_right_hand_landmarks': holistic_results.right_hand_landmarks,
            'holistic_pose_landmarks': holistic_results.pose_landmarks,
            'holistic_pose_world_landmarks': holistic_results.pose_world_landmarks,
            'hand_multi_hand_landmarks': hand_results.multi_hand_landmarks,
            'hand_multi_hand_world_landmarks': hand_results.multi_hand_world_landmarks,
        })

    save_path = info[vid].replace('original_video', 'pose') + "_pose.pkl"
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'wb') as handle:
        pickle.dump(pose_estimation, handle, protocol=pickle.HIGHEST_PROTOCOL)


def run_mp_cmd(processes, process_func, process_args):
    with Pool(processes) as p:
        outputs = list(tqdm(p.imap(process_func, process_args), total=len(process_args)))
    return outputs


vid_dir_list = sorted(glob.glob("./MSSL_dataset/processed/*/original_video/*"))
remain_vid_dir_list = []
# check pose generation results when estimating unsuccessfully
for vid_info in vid_dir_list:
    if not os.path.exists(vid_info.replace("original_video", "pose")):
        remain_vid_dir_list.append(vid_info)
# estimate_pose(0, vid_dir_list)
run_mp_cmd(10, partial(estimate_pose, info=remain_vid_dir_list), np.arange(len(remain_vid_dir_list)))
