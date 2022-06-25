import os
import cv2
import pdb
import glob
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool


def saved_video(avi_idx, video_list, save_dir):
    avi_path = video_list[avi_idx]
    fname = os.path.basename(avi_path)[:-4]
    saved_path = f"{save_dir}/{fname}"
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    cap = cv2.VideoCapture(avi_path)
    frame_idx = 0
    ret, frame = cap.read()
    while ret:
        cv2.imwrite(f"{saved_path}/{str(frame_idx).zfill(5)}.jpg", frame)
        ret, frame = cap.read()
        frame_idx += 1


def run_mp_cmd(processes, process_func, process_args):
    with Pool(processes) as p:
        outputs = list(tqdm(p.imap(process_func, process_args), total=len(process_args)))
    return outputs


src_dir = [
    "./MSSL_dataset/TRAIN/MSSL_TRAIN_SET_VIDEOS_ELAN/",
    "./MSSL_dataset/VALIDATION/MSSL_VAL_SET_VIDEOS/",
    "./MSSL_dataset/MSSL_TEST_SET_VIDEOS/",
]
tgt_dir = [
    "./MSSL_dataset/processed/train/original_video/",
    "./MSSL_dataset/processed/valid/original_video/",
    "./MSSL_dataset/processed/test/original_video/",
]

for src_path, tgt_path in zip(src_dir, tgt_dir):
    mp4_list = sorted(glob.glob(f"{src_path}/*.mp4"))[:10]
    run_mp_cmd(10, partial(saved_video, video_list=mp4_list, save_dir=tgt_path), np.arange(len(mp4_list)))
    break
