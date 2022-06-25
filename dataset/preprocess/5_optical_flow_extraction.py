import os
import cv2
import pdb
import glob
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool


def compute_TVL1(prev, curr, bound=20):
    "Compute the TV-L1 optical flow."
    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None)
    flow = np.clip(flow, -20, 20)
    assert flow.dtype == np.float32
    flow = (flow + bound) * (255.0 / (2 * bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0
    return flow


def saved_video(idx, video_list):
    avi_path = video_list[idx]
    saved_path = avi_path.replace("video", "flow")
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    img_list = sorted(glob.glob(f"{avi_path}/*.jpg"))
    gray_img = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY) for img_path in img_list]
    for idx, (pre_img, img) in enumerate(tqdm(zip(gray_img[:-1], gray_img[1:]), total=len(gray_img) - 1)):
        flow = compute_TVL1(pre_img, img)
        cv2.imwrite(f"{saved_path}/{str(idx).zfill(5)}_u.jpg", flow[:, :, 0])
        cv2.imwrite(f"{saved_path}/{str(idx).zfill(5)}_v.jpg", flow[:, :, 1])


def run_mp_cmd(processes, process_func, process_args):
    with Pool(processes) as p:
        outputs = list(tqdm(p.imap(process_func, process_args), total=len(process_args)))
    return outputs


for mode in ["train", "valid", "test"]:
    vid_dir_list = sorted(glob.glob(f"./MSSL_dataset/processed/{mode}/video/*_256x256"))[:1]
    run_mp_cmd(1, partial(saved_video, video_list=vid_dir_list), np.arange(len(vid_dir_list)))
