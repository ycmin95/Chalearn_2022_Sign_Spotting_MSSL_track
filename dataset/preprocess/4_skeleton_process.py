import cv2
import pdb
import glob
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def transform(pose, left_boundary):
    pose[:, 0] = pose[:, 0] * 256 * 16 / 9 - 256 * 16 / 9 * left_boundary / 1280
    pose[:, 1] = pose[:, 1] * 256
    pose[:, 2] = pose[:, 2] * 256 * 16 / 9
    return pose


if __name__ == "__main__":
    for mode in ["train", "valid", "test"]:
        pose_files = glob.glob(f"./MSSL_dataset/processed/{mode}/pose/*_pose.pkl")
        # pose_files = glob.glob("MSSL/processed/FINAL_TRAIN/MSSL_FINAL_TRAIN_SET_VIDEOS_ELAN/*_pose.pkl")
        pose_keep_ind = [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 23, 24]
        hand_tree = [0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
        pose_tree = [0, 0, 1, 2, 0, 4, 5, 0, 0, 0, 0, 9, 10, 9, 10] + [11] + \
                    [i + 15 for i in hand_tree] + [12] + \
                    [i + 36 for i in hand_tree]
        neighbor_link = [(i, pose_tree[i]) for i in range(len(pose_tree))]
        skeleton_dict = dict()
        for pose_path in tqdm(pose_files):
            skeleton_info = pickle.load(open(pose_path, "rb"))
            mean_x = []
            upper_kp = [0, 11, 12, 23, 24]
            for pose in skeleton_info:
                value = pose['holistic_pose_landmarks']
                if value is not None:
                    x = (np.array([lm.x for lm in value.landmark]) * 1280).astype(np.int32)
                    mean_x.append(x[upper_kp].mean())
            boundary = int(np.mean(mean_x))
            start_idx = boundary - 360

            fname = pose_path[-17:-9]
            frame_idx = []
            skeleton_list = []

            print(pose_path)
            for pose_idx, pose_info in enumerate(skeleton_info):
                if pose_info['holistic_pose_landmarks'] is None or pose_info['holistic_face_landmarks'] is None:
                    continue
                facial_landmark = transform(np.array(
                    [[landmark.x, landmark.y, landmark.z] for landmark in pose_info['holistic_face_landmarks'].landmark]
                ), start_idx)
                pose_landmark = transform(np.array(
                    [[landmark.x, landmark.y, landmark.z] for landmark in pose_info['holistic_pose_landmarks'].landmark]
                ), start_idx)

                if pose_info['holistic_left_hand_landmarks'] is not None:
                    left_hand_landmark = transform(np.array(
                        [[landmark.x, landmark.y, landmark.z] for landmark in
                         pose_info['holistic_left_hand_landmarks'].landmark]
                    ), start_idx)
                else:
                    left_hand_landmark = pose_landmark[15][None].repeat(21, axis=0)

                if pose_info['holistic_right_hand_landmarks'] is not None:
                    right_hand_landmark = transform(np.array(
                        [[landmark.x, landmark.y, landmark.z] for landmark in
                         pose_info['holistic_right_hand_landmarks'].landmark]
                    ), start_idx)
                else:
                    right_hand_landmark = pose_landmark[16][None].repeat(21, axis=0)
                saved_pose = np.concatenate([pose_landmark[pose_keep_ind], left_hand_landmark, right_hand_landmark],
                                            axis=0)
                frame_idx.append(pose_idx)
                skeleton_list.append(saved_pose[None])

            skeleton_dict[fname] = {
                'frame_idx': frame_idx,
                'skeleton': np.concatenate(skeleton_list, axis=0)
            }

        with open(f"./MSSL_dataset/processed/{mode}_pose.pkl", 'wb') as handle:
            pickle.dump(skeleton_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
