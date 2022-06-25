import os
import pdb
import glob

if __name__ == "__main__":
    # gather video and pose
    train_dir = "./MSSL/processed/TRAIN/MSSL_TRAIN_SET_VIDEOS_ELAN"
    valid_dir = "./MSSL/processed/VALIDATION/MSSL_VAL_SET_VIDEOS"
    save_dir = "./MSSL/processed/FINAL_TRAIN/MSSL_FINAL_TRAIN_SET_VIDEOS_ELAN"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    video_dir = glob.glob(f"{train_dir}/*_256x256") + glob.glob(f"{valid_dir}/*_256x256")
    video_dir = [os.path.abspath(vpath) for vpath in sorted(video_dir)]
    for vpath in video_dir:
        img_list = glob.glob(f"{vpath}/*_u.jpg") + glob.glob(f"{vpath}/*_v.jpg")
        if len(img_list) > 0:
            for img_path in img_list:
                os.system(f"rm {img_path}")
            print(vpath)

    #     vname = vpath.rsplit("/", 1)[1]
    #     os.system(f"ln -s {vpath} {save_dir}/{vname}")
    #
    # pose_dir = glob.glob(f"{train_dir}/*_pose.pkl") + glob.glob(f"{valid_dir}/*_pose.pkl")
    # pose_dir = [os.path.abspath(vpath) for vpath in sorted(pose_dir)]
    # for ppath in pose_dir:
    #     vname = ppath.rsplit("/", 1)[1]
    #     os.system(f"ln -s {ppath} {save_dir}/{vname}")
    #
    # # gather optical flow
    # train_flow_dir = "./MSSL/processed/TRAIN_flow/MSSL_TRAIN_flow_SET_VIDEOS_ELAN"
    # valid_flow_dir = "./MSSL/processed/VALIDATION_flow/MSSL_VAL_SET_VIDEOS"
    # save_flow_dir = "./MSSL/processed/FINAL_TRAIN_flow/MSSL_FINAL_TRAIN_SET_VIDEOS_ELAN/"
    # if not os.path.exists(save_flow_dir):
    #     os.makedirs(save_flow_dir)
    # video_dir = glob.glob(f"{train_flow_dir}/*_256x256") + glob.glob(f"{valid_flow_dir}/*_256x256")
    # video_dir = [os.path.abspath(vpath) for vpath in sorted(video_dir)]
    # for vpath in video_dir:
    #     vname = vpath.rsplit("/", 1)[1]
    #     os.system(f"ln -s {vpath} {save_flow_dir}/{vname}")
    #
    # # gather framewise label
    # train_flabel_dir = "./MSSL/TRAIN/MSSL_TRAIN_SET_FRAMEWISE_LABELS_NPY"
    # valid_flabel_dir = "./MSSL/VALIDATION/MSSL_VAL_SET_FRAMEWISE_LABELS_NPY"
    # save_flabel_dir = "./MSSL/FINAL_TRAIN/MSSL_TRAIN_SET_FRAMEWISE_LABELS_NPY"
    # if not os.path.exists(save_flabel_dir):
    #     os.makedirs(save_flabel_dir)
    # video_dir = glob.glob(f"{train_flabel_dir}/*") + glob.glob(f"{valid_flabel_dir}/*")
    # video_dir = [os.path.abspath(vpath) for vpath in sorted(video_dir)]
    # for vpath in video_dir:
    #     vname = vpath.rsplit("/", 1)[1]
    #     os.system(f"ln -s {vpath} {save_flabel_dir}/{vname}")
