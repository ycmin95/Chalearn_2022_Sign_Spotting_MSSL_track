import os
import sys
import pdb
import tqdm
import itertools
import pickle
import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm

from utils import import_class, AverageMeter
from utils.options import get_config


class Tester(object):
    def __init__(
            self, device, data_loader,
            model, recorder, decoder, evaluator, official_evaluator, data_type
    ):
        self.config = get_config()
        self.device = device
        self.recorder = recorder

        self.data_loader = data_loader

        self.evaluator = evaluator

        self.model_mode = self.config.train['model_mode']
        self.model = model
        self.decoder = decoder

        self.official_evaluator = official_evaluator
        self.data_type = data_type

    def start(self, test_type):
        if test_type == 'eval_video':
            self._eval_video()

    def _eval_video(self):
        self.model[self.model_mode].eval()

        test_gt_paths, test_pred_paths = [], []
        pred_pickle_file = {}
        loader = self.data_loader['video_valid']

        progress_bar = tqdm.tqdm(
            desc='Validate iter', ncols=80,
            total=len(loader),
            initial=0
        )

        save_root = f"./{os.path.join(self.config.data['work_dir'], self.config.proj_name)}/prediction_validate"
        if not os.path.exists(save_root):
            os.mkdir(save_root)
        prob_file = open(os.path.join(save_root, 'probs.txt'), 'w')

        for batch_idx, data in enumerate(loader):
            if self.data_type == 'video_feature':
                vid = self.device.data_to_device(data[0])
                lgt = self.device.data_to_device(data[1])
                video_file_name, start_frame, end_frame = data[4][0]
                with torch.no_grad():
                    cls_logits, reg_result = self.model[self.model_mode](vid, lgt)
                    self.save_probs(cls_logits, prob_file)

            test_gt_paths.append(f'./MSSL_dataset/VALIDATION/MSSL_VAL_SET_GT_TXT/{video_file_name}.txt')
            save_path = os.path.join(save_root, video_file_name)
            test_pred_paths.append(save_path)

            recog_result = self.decoder.decode_feature(cls_logits, reg_result, start_frame, end_frame, save_path)
            split_results = self.evaluator.generate_labels_start_end_time(recog_result)

            pred_pickle_file[video_file_name] = []
            for idx, item in enumerate(split_results[0]):
                if item != -1:
                    pred_pickle_file[video_file_name].append(
                        [item, split_results[1][idx] * 40, split_results[2][idx] * 40]
                    )
            progress_bar.update(1)

        f1_score = self.offical_eval(save_root, pred_pickle_file)
        prob_file.close()

    def offical_eval(self, save_root, pred_pickle_file):
        gt_pkl_path = "./MSSL_dataset/VALIDATION/MSSL_VAL_SET_GT.pkl"
        gt_pkl_path = os.path.abspath(gt_pkl_path)
        if not os.path.exists(f"{save_root}/ref"):
            os.makedirs(f"{save_root}/ref")
        if not os.path.exists(f"{save_root}/res"):
            os.makedirs(f"{save_root}/res")
        with open(f"{save_root}/res/predictions.pkl", 'wb') as handle:
            pickle.dump(pred_pickle_file, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if not os.path.exists(f"{save_root}/ref/ground_truth.pkl"):
            os.system(f"ln -s {gt_pkl_path} {save_root}/ref/ground_truth.pkl")
        return self.official_evaluator(folder_in=save_root, folder_out=save_root)

    def save_probs(self, logits, file):
        logits = logits.cpu().softmax(dim=-1).max(dim=-1)[0].numpy()
        logits = np.around(logits, 6)
        for i in range(logits.shape[0]):
            file.write(str(logits[i]) + ' ')
        file.write('\n')
