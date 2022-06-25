import os
import sys
import pdb
import tqdm
import itertools
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm

from utils import import_class, AverageMeter
from utils.options import get_config

plt.switch_backend('agg')


class Tester(object):
    def __init__(
            self, device, data_loader,
            model, recorder, decoder, evaluator, official_evaluator
    ):
        self.device = device
        self.config = get_config()
        self.recorder = recorder

        self.data_loader = data_loader

        self.evaluator = evaluator
        self.official_evaluator = official_evaluator

        self.model_mode = self.config.train['model_mode']
        self.model = model
        self.decoder = decoder

    def start(self, test_type):
        if test_type == 'eval_clip':
            self._eval_clip()
        elif test_type == 'eval_video':
            self._eval_video()
        elif test_type == 'save_feature':
            self._save_feature()

    def _eval_clip(self):
        self.model[self.model_mode].eval()

        loader = self.data_loader['clip_valid']

        blank_top1 = AverageMeter()
        non_blank_top1 = AverageMeter()
        blank_top5 = AverageMeter()
        non_blank_top5 = AverageMeter()

        progress_bar = tqdm.tqdm(
            desc='Validate iter', ncols=80,
            total=len(loader),
            initial=0
        )

        for batch_idx, data in enumerate(loader):
            vid = self.device.data_to_device(data[0])
            target = self.device.data_to_device(data[1])

            vid = vid.transpose(1, 2)

            with torch.no_grad():
                ret_dict = self.model[self.model_mode](vid)

            # output = ret_dict["classify_logits"]
            batch, t, num_classes = ret_dict['conv_predictions'].shape
            output = ret_dict['conv_predictions'].view(-1, num_classes)
            target = target.view(-1)

            # prec1, prec5 = self.evaluator.eval_segment(output.data, target, topk=(1, 5))
            if (target == 0).sum() > 0:
                prec1, prec5 = self.evaluator.eval_clip(output.data[torch.where(target == 0)[0]],
                                                        target[target == 0].view(-1), topk=(1, 5))
                blank_top1.update(prec1.item(), (target == 0).sum().item())
                blank_top5.update(prec5.item(), (target == 0).sum().item())

            if (target != 0).sum() > 0:
                prec1, prec5 = self.evaluator.eval_clip(output.data[torch.where(target != 0)[0]],
                                                        target[target != 0].view(-1), topk=(1, 5))
                non_blank_top1.update(prec1.item(), (target != 0).sum().item())
                non_blank_top5.update(prec5.item(), (target != 0).sum().item())

            progress_bar.update(1)
        print("")
        self.recorder.print_log(
            '\nBlank Results: Validate done. Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.
                format(top1=blank_top1, top5=blank_top5)
        )
        self.recorder.print_log(
            '\nNon-Blank Results: Validate done. Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.
                format(top1=non_blank_top1, top5=non_blank_top5)
        )
        return blank_top1.avg

    def _eval_video(self):
        self.model[self.model_mode].eval()

        test_gt_paths, test_pred_paths = [], []
        loader = self.data_loader['video_valid']

        progress_bar = tqdm.tqdm(
            desc='Validate iter', ncols=80,
            total=len(loader),
            initial=0
        )

        save_root = f"./{os.path.join(self.config.data['work_dir'], self.config.proj_name)}/prediction_validate"
        if not os.path.exists(save_root):
            os.mkdir(save_root)
        pred_pickle_file = dict()

        for batch_idx, data in enumerate(loader):
            vid = self.device.data_to_device(data[0])
            label = self.device.data_to_device(data[1])
            video_file_name = data[-1][0][0]
            start_frame = data[-1][0][1]
            end_frame = data[-1][0][2]

            test_gt_paths.append(f'./dataset/MSSL_dataset/VALIDATION/MSSL_VAL_SET_GT_TXT/{video_file_name}.txt')
            save_path = os.path.join(save_root, video_file_name)
            test_pred_paths.append(save_path)

            with torch.no_grad():
                ret_dict, _ = self.model[self.model_mode](vid, video=True)  # [t, c, h, w]
            recog_result = self.decoder.decode(ret_dict, start_frame, end_frame, save_path)
            split_results = self.evaluator.generate_labels_start_end_time(recog_result)

            from itertools import groupby
            print("")
            print("Recog:", [x[0] for x in groupby(recog_result.astype(int))])
            print("Label:", [x[0] for x in groupby(label.int()[0].tolist())])

            pred_pickle_file[video_file_name] = []
            for idx, item in enumerate(split_results[0]):
                if item != -1:
                    pred_pickle_file[video_file_name].append(
                        [item, split_results[1][idx] * 40, split_results[2][idx] * 40]
                    )
            progress_bar.update(1)

        print("")
        import pickle
        gt_pkl_path = "./dataset/MSSL_dataset/VALIDATION/MSSL_VAL_SET_GT.pkl"
        gt_pkl_path = os.path.abspath(gt_pkl_path)
        if not os.path.exists(f"{save_root}/ref"):
            os.makedirs(f"{save_root}/ref")
        if not os.path.exists(f"{save_root}/res"):
            os.makedirs(f"{save_root}/res")
        with open(f"{save_root}/res/predictions.pkl", 'wb') as handle:
            pickle.dump(pred_pickle_file, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if not os.path.exists(f"{save_root}/ref/ground_truth.pkl"):
            os.system(f"ln -s {gt_pkl_path} {save_root}/ref/ground_truth.pkl")
        print("Official evaluation results: ")
        self.official_evaluator(folder_in=save_root, folder_out=save_root)
        print("Custom evaluation results: ")

    def _save_feature(self):
        self.model[self.model_mode].eval()
        loader = self.data_loader['video_valid']
        progress_bar = tqdm.tqdm(
            desc='Save iter', ncols=80,
            total=len(loader),
            initial=0
        )

        save_root = f'./{os.path.join(self.config.data.work_dir, self.config.proj_name)}/features'
        if not os.path.exists(save_root):
            os.mkdir(save_root)

        for batch_idx, data in enumerate(loader):
            vid = self.device.data_to_device(data[0])
            vid_lgt = self.device.data_to_device(data[1])
            label = self.device.data_to_device(data[2])
            label_lgt = self.device.data_to_device(data[3])
            video_file_name, start_frame, end_frame = data[4][0]

            save_path = os.path.join(save_root, video_file_name)

            with torch.no_grad():
                ret_dict = self.model[self.model_mode](vid[0], inputs_type='video')  # [t, c, h, w]

            save_file = {
                "label": label.cpu(),
                "features": ret_dict['sequence_features'].cpu(),
                "video_info": (video_file_name, start_frame, end_frame)
            }

            np.save(save_path, save_file)

            progress_bar.update(1)
