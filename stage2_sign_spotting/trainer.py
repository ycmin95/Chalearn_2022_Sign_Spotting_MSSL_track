import os
import sys
import pdb
from tqdm import tqdm
import itertools
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
# from lib.baseline.sign_spotting.models import transformer

from utils import import_class, AverageMeter
from utils.options import get_config


class DetectorTrainerModel(object):
    def __init__(
            self, rng,
            logger, recorder,
            device, data_loader,
            model, decoder, criterion,
            optimizer, evaluator, official_evaluator,
            data_type='fusion_feature'
    ):
        self.config = get_config()
        self.rng = rng
        self.logger = logger
        self.recorder = recorder
        self.device = device

        self.data_loader = data_loader

        self.evaluator = evaluator

        self.model_mode = self.config.train['model_mode']
        self.model = model
        self.decoder = decoder

        self.criterion = criterion
        self.optimizer = optimizer

        self.reg_range = [[0, 2], [2, 4], [4, 8]]
        self.max_seq_len = 750
        self.num_classes = 61

        self.official_evaluator = official_evaluator
        self.data_type = data_type

    def start(self):
        print("Start epoch: ", self.config.optimizer['start_epoch'], "Logger step: ", self.logger.global_step)
        best_f1_score = 0.0

        for epoch in range(self.config.optimizer['start_epoch'], self.config.train['num_epoch']):
            self._train(epoch)
            self.optimizer[self.model_mode].scheduler.step()

            if epoch % self.config.train['eval_video_interval'] == 0 or epoch == self.config.train['num_epoch'] - 1:
                f1_score = []
                # for key in ['feature', 'cnn', 'rnn', 'transformer']:
                for key in ['cnn']:
                    f1_score.append(self._eval_video(epoch, key))
                f1_score = max(f1_score)
                best_f1_score = max(f1_score, best_f1_score)

                self.logger.save_ckpt(state={
                    'epoch': epoch,
                    'model_state_dict': self.model[self.model_mode].state_dict(),
                    'optimizer_state_dict': self.optimizer[self.model_mode].state_dict(),
                    'best_metric_val': self.logger.best_metric_val,
                    'rng_state': self.rng.save_rng_state(),
                    "global_step": self.logger.global_step
                }, cur_metric_val=f1_score)

        self.logger.add_scalar('f1_score', best_f1_score)
        self.logger.close()

    def _train(self, epoch):
        self.model[self.model_mode].train()
        loader = self.data_loader['video_train']

        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()
        clr = [group['lr'] for group in self.optimizer[self.model_mode].optimizer.param_groups]

        for batch_idx, data in enumerate(tqdm(loader)):
            if self.data_type == 'video_feature':
                feature = self.device.data_to_device(data[0])
                lgt = self.device.data_to_device(data[1])
                label = self.device.data_to_device(data[2])
                offset = self.device.data_to_device(data[3])
                cls_logits, reg_result = self.model[self.model_mode](feature, lgt)
                loss = self.criterion['detector_loss'](cls_logits, reg_result, label, offset, lgt)
            elif self.data_type == 'fusion_feature':
                feature = self.device.data_to_device(data[0])
                lgt = self.device.data_to_device(data[1])
                label = self.device.data_to_device(data[2])
                offset = self.device.data_to_device(data[3])
                cls_logits, reg_result = self.model[self.model_mode](feature, lgt)
                loss = self.criterion['detector_loss'](cls_logits, reg_result, label, offset, lgt)

            losses.update(loss.item(), feature.size(0))
            if np.isinf(loss.item()):
                print(data[-1])
                continue
            self.optimizer[self.model_mode].zero_grad()
            loss.backward()

            if batch_idx % self.recorder.log_interval == 0 or batch_idx == len(loader) - 1:
                self.recorder.print_log(
                    '\nEpoch: {}, Batch({}/{}) done. lr:{:.6f} Loss: {loss.val:.4f} ({loss.avg:.4f})\n'
                        .format(epoch, batch_idx, len(loader), clr[0], loss=losses)
                )

            self.optimizer[self.model_mode].step()

            self.logger.step(1)
            self.logger.add_scalar('train_loss', losses.avg)
            # progress_bar.update(1)

    def _eval_video(self, epoch, key):
        self.model[self.model_mode].eval()

        test_gt_paths, test_pred_paths = [], []
        loader = self.data_loader['video_valid']

        save_root = f"./{os.path.join(self.config.data['work_dir'], self.config.proj_name)}/prediction_{epoch}" + key
        if not os.path.exists(save_root):
            os.mkdir(save_root)

        pred_pickle_file = {}
        for batch_idx, data in enumerate(tqdm(loader)):
            if self.data_type == 'video_feature':
                vid = self.device.data_to_device(data[0])
                lgt = self.device.data_to_device(data[1])
                video_file_name, _, start_frame, end_frame = data[4][0]
                with torch.no_grad():
                    cls_logits, reg_result = self.model[self.model_mode](vid, lgt)
            elif self.data_type == 'fusion_feature':
                vid = self.device.data_to_device(data[0])
                lgt = self.device.data_to_device(data[1])
                video_file_name, _, start_frame, end_frame = data[4][0]
                with torch.no_grad():
                    cls_logits, reg_result = self.model[self.model_mode](vid, lgt)

            test_gt_paths.append(
                f'./MSSL_dataset/VALIDATION/MSSL_VAL_SET_GT_TXT/{video_file_name}.txt')
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

        f1_score, pre, recall = self.offical_eval(save_root, pred_pickle_file)

        self.recorder.print_log(
            '\tEpoch: {} done. {} F1 score: {}'
                .format(epoch, key, f1_score)
        )

        return f1_score

    def offical_eval(self, save_root, pred_pickle_file):
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
        return self.official_evaluator(folder_in=save_root, folder_out=save_root)

    def generate_fpn_label_reg(self, fpn_level, label, seg):
        "label:N, ; seg:N x 2"
        reg_range = self.reg_range[:fpn_level]
        points_list = []
        for i in range(fpn_level):
            stride = 2 ** i
            # reg_range_now = reg_range[i]
            points = torch.arange(0, self.max_seq_len, stride)[:, None]
            reg_range_now = torch.tensor(reg_range[i])[None].repeat(points.size(0), 1)
            padded_stride = torch.tensor(stride)[None].repeat(points.size(0), 1)
            points_list.append(torch.cat((points, reg_range, padded_stride), dim=1))
        # points = torch.cat(points_list, dim=0)
        return points_list

    def get_points_clip(self, fpn_feats, points_list):
        assert len(fpn_feats) == len(points_list)
        points_clip_list = []
        for i in range(len(fpn_feats)):
            assert len(fpn_feats[i]) <= len(points_list[i])
            points_clip_list.append(points_list[i][0:len(fpn_feats[i])])
        # return points_clip_list
        return torch.cat(points_clip_list, dim=0)

    def get_fpn_label_for_single_video(self, points, gt_labels, gt_segs):
        """
        gt_segs: N x 2, gt_labels: N, points pts x 4
        points[i] : position, reg_range[0,1], stride
        """
        num_points = points.size(0)
        seg_len = gt_segs[:, 1] - gt_segs[:, 0]
        lens = seg_len[None, :].repeat(num_points, 1)  # [pts,N]

        gt_segs_expand = gt_segs.expand(num_points, gt_segs.size(0), 2)  # [pts, N, 2]
        left = points[:, 0, None] - gt_segs_expand[:, :, 0]  # [pts, N]
        right = points[:, 0, None] - gt_segs_expand[:, :, 1]  # [pts, N]
        reg_target = torch.stack((left, right), dim=-1)  # [pts, N, 2]

        max_reg_dis = reg_target.max(-1)[0]  # [pts,N]

        inside_reg_dis = torch.logical_and(
            (max_reg_dis >= points[:, 1, None]),
            (max_reg_dis <= points[:, 2, None])
        )
        # the length of segment reg distance beyond reg distance is set to inf
        lens.masked_fill_(inside_reg_dis, float('inf'))

        min_len, min_index = lens.min(-1)

        reg_target = reg_target[range(num_points), min_index]  # [pts, 2]

        min_len_mask = torch.logical_and(
            (lens <= (min_len[:, None] + 1e-3)),
            (lens <= float('inf'))
        ).to(reg_target.dtype)  # [pts,N]

        gt_labels_one_hot = F.one_hot(gt_labels, self.num_classes).to(reg_target.dtype)  # [N, classes]
        cls_target = min_len_mask @ gt_labels_one_hot  # [pts,classes]

        cls_target.clamp_(min=0.0, max=1.0)
        reg_target /= points[:, 3, None]

        return cls_target, reg_target
