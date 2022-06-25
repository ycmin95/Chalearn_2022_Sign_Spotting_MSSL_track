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


class Trainer(object):
    def __init__(
            self, rng,
            logger, recorder,
            device, data_loader,
            model, decoder, criterion,
            optimizer, evaluator, official_evaluator
    ):
        self.rng = rng
        self.config = get_config()
        self.logger = logger
        self.recorder = recorder
        self.device = device

        self.data_loader = data_loader

        self.evaluator = evaluator
        self.official_evaluator = official_evaluator

        self.model_mode = self.config.train['model_mode']
        self.model = model
        self.decoder = decoder

        self.criterion = criterion
        self.optimizer = optimizer

    def start(self):
        print("Start epoch:", self.config.optimizer['start_epoch'], "Logger step: ", self.logger.global_step)
        best_acc = 0.0

        for epoch in range(self.config.optimizer['start_epoch'], self.config.train['num_epoch']):
            self._train(epoch)
            self.optimizer[self.model_mode].scheduler.step()

            if epoch % self.config.train['eval_clip_interval'] == 0:
                acc = self._eval_clip(epoch)
                best_acc = max(acc, best_acc)

                self.logger.save_ckpt(state={
                    'epoch': epoch,
                    'model_state_dict': self.model[self.model_mode].state_dict(),
                    'optimizer_state_dict': self.optimizer[self.model_mode].state_dict(),
                    'best_metric_val': self.logger.best_metric_val,
                    'rng_state': self.rng.save_rng_state(),
                    "global_step": self.logger.global_step
                }, cur_metric_val=acc)

            if epoch % self.config.train['eval_video_interval'] == 0 or epoch == self.config.train['num_epoch'] - 1:
                self._eval_video(epoch)

        self.logger.add_scalar('validate_prec@1', best_acc)

        self.logger.close()

    def _train(self, epoch):
        self.model[self.model_mode].train()

        loader = self.data_loader['clip_train']
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()
        clr = [group['lr'] for group in self.optimizer[self.model_mode].optimizer.param_groups]

        progress_bar = tqdm.tqdm(
            desc='Train iter', ncols=80,
            total=len(loader),
            initial=0
        )

        for batch_idx, data in enumerate(loader):
            vid = self.device.data_to_device(data[0])
            target = self.device.data_to_device(data[1])
            vid = vid.transpose(1, 2)
            with torch.autograd.set_detect_anomaly(True):
                ret_dict = self.model[self.model_mode](vid)
                batch, t, num_classes = ret_dict['conv_predictions'].shape
                output = ret_dict['conv_predictions'].view(-1, num_classes)
                loss = self.criterion['ce_loss'](ret_dict['conv_predictions'].view(-1, num_classes), target.view(-1))

            prec1, prec5 = self.evaluator.eval_clip(output.data, target.view(-1), topk=(1, 5))

            losses.update(loss.item(), vid.size(0))
            top1.update(prec1.item(), vid.size(0))
            top5.update(prec5.item(), vid.size(0))

            if np.isinf(loss.item()):
                print(data[-1])
                continue

            self.optimizer[self.model_mode].zero_grad()
            loss.backward()

            self.optimizer[self.model_mode].step()

            if batch_idx % self.recorder.log_interval == 0 or batch_idx == len(loader) - 1:
                self.recorder.print_log(
                    '\nEpoch: {}, Batch({}/{}) done. lr:{:.6f} Loss: {loss.val:.4f} ({loss.avg:.4f}) '
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'
                        .format(epoch, batch_idx, len(loader), clr[0], loss=losses, top1=top1, top5=top5)
                )

            self.logger.step(1)
            self.logger.add_scalar('train_loss', losses.avg)
            self.logger.add_scalar('train_prec@1', top1.avg)
            self.logger.add_scalar('train_prec@5', top5.avg)
            progress_bar.update(1)

    def _eval_clip(self, epoch):
        self.model[self.model_mode].eval()

        loader = self.data_loader['clip_valid']

        top1 = AverageMeter()
        blank_top1 = AverageMeter()
        non_blank_top1 = AverageMeter()
        top5 = AverageMeter()
        blank_top5 = AverageMeter()
        non_blank_top5 = AverageMeter()
        losses = AverageMeter()

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

            batch, t, num_classes = ret_dict['conv_predictions'].shape

            output = ret_dict['conv_predictions'].view(-1, num_classes)

            loss = self.criterion['ce_loss'](ret_dict['conv_predictions'].view(-1, num_classes), target.view(-1))
            # loss += self.criterion['ce_loss'](ret_dict['lstm_predictions'].view(-1, num_classes), target.view(-1))

            target = target.view(-1)
            prec1, prec5 = self.evaluator.eval_clip(output.data, target.view(-1), topk=(1, 5))
            top1.update(prec1.item(), vid.size(0))
            top5.update(prec5.item(), vid.size(0))

            if (target == 0).sum() > 0:
                prec1, prec5 = self.evaluator.eval_clip(output.data[torch.where(target == 0)[0]],
                                                        target[target == 0].view(-1), topk=(1, 5))
                blank_top1.update(prec1.item(), vid.size(0))
                blank_top5.update(prec5.item(), vid.size(0))

            if (target != 0).sum() > 0:
                prec1, prec5 = self.evaluator.eval_clip(output.data[torch.where(target != 0)[0]],
                                                        target[target != 0].view(-1), topk=(1, 5))
                non_blank_top1.update(prec1.item(), vid.size(0))
                non_blank_top5.update(prec5.item(), vid.size(0))

            losses.update(loss.item(), vid.size(0))
            progress_bar.update(1)

        print("")
        self.recorder.print_log(
            '\nAll Results: Validate done. Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.
                format(top1=top1, top5=top5)
        )
        self.recorder.print_log(
            '\nBlank Results: Validate done. Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.
                format(top1=blank_top1, top5=blank_top5)
        )
        self.recorder.print_log(
            '\nNon-Blank Results: Validate done. Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.
                format(top1=non_blank_top1, top5=non_blank_top5)
        )

        self.logger.step(1)
        self.logger.add_scalar('validate_loss', losses.avg)
        self.logger.add_scalar('validate_prec@1', non_blank_top1.avg)
        self.logger.add_scalar('validate_prec@5', non_blank_top5.avg)

        return top1.avg

    def _eval_video(self, epoch):
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
