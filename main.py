import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import pdb
import random
import itertools
import importlib
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from utils import get_config, import_class
from utils import Logger, Recorder
from utils import RandomState, GpuDataParallel, Optimizer
from utils import Decoder
from evaluator.evaluator import Evaluator
from evaluator.official_evaluator import evaluate as OffEvaluator


class Processor(object):
    def __init__(self):
        self.config = get_config()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config.device)
        self.device = GpuDataParallel()
        torch.multiprocessing.set_sharing_strategy('file_system')
        self._load_logger()
        self._load_recorder()
        if self.config.misc['random_seed']:
            self.rng = RandomState(seed=self.config.misc['random_seed'])
        self.data_loader = dict()
        self._load_data()
        self._load_evaluator()
        self._load_decoder()
        self._load_framework()
        if self.config.phase == 'train':
            self._load_trainer()
        elif self.config.phase == 'test':
            self._load_tester()

    def _load_logger(self):
        print('loading logger..')
        self.logger = Logger(
            ckpt_path=os.path.join(self.config.data['ckpt_path'], self.config.proj_name),
            tsbd_path=os.path.join(self.config.data['viz_path'], self.config.proj_name)
        )
        print('logger load finished!')

    def _load_recorder(self):
        print('loading recorder')
        self.recorder = Recorder(
            work_path=os.path.join(self.config.data['work_dir'], self.config.proj_name),
            print_log=self.config.data['print_log'],
            log_interval=self.config.data['log_interval']
        )
        print('recorder load finished!')

    def _load_data(self):
        print('loading data...')
        dataset_list = zip(
            ["train", "valid"],
            [True, False]
        )
        if len(self.config.segment_feeder) > 0:
            segment_feeder = import_class(self.config.segment_feeder['class'])
        if len(self.config.video_feeder) > 0:
            video_feeder = import_class(self.config.video_feeder['class'])
        for idx, (mode, train_flag) in enumerate(dataset_list):
            if len(self.config.segment_feeder) > 0:
                arg = self.config.segment_feeder['args']
                arg["mode"] = mode
                arg["transform_mode"] = train_flag
                batch_size = self.config.train['segment_batch_size'] if mode == "train" else self.config.test[
                    'segment_batch_size']
                dataset = segment_feeder(**arg)
                self.data_loader['clip_' + mode.lower()] = torch.utils.data.DataLoader(
                    dataset=dataset,
                    # sampler=ImbalancedDatasetSampler(dataset) if mode == 'TRAIN' else None,
                    batch_size=batch_size,
                    shuffle=train_flag,
                    drop_last=train_flag,
                    num_workers=self.config.misc['num_workers'],
                    pin_memory=False
                )
            if len(self.config.video_feeder) > 0:
                arg = self.config.video_feeder['args']
                arg["mode"] = mode
                arg["transform_mode"] = train_flag
                batch_size = self.config.train['video_batch_size'] if mode == "train" else self.config.test[
                    'video_batch_size']
                self.data_loader['video_' + mode.lower()] = torch.utils.data.DataLoader(
                    dataset=video_feeder(**arg),
                    batch_size=batch_size,
                    shuffle=train_flag,
                    drop_last=train_flag,
                    num_workers=self.config.misc['num_workers'],
                    collate_fn=video_feeder.collate_fn,
                    pin_memory=False
                )
        print('data load finished!')

    def _load_decoder(self):
        print('loading decoder...')
        self.decoder = Decoder(**self.config.decoder['args'])
        print('decoder load finished!')

    def _load_evaluator(self):
        print('loading evaluator...')
        self.evaluator = Evaluator()
        self.official_evaluator = OffEvaluator
        print('evaluator load finished!')

    def _load_framework(self):
        print('loading framework...')
        framework_class = import_class(self.config.framework)
        self.framework = framework_class(self.device, self.rng)
        self.logger.global_step = self.framework.global_step
        print('framework load finished!')

    def _load_trainer(self):
        print('loading trainer...')
        trainer_class = import_class(self.config.train['trainer'])
        self.trainer = trainer_class(self.rng, self.logger, self.recorder,
                                     self.device, self.data_loader,
                                     self.framework.model, self.decoder, self.framework.criterion,
                                     self.framework.optimizer, self.evaluator, self.official_evaluator)
        print('trainer load finished!')

    def _load_tester(self):
        print('loading tester...')
        tester_class = import_class(self.config.test['tester'])
        self.tester = tester_class(self.device, self.data_loader, self.framework.model,
                                   self.recorder, self.decoder, self.evaluator, self.official_evaluator)

        print('tester load finished!')

    def start(self):
        cudnn.benchmark = True
        if self.config.phase == 'train':
            self.trainer.start()
        elif self.config.phase == 'test':
            self.tester.start(self.config.test['type'])


def main():
    processor = Processor()
    processor.start()


if __name__ == '__main__':
    main()
