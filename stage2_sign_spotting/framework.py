import os
import pdb
import random
import itertools
import importlib
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from utils import get_config, import_class
from utils import RandomState, GpuDataParallel, Optimizer


class Framework():
    def __init__(self, device, rng):
        self.config = get_config()
        self.device = device
        self.rng = rng
        self.global_step = 1
        self.model_mode = self.config.train['model_mode']
        self._load_model()
        self._load_optimizer()

        if self.config.model['weights']:
            self._load_weights()

        if self.config.phase == 'train':
            self._load_criterion()

    def _model_to_device(self, model):
        model = model.to(self.device.output_device)

        if len(self.device.gpu_list) > 1:
            model = nn.DataParallel(
                model,
                device_ids=self.device.gpu_list,
                output_device=self.device.output_device
            )
            from sync_batchnorm import convert_model
            model = convert_model(model)
            model = model.cuda()
        return model

    def _load_weights(self):
        state_dict = torch.load(self.config.model['weights'])
        if isinstance(state_dict, dict):
            print("Loading from checkpoint...")

            if len(state_dict['rng_state']['cuda']) > 1 and len(self.device.gpu_list) == 1:
                for save_key in state_dict.keys():
                    if "model_state_dict" in save_key:
                        state_dict[save_key] = OrderedDict(
                            [(k.replace('.module', ''), v) for k, v in state_dict[save_key].items()]
                        )
            if len(torch.cuda.get_rng_state_all()) == len(state_dict['rng_state']['cuda']):
                self.rng.set_rng_state(state_dict['rng_state'])
            self.model[self.model_mode].load_state_dict(state_dict['model_state_dict'], strict=False)
            self.config.optimizer['start_epoch'] = state_dict["epoch"] + 1
            self.global_step = state_dict['global_step']
            self.optimizer[self.model_mode].load_state_dict(state_dict['optimizer_state_dict'])
            print(self.optimizer[self.model_mode].scheduler.last_epoch)
            self.optimizer[self.model_mode].scheduler.last_epoch = self.config.optimizer['start_epoch']
            print(self.optimizer[self.model_mode].scheduler.last_epoch)
        else:
            print("Loading pretrained model...")
            for w in self.config.model['ignore_weights']:
                if state_dict.pop(w, None) is not None:
                    print('Sucessfully Remove Weights: {}.'.format(w))
                else:
                    print('Can Not Remove Weights: {}.'.format(w))
            self.model.load_state_dict(state_dict, strict=True)

    def _load_model(self):
        print('loading model...')
        self.device.set_device(self.config.device)

        model_class = import_class(self.config.model['class'])
        model = model_class(**self.config.model['args'])

        self.model = dict()
        self.model[self.model_mode] = self._model_to_device(model)
        print('model load finished!')

    def _load_optimizer(self):
        print('loading optimizer...')

        self.optimizer = dict()
        self.optimizer[self.model_mode] = Optimizer(
            self.model[self.model_mode].parameters(), self.config.optimizer
        )
        print('optimizer load finished!')

    def _load_criterion(self):
        print('loading criterion...')
        self.criterion = importlib.import_module(self.config.train['criterion']).get_criterion()
        print('criterion load finished!')
