import os
import pdb
import yaml
import argparse
from pprint import pprint
from functools import wraps


def get_parser():
    parser = argparse.ArgumentParser(
        description='The pytorch implementation for solution of '
                    'ECCV 22 Chalearn Sign Spotting Challenge (track 2: OSLWL).')
    parser.add_argument('--config', default='./config/train/video.yml', type=str,
                        help='to set the parameters')
    parser.add_argument('--device', default=0, type=str,
                        help='the indexes of GPUs for training or testing')
    parser.add_argument('--proj-name', default='', type=str,
                        help='the work folder for storing results')
    parser.add_argument('--phase', default='', type=str,
                        help='can be train or test')
    parser.add_argument('--framework', default='', type=str,
                        help='the initialization of model, weight and optimization')
    parser.add_argument('--data', default=dict(),
                        help='args about the model, log save')
    parser.add_argument('--segment-feeder', default=dict(),
                        help='args about the clip-wise dataloader')
    parser.add_argument('--video-feeder', default=dict(),
                        help='args about the video-wise dataloader')
    parser.add_argument('--train', default=dict(),
                        help='args about the training process')
    parser.add_argument('--test', default=dict(),
                        help='args about the training process')
    parser.add_argument('--decoder', default=dict(),
                        help='args about the decoder')
    parser.add_argument('--model', default=dict(),
                        help='args about the model')
    parser.add_argument('--optimizer', default=dict(),
                        help='args about the optimizer')
    parser.add_argument('--misc', default=dict(),
                        help='other useful configuration')
    return parser


def get_config():
    sparser = get_parser()
    p = sparser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            try:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                default_arg = yaml.load(f)
        default_arg = {k.lower(): v for k, v in default_arg.items()}
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        sparser.set_defaults(**default_arg)
    args = sparser.parse_args()

    print('======CONFIGURATION START======')
    pprint(args)
    print('======CONFIGURATION END======')
    return args
