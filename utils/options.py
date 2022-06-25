import os
import pdb
import yaml
import argparse
from pprint import pprint
from functools import wraps
from easydict import EasyDict as edict


def get_parser():
    parser = argparse.ArgumentParser(
        description='The pytorch implementation for '
                    'ECCV 22 Chalearn Sign Spotting Challenge (track 2: OSLWL).')
    parser.add_argument('--config', default='./config/train/video.yml', type=str,
                        help='to set the parameters')
    parser.add_argument('--device', default=0, type=str,
                        help='supplementary info of the task, will be appended to project name')
    parser.add_argument('--proj-name', default='', type=str,
                        help='supplementary info of the task, will be appended to project name')
    parser.add_argument('--phase', default='', type=str,
                        help='supplementary info of the task, will be appended to project name')
    parser.add_argument('--framework', default='', type=str,
                        help='supplementary info of the task, will be appended to project name')
    parser.add_argument('--data', default=dict(),
                        help='data loader will be used')
    parser.add_argument('--segment-feeder', default=dict(),
                        help='data loader will be used')
    parser.add_argument('--video-feeder', default=dict(),
                        help='data loader will be used')
    parser.add_argument('--train', default=dict(),
                        help='data loader will be used')
    parser.add_argument('--test', default=dict(),
                        help='data loader will be used')
    parser.add_argument('--decoder', default=dict(),
                        help='data loader will be used')
    parser.add_argument('--model', default=dict(),
                        help='data loader will be used')
    parser.add_argument('--optimizer', default=dict(),
                        help='data loader will be used')
    parser.add_argument('--misc', default=dict(),
                        help='data loader will be used')
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
