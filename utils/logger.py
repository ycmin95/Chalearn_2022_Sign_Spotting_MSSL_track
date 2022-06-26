import os
import shutil

import torch


# from tensorboardX import SummaryWriter


class Logger:
    def __init__(self, ckpt_path, tsbd_path, global_step=0, best_metric_val=float('inf'), install_tsbd=False):
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        if not os.path.exists(tsbd_path):
            os.makedirs(tsbd_path)
        self.ckpt_path = ckpt_path
        self.install_tsbd = install_tsbd
        if install_tsbd:
            self.writer = SummaryWriter(tsbd_path)
        else:
            self.writer = None
        self.global_step = global_step
        self.best_metric_val = best_metric_val

    def step(self, step):
        self.global_step += step

    def add_scalar(self, name, val):
        if self.install_tsbd:
            self.writer.add_scalar(name, val, self.global_step)
        else:
            pass

    def add_image(self, name, img):
        if self.install_tsbd:
            self.writer.add_image(name, img, self.global_step)
        else:
            pass

    def add_histogram(self, tag, values, bins=1000):
        if self.install_tsbd:
            self.writer.add_histogram(tag, values, self.global_step, bins)
        else:
            pass

    def add_embedding(self, tag, feats, labels):
        """Log a graph of embeddings of given features with labels"""
        if self.install_tsbd:
            self.writer.add_embedding(mat=feats, tag=tag, metadata=labels, global_step=self.global_step)
        else:
            pass

    def add_graph(self, net, input_shape, device):
        if self.install_tsbd:
            dump_input = torch.rand(input_shape, device=device)
            self.writer.add_graph(net, (dump_input,), verbose=False)
        else:
            pass

    def add_figure(self, tag, figure):
        if self.install_tsbd:
            self.writer.add_figure(tag, figure, self.global_step)
        else:
            pass

    def close(self):
        if self.install_tsbd:
            self.writer.close()
        else:
            pass

    def save_ckpt(self, state, cur_metric_val):
        path_latest = os.path.join(self.ckpt_path, f'latest_{cur_metric_val}.pth')
        path_best = os.path.join(self.ckpt_path, 'best_model.pth')
        torch.save(state, path_latest)
        if cur_metric_val > self.best_metric_val:
            shutil.copyfile(path_latest, path_best)
            self.best_metric_val = cur_metric_val

    def save_ckpt_epoch(self, state, epoch):
        path_latest = os.path.join(self.ckpt_path, f'latest_{epoch}.pth')
        torch.save(state, path_latest)
