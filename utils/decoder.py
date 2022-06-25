import pdb
import numpy as np


class Decoder(object):
    def __init__(self, stride, duration, bg_class):
        self.stride = stride
        self.duration = duration
        self.bg_class = bg_class

    def decode(self, pred, start_frame, end_frame, save_path):
        pred = pred.cpu().numpy()
        recog = np.zeros(end_frame + 1)
        model_recog = self.max_decode(pred)
        for pos in model_recog.nonzero()[0]:
            left = start_frame + max(self.stride * pos - self.duration // 2, 0)
            right = min(start_frame + self.stride * pos + self.duration // 2, end_frame + 1)
            recog[left:right] = model_recog[pos]
        self.save_file(recog, save_path)
        return recog

    def decode_with_probs(self, pred, start_frame, end_frame, save_path):
        pred = pred.softmax(-1)
        pred = pred.cpu().numpy()
        recog = np.zeros(end_frame + 1)
        probs = np.zeros((end_frame + 1, pred.shape[-1]))
        model_recog = self.max_decode(pred)
        for pos in model_recog.nonzero()[0]:
            left = start_frame + max(self.stride * pos - self.duration // 2, 0)
            right = min(start_frame + self.stride * pos + self.duration // 2, end_frame + 1)
            recog[left:right] = model_recog[pos]
            probs[left:right] = pred[pos]
        self.save_file(recog, save_path)
        return recog, probs

    def max_decode(self, pred):
        results = np.argmax(pred, axis=-1)
        results[results > 60] = 0
        return results

    def save_file(self, recog, save_path):
        np.save(save_path, recog)

    def save_file(self, recog, save_path):
        np.save(save_path, recog)

    def decode_feature(self, preds, offset, start_frame, end_frame, save_path):
        preds = preds.cpu().numpy()
        length, num_class = preds.shape
        offset = np.around(offset.cpu().numpy(), 0).astype(int)
        results = np.argmax(preds, axis=-1)
        results[results > 60] = 0
        # print(results)
        # print(offset)
        # length = results.shape[0]
        vote = np.zeros((length, num_class))
        for position in results.nonzero()[0]:
            start, end = max(position - offset[position][0], 0), min(position + offset[position][1], length - 1)
            vote[start:end + 1, results[position]] += 1
        # print(vote)
        model_recog = np.argmax(vote, axis=-1)
        recog = np.zeros(end_frame + 1)
        for pos in model_recog.nonzero()[0]:
            left = start_frame + max(self.stride * pos - self.duration // 2, 0)
            right = min(start_frame + self.stride * pos + self.duration // 2, end_frame + 1)
            recog[left:right] = model_recog[pos]
        self.save_file(recog, save_path)
        return recog
