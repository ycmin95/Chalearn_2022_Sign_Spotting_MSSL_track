import pdb
import numpy as np


class Evaluator(object):
    def eval_clip(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def eval_video(self, output, target, topk=(1,)):
        """
            output: [b, t, n]
            target: [b, t]
        """
        batch_size, temporal_size = target.size(0), target.size(1)
        maxk = max(topk)
        output = output.view(batch_size * temporal_size, -1)
        target = target.view(batch_size * temporal_size)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
    def read_file(self, path):
        labels, starts, ends = [], [], []
        for line in open(path, "r"):
            label, start, end = line[:-1].split(',')
            labels.append(int(label))
            starts.append(int(start) // 40)
            ends.append(int(end) // 40)
        return labels, starts, ends

    def generate_labels_start_end_time(self, framewise_labels, bg_class=['background']):
        labels = []
        starts = []
        ends = []
        last_label = framewise_labels[0]
        if framewise_labels[0] not in bg_class:
            labels.append(framewise_labels[0])
            starts.append(0)
        for i in range(len(framewise_labels)):
            if framewise_labels[i] != last_label:
                if framewise_labels[i] not in bg_class:
                    labels.append(framewise_labels[i])
                    starts.append(i)
                if last_label not in bg_class:
                    ends.append(i - 1)
                last_label = framewise_labels[i]
        if last_label not in bg_class:
            ends.append(i)
        for idx in range(len(labels)):
            labels[idx] = int(labels[idx]) - 1
        return labels, starts, ends

    def generate_framewise_labels(self, labels, starts, ends, video_length):
        framewise_labels = np.zeros(video_length)
        for idx in range(len(labels)):
            framewise_labels[starts[idx]:ends[idx] + 1] = labels[idx] + 1
        return framewise_labels

    def f_score(self, p_label, p_start, p_end, y_label, y_start, y_end, overlap):
        tp, fp = 0, 0

        hits = np.zeros(len(y_label))

        for j in range(len(p_label)):
            intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
            union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
            IoU = (1.0 * intersection / union) * ([p_label[j] == y_label[x] for x in range(len(y_label))])
            idx = np.array(IoU).argmax()
            
            if IoU[idx] >= overlap and not hits[idx]:
                tp += 1
                hits[idx] = 1
            else:
                fp += 1
        fn = len(y_label) - sum(hits)
        return float(tp), float(fp), float(fn)

    def eval_spotting(self, test_gt_paths, test_pred_paths):
        overlaps = np.arange(0.2, 0.8, 0.05)
        tps, fps, fns = 0, 0, 0
        for test_gt_path, test_pred_path in zip(test_gt_paths, test_pred_paths):
            y_label, y_start, y_end = self.read_file(test_gt_path)
            p_label, p_start, p_end = self.generate_labels_start_end_time(np.load(test_pred_path + '.npy'), [0])
            for overlap in overlaps:
                tp, fp, fn = self.f_score(p_label, p_start, p_end, y_label, y_start, y_end, overlap)
                tps += tp
                fps += fp
                fns += fn
        precision = 0.0 if (tps + fps) == 0 else tps / float(tps + fps)
        recall = 0.0 if (tps + fns) == 0 else tps / float(tps + fns)
        f1_score = 0.0 if (precision + recall) == 0.0 else 2.0 * (precision * recall) / (precision + recall)
        # print("Precision: {:.2f}, Recall: {:.2f}, F1 score: {:.2f}".format(precision, recall, f1_score))
        return precision, recall, f1_score
