import os
import pdb
import argparse
import numpy as np
import evaluator.evaluation_utils as evaluation_utils
import pickle


def create_folder(folder):
    print('create_folder: {}'.format(folder))
    try:
        os.makedirs(folder)
        print("Directory ", folder, " Created ")
    except FileExistsError:
        print("Directory ", folder, " already exists")


def partial_evaluate(p, gt, threshold_iou):
    global_tp = 0
    global_fp = 0
    global_fn = 0
    # only evaluate files in prediction
    # for fidx in gt.keys():
    # correct, gt, pred
    correct_class = {i: [0, 0, 0] for i in range(61)}

    for fidx in p.keys():
        data_gt = np.asarray(gt[fidx])
        try:
            data_p = np.asarray(p[fidx])
        except:
            data_p = np.array([])

        data_performances = np.array(
            [evaluation_utils.extract_performances(data_gt, data_p, iou_idx, correct_class)
             for iou_idx in threshold_iou]
        )  # [0]

        tp = np.mean(data_performances[:, 0])
        fp = np.mean(data_performances[:, 1])
        fn = np.mean(data_performances[:, 2])

        global_tp = global_tp + tp
        global_fp = global_fp + fp
        global_fn = global_fn + fn

    # for k, v in correct_class.items():
    #     recall = 0 if v[1] == 0 else v[0] / v[1]
    #     precious = 0 if v[2] == 0 else v[0] / v[2]
    #     print(f"{k} \t {v[0]} \t {v[1]} \t {v[2]} \t Recall: {recall:.2f} \t Precious: {precious:.2f}")

    avg_precision, avg_recall, avg_f1 = evaluation_utils.calculate_metrics(global_tp, global_fp, global_fn)
    print('********************************************************************************')
    print('TOTAL_TRAINING_SAMPLES: {}'.format(len(gt.keys())))
    print('TOTAL_EVALUATED_SAMPLES: {}'.format(len(p.keys())))
    print(' -- global_tp: {}'.format(str(global_tp).replace('.', ',')))
    print(' -- global_fp: {}'.format(str(global_fp).replace('.', ',')))
    print(' -- global_fn: {}'.format(str(global_fn).replace('.', ',')))

    print(' -- global_precision: {}'.format(str(avg_precision).replace('.', ',')))
    print(' -- global_recall: {}'.format(str(avg_recall).replace('.', ',')))
    print(' -- global_f1: {}'.format(str(avg_f1).replace('.', ',')))
    print('********************************************************************************')
    return avg_f1, avg_precision, avg_recall, correct_class


def evaluate(folder_in, folder_out, pkl_file='predictions.pkl', threshold_iou_min=0.2, threshold_iou_max=0.8,
             threshold_iou_step=0.05, args=None):
    if args is not None:
        folder_in = args.input
        folder_out = args.output
        threshold_iou_min = args.threshold_iou_min
        threshold_iou_max = args.threshold_iou_max
        threshold_iou_step = args.threshold_iou_step
    if threshold_iou_max > threshold_iou_min:
        threshold_iou = np.arange(start=threshold_iou_min, stop=threshold_iou_max + (threshold_iou_step - 0.0001),
                                  step=threshold_iou_step)
    else:
        threshold_iou = np.array([threshold_iou_min])

    print('folder_in: {}'.format(folder_in))
    print('folder_out: {}'.format(folder_out))
    print('threshold_iou: {}'.format(threshold_iou))

    # folder_out_metrics = os.path.join(folder_out, 'metrics')
    # create_folder(folder_out_metrics)

    ref_path = os.path.join(folder_in, 'ref')
    res_path = os.path.join(folder_in, 'res')

    with open(os.path.join(ref_path, 'ground_truth.pkl'), 'rb') as f:
        gt = pickle.load(f, encoding='bytes')

    with open(os.path.join(res_path, pkl_file), 'rb') as f:
        p = pickle.load(f, encoding='bytes')

    avg_f1, avg_precision, avg_recall, ret = partial_evaluate(p, gt, threshold_iou)
    with open(os.path.join(folder_out, 'scores.txt'), 'w') as f:
        f.write('{}\n'.format(avg_f1))
    signer_list = sorted(set([info.split("_")[0] for info in [*p.keys()]]))
    with open(os.path.join(folder_out, 'scores.txt'), 'a') as f:
        for signer in signer_list:
            print(f"Signer {signer} evaluation results:")
            signer_prediction = dict([*filter(lambda x: signer in x[0], p.items())])
            _, _, _, _ = partial_evaluate(signer_prediction, gt, threshold_iou)
            f.write(f'Signer {signer}: {avg_f1}\n')
    return avg_f1, avg_precision, avg_recall


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Result Output')
    parser.add_argument('--input', required=True, default='', type=str)
    parser.add_argument('--output', required=True, default='', type=str)
    parser.add_argument('--threshold_iou_min', required=False, default=0.2, type=float)
    parser.add_argument('--threshold_iou_max', required=False, default=0.8, type=float)
    parser.add_argument('--threshold_iou_step', required=False, default=0.05, type=float)
    arg = parser.parse_args()
    evaluate(arg)

# python evaluate.py --input ./input --output ./output --threshold_iou_min 0.2 --threshold_iou_max 0.8 --threshold_iou_step 0.05
