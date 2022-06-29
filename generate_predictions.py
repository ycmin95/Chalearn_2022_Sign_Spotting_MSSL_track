import os
import pdb
import glob
import torch
import pickle
import numpy as np
from tqdm import tqdm
from itertools import groupby
from collections import OrderedDict
from utils.decoder import Decoder
from utils import get_config, import_class
from stage2_sign_spotting.models import detection_fix, detection_test
from evaluator.evaluator import Evaluator
from evaluator.official_evaluator import evaluate as OffEvaluator


def overlap(pred1, idx1, pred2, idx2):
    try:
        if np.mean(pred2[1][idx2]) <= np.mean(pred1[1][idx1]) <= np.mean(pred2[2][idx2]):
            return True
        if np.mean(pred2[1][idx2]) <= np.mean(pred1[2][idx1]) <= np.mean(pred2[2][idx2]):
            return True
    except ValueError:
        pdb.set_trace()
    return False


def merge_results(predictions):
    # class, start_frame, end_frame
    final_results = [[], [], [], [], []]
    flag = dict()
    for idx, prediction in enumerate(predictions[::-1]):
        pred, probs = prediction
        for i in range(len(pred[0])):
            if pred[0][i] != -1:
                final_results[0].append(pred[0][i])
                final_results[1].append([pred[1][i]])
                final_results[2].append([pred[2][i]])
    for idx, item in enumerate(final_results[0]):
        if idx not in flag.keys():
            for pred_idx, prediction in enumerate(predictions[:-1]):
                pred, probs = prediction
                for idx_tmp, item_tmp in enumerate(pred[0]):
                    if item_tmp != -1 and item_tmp == item and overlap(final_results, idx, pred, idx_tmp):
                        final_results[1][idx].append(pred[1][idx_tmp])
                        final_results[2][idx].append(pred[2][idx_tmp])
            flag[idx] = True
            for idx_tmp, item_tmp in enumerate(final_results[0]):
                if idx_tmp not in flag.keys():
                    if item_tmp != -1 and overlap(final_results, idx, final_results, idx_tmp):
                        flag[idx_tmp] = False
    ret_results = [[], [], []]
    for k, v in flag.items():
        if v:
            ret_results[0].append(final_results[0][k])
            ret_results[1].append(int(np.mean(final_results[1][k])))
            ret_results[2].append(int(np.mean(final_results[2][k])))
    return ret_results


if __name__ == "__main__":
    visible_device = 0
    split = 'test'

    os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)
    device = 0
    eval_weights = [0]

    weights_list = [
        "./weights/final_model.pth",
    ]
    weights_cnt = len(weights_list)

    # video, mask_video, flow, skeleton, fusion
    modality = ["fusion"]
    eval_modality = [modality[idx] for idx in eval_weights]

    module_list = [
        detection_test.Detector,
    ]

    args_list = [
        {
            "modality": [0, 1, 2, 3],
            "input_dim": 512,
            "hidden_dim": 512,
            "num_classes": 62,
            "dropout": 0.3,
        },
    ]

    model_list = []
    feeder_list = []

    for weight_path, module, args in zip(weights_list, module_list, args_list):
        print(args)
        model = module(**args)
        weights = torch.load(weight_path)['model_state_dict']
        model.load_state_dict(weights, strict=True)
        model = model.to(device)
        model.eval()
        model_list.append(model)

    feature_feeder = import_class("dataset.MSSLFeatureClipFeeder")
    for i in range(len(weights_list)):
        data_loader = torch.utils.data.DataLoader(
            dataset=feature_feeder(data_type='fusion_feature', mode=split, feature_dir='features'),
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=False
        )
        feeder_list.append(data_loader)
    decoder = Decoder(stride=4, duration=8, bg_class=0)
    evaluator = Evaluator()
    official_evaluator = OffEvaluator

    pred_pickle_file = dict()
    test_gt_paths = []
    test_pred_paths = []
    save_root = "./submission/prediction_validate"
    if not os.path.exists(f"{save_root}"):
        os.makedirs(f"{save_root}")
    for batch_idx, data in enumerate(tqdm(zip(*[feeder_list[idx] for idx in eval_weights]))):
        video_file_name = data[0][-1][0][0]
        start_frame = data[0][-1][2][0]
        end_frame = data[0][-1][3][0]
        label = data[0][1]

        # print("Label: ", [x[0] for x in groupby(label[0].numpy().astype(int))])
        print(video_file_name)
        if split == 'train':
            test_gt_paths.append(f'./dataset/MSSL_dataset/TRAIN/MSSL_TRAIN_SET_GT_TXT/{video_file_name}.txt')
        elif split == 'valid':
            test_gt_paths.append(f'./dataset/MSSL_dataset/VALIDATION/MSSL_VAL_SET_GT_TXT/{video_file_name}.txt')
        save_path = os.path.join(save_root, video_file_name)
        test_pred_paths.append(save_path)

        results = []
        with torch.no_grad():
            for meta_data, model, mod in zip(data, [model_list[idx] for idx in eval_weights], modality):
                inputs = meta_data[0].to(device)
                print(inputs.shape, data[0][-1])
                outputs, feats = model(inputs, len_x=torch.LongTensor([inputs.shape[1]]).to(device))
                recog_result = decoder. \
                    decode_feature(outputs, feats, start_frame, end_frame, save_path)
                probs = None
                split_results = evaluator.generate_labels_start_end_time(recog_result)
                print(mod, [x[0] for x in groupby(recog_result.astype(int))])
                results.append([split_results, probs])
        if len(results) > 1:
            split_results = merge_results(results)
        else:
            split_results = results[0][0]
        pred_pickle_file[video_file_name] = []
        for idx, item in enumerate(split_results[0]):
            if item != -1:
                pred_pickle_file[video_file_name].append(
                    [item, split_results[1][idx] * 40, split_results[2][idx] * 40]
                )

    if split in ['train', 'valid']:
        if split == 'train':
            gt_pkl_path = "./dataset/MSSL_dataset/TRAIN/MSSL_TRAIN_SET_GT.pkl"
        if split == 'valid':
            gt_pkl_path = "./dataset/MSSL_dataset/VALIDATION/MSSL_VAL_SET_GT.pkl"
        gt_pkl_path = os.path.abspath(gt_pkl_path)
        if not os.path.exists(f"{save_root}/ref"):
            os.makedirs(f"{save_root}/ref")
        if not os.path.exists(f"{save_root}/res"):
            os.makedirs(f"{save_root}/res")
        with open(f"{save_root}/res/predictions.pkl", 'wb') as handle:
            pickle.dump(pred_pickle_file, handle, protocol=4)
        if os.path.exists(f"{save_root}/ref/ground_truth.pkl"):
            os.system(f"unlink {save_root}/ref/ground_truth.pkl")
        os.system(f"ln -s {gt_pkl_path} {save_root}/ref/ground_truth.pkl")
        print("Official evaluation results: ")
        official_evaluator(folder_in=save_root, folder_out=save_root)
    else:
        if not os.path.exists(f"{save_root}/res"):
            os.makedirs(f"{save_root}/res")
        with open(f"{save_root}/res/predictions.pkl", 'wb') as handle:
            pickle.dump(pred_pickle_file, handle, protocol=4)
