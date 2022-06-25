import os
import pdb
import glob
import torch
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
    # print([[predictions[0][0][0][k], predictions[0][0][1][k], predictions[0][0][2][k]] for k in
    #        range(len(predictions[0][0][0])) if predictions[0][0][0][k] != -1])
    # print([[predictions[1][0][0][k], predictions[1][0][1][k], predictions[1][0][2][k]] for k in
    #        range(len(predictions[1][0][0])) if predictions[1][0][0][k] != -1])
    print(ret_results[0])
    print([[ret_results[0][k], ret_results[1][k], ret_results[2][k]] for k in
           range(len(ret_results[0])) if ret_results[0][k] != -1])
    return ret_results

    # def merge_results(predictions):
    #     # class, start_frame, end_frame
    #     final_results = [[], [], [], [], []]
    #     flag = dict()
    #     for idx, prediction in enumerate(predictions):
    #         pred, probs = prediction
    #         for i in range(len(pred[0])):
    #             if pred[0][i] != -1:
    #                 if idx == len(predictions) - 1:
    #                     flag[len(final_results[0])] = True
    #                 final_results[0].append(pred[0][i])
    #                 final_results[1].append([pred[1][i]])
    #                 final_results[2].append([pred[2][i]])
    #     for idx, item in enumerate(final_results[0]):
    #         if idx in flag.keys():
    #             for idx_tmp, item_tmp in enumerate(final_results[0]):
    #                 if idx_tmp not in flag.keys() and item_tmp != -1 and \
    #                         overlap(final_results, idx, final_results, idx_tmp):
    #                     flag[idx_tmp] = False
    # for idx, item in enumerate(final_results[0]):
    #     if idx not in flag.keys():
    #         # final_results[1][idx].append(final_results[1][idx])
    #         # final_results[2][idx].append(final_results[2][idx])
    #         flag[idx] = True
    #         for idx_tmp, item_tmp in enumerate(final_results[0]):
    #             if idx_tmp not in flag.keys():
    #                 if item_tmp != -1 and overlap(final_results, idx, final_results, idx_tmp):
    #                     flag[idx_tmp] = False
    # ret_results = [[], [], []]
    # try:
    #     for k, v in flag.items():
    #         if v:
    #             # if final_results[3][k] > 0.8:
    #             ret_results[0].append(final_results[0][k])
    #             ret_results[1].append(int(np.mean(final_results[1][k])))
    #             ret_results[2].append(int(np.mean(final_results[2][k])))
    #             # ret_results[3].append(final_results[3][k])
    # except IndexError:
    #     pdb.set_trace()
    # pdb.set_trace()
    # return ret_results


if __name__ == "__main__":
    visible_device = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)
    device = 0

    # video, mask_video, flow, skeleton
    modality = ["fusion"] * 5
    eval_modality = [4]
    modality = [modality[idx] for idx in eval_modality]

    weights_list = [
        # "experiment/ckpt/0624_final_trainv2_modality_0/latest_0.8516279865427382.pth",
        "experiment/ckpt/0624_final_trainv2_modality_1/latest_0.8463529077207452.pth",
        "experiment/ckpt/0624_final_trainv2_modality_2/latest_0.8507932871298471.pth",
        "experiment/ckpt/0624_final_trainv2_modality_3/latest_0.8442400525579138.pth",
        "experiment/ckpt/0623_final_old_feats_rd1/latest_0.5744001647616108.pth",
        "experiment/ckpt/0623_final_new_feats/latest_0.8573551263001487.pth",
    ]

    module_list = [
                      detection_test.Detector,
                  ] * len(weights_list)

    args_list = [
        {
            "modality": [1],
            "input_dim": 512,
            "hidden_dim": 512,
            "num_classes": 62,
            "dropout": 0.3,
        },
        {
            "modality": [2],
            "input_dim": 512,
            "hidden_dim": 512,
            "num_classes": 62,
            "dropout": 0.3,
        },
        {
            "modality": [3],
            "input_dim": 512,
            "hidden_dim": 512,
            "num_classes": 62,
            "dropout": 0.3,
        },
        {
            "modality": [0, 1, 2, 3],
            "input_dim": 512,
            "hidden_dim": 512,
            "num_classes": 62,
            "dropout": 0.3,
        },
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
            dataset=feature_feeder(data_type='fusion_feature', mode='test', feature_dir='features_new'),
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=False
        )
        feeder_list.append(data_loader)
    feeder_list[-2] = torch.utils.data.DataLoader(
        dataset=feature_feeder(data_type='fusion_feature', mode='test'),
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False
    )
    decoder = Decoder(stride=4, duration=8, bg_class=0)
    evaluator = Evaluator()
    official_evaluator = OffEvaluator

    pred_pickle_file = dict()
    test_gt_paths = []
    test_pred_paths = []
    save_root = "./submission/prediction_validate"
    for batch_idx, data in enumerate(tqdm(zip(*[feeder_list[idx] for idx in eval_modality]))):
        video_file_name = data[0][-1][0][0]
        start_frame = data[0][-1][2][0]
        end_frame = data[0][-1][3][0]
        label = data[0][1]

        # print("Label: ", [x[0] for x in groupby(label[0].numpy().astype(int))])
        print(video_file_name)
        # test_gt_paths.append(f'./dataset/MSSL_dataset/VALIDATION/MSSL_VAL_SET_GT_TXT/{video_file_name}.txt')
        test_gt_paths.append(f'./dataset/MSSL_dataset/TRAIN/MSSL_TRAIN_SET_GT_TXT/{video_file_name}.txt')
        save_path = os.path.join(save_root, video_file_name)
        test_pred_paths.append(save_path)

        results = []
        with torch.no_grad():
            for meta_data, model, mod in zip(data, [model_list[idx] for idx in eval_modality], modality):
                inputs = meta_data[0].to(device)
                print(inputs.shape, data[0][-1])
                # if not os.path.exists(f"./features_old/{mod}"):
                #     os.makedirs(f"./features_old/{mod}")
                # np.save(f"./features_old/{mod}/{video_file_name}.npy", feats[:, :, 0].cpu().detach().numpy())
                if mod == 'fusion':
                    outputs, feats = model(inputs, len_x=torch.LongTensor([inputs.shape[1]]).to(device))
                    recog_result = decoder. \
                        decode_feature(outputs, feats, start_frame, end_frame, save_path)
                    probs = None
                else:
                    outputs, feats = model(inputs, video=True)
                    if len(outputs.shape) == 2:
                        outputs = outputs[:, None]
                    recog_result, probs = decoder. \
                        decode_with_probs(outputs, start_frame, end_frame, save_path)
                split_results = evaluator.generate_labels_start_end_time(recog_result)
                print(mod, [x[0] for x in groupby(recog_result.astype(int))])
                # import ctcdecode
                # vocab = [chr(x) for x in range(20000, 20000 + 61)]
                # ctc_decoder = ctcdecode.CTCBeamDecoder(vocab, beam_width=10, blank_id=0,
                #                                        num_processes=10)
                # beam_result, beam_scores, timesteps, out_seq_len = \
                #     ctc_decoder.decode(outputs.permute(1, 0, 2).softmax(-1).cpu(), torch.LongTensor([len(outputs)]))
                # print(beam_result[0][0][:out_seq_len[0][0]])
                # pdb.set_trace()
                results.append([split_results, probs])
        if len(results) > 1:
            split_results = merge_results(results)
        else:
            print("in")
            split_results = results[0][0]
        pred_pickle_file[video_file_name] = []
        for idx, item in enumerate(split_results[0]):
            if item != -1:
                pred_pickle_file[video_file_name].append(
                    [item, split_results[1][idx] * 40, split_results[2][idx] * 40]
                )
    import pickle

    # gt_pkl_path = "/home/ycmin/skeleton/Sign_Spotting/dataset/MSSL_dataset/VALIDATION/MSSL_VAL_SET_GT.pkl"
    gt_pkl_path = "dataset/MSSL_dataset/TRAIN/MSSL_TRAIN_SET_GT.pkl"
    if not os.path.exists(f"{save_root}/ref"):
        os.makedirs(f"{save_root}/ref")
    if not os.path.exists(f"{save_root}/res"):
        os.makedirs(f"{save_root}/res")
    with open(f"{save_root}/res/predictions.pkl", 'wb') as handle:
        pickle.dump(pred_pickle_file, handle, protocol=4)
    if not os.path.exists(f"{save_root}/ref/ground_truth.pkl"):
        os.system(f"ln -s {gt_pkl_path} {save_root}/ref/ground_truth.pkl")
    print("Official evaluation results: ")
    official_evaluator(folder_in=save_root, folder_out=save_root)
    # print("Custom evaluation results: ")
    # precision, recall, f1_score = evaluator.eval_spotting(test_gt_paths, test_pred_paths)
    # print("Validate done. Precision: {:.2f}, Recall: {:.2f}, F1 score: {:.2f}".format(precision, recall, f1_score))
