import os
import pdb
import torch
import pickle
import numpy as np
from tqdm import tqdm
from itertools import groupby
from collections import OrderedDict
from utils.decoder import Decoder
from utils import get_config, import_class
from stage1_extract_feature.models import resnet
from stage1_extract_feature.models import tmodel
from stage1_extract_feature.models import fmodel
from stage1_extract_feature.models import st_gcn_seq
from stage2_sign_spotting.models import detection_fix
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


if __name__ == "__main__":
    visible_device = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)
    device = 0

    # video, mask_video, flow, skeleton
    modality = ["video", "mask_video", "flow", "skeleton"]
    modality_idx = [2]

    weights_list = [
        "./trained_model/video.pth",
        "./trained_model/mask_video.pth",
        # "./trained_model/flow.pth",
        "./trained_model/new_flow.pth",
        "./trained_model/skeleton.pth",
        # "/home/ycmin/skeleton/Sign_Spotting/experiment/ckpt/0619_retrain_video_baseline/latest_90.39248328207444.pth",
        # "/home/ycmin/skeleton/Sign_Spotting/experiment/ckpt/0619_retrain_mask_video_baseline/latest_90.0821082926554.pth",
        # "/home/ycmin/skeleton/Sign_Spotting/experiment/ckpt/0619_retrain_flow_baseline/latest_88.64579510433691.pth",
        # "/home/ycmin/skeleton/Sign_Spotting/experiment/ckpt/0619_retrain_skeleton_baseline/latest_85.25718800259587.pth",
    ]

    module_list = [
        tmodel.TModel,
        tmodel.TModel,
        fmodel.TModel,
        st_gcn_seq.SeqSTGCN,
    ]

    args_list = [
        {
            "backbone": 'i3d',
            "num_classes": 61,
        },
        {
            "backbone": 'i3d',
            "num_classes": 61,
        },
        {
            "backbone": 'p3d',
            "num_classes": 61,
        },
        {
            "num_classes": 61,
            "in_channels": 3,
            "dropout": 0.5,
            "edge_importance_weighting": True,
            "temporal_kernel_size": 9,
            "graph_args": {
                "layout": 'mediapipe',
                "strategy": 'spatial',
            }
        },
    ]

    model_list = []
    feeder_list = []

    for weight_path, module, args in zip(weights_list, module_list, args_list):
        print(args)
        model = module(**args)
        weights = torch.load(weight_path)['model_state_dict']
        weights = OrderedDict(
            [(k.replace('module.', ''), v) for k, v in weights.items()]
        )
        model.load_state_dict(weights, strict=True)
        model = model.to(device)
        model.eval()
        model_list.append(model)

    video_feeder = import_class("dataset.MSSLVideoFeeder")
    feeder_args = [
        {
            "data_type": "video",
            "mode": "test",
            "transform_mode": False,
        },
        {
            "data_type": "video",
            "mode": "test",
            "mask": True,
            "transform_mode": False,
        },
        {
            "data_type": "flow",
            "mode": "valid",
            "transform_mode": False,
        },
        {
            "data_type": "skeleton",
            "mode": "test",
            "transform_mode": False,
        },
    ]
    for args in feeder_args:
        data_loader = torch.utils.data.DataLoader(
            dataset=video_feeder(**args),
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=10,
            pin_memory=False
        )
        print(len(data_loader))
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
    for mod_idx in modality_idx:
        for batch_idx, data in enumerate(tqdm(feeder_list[mod_idx])):
            video_file_name = data[-1][0][0]
            start_frame = data[-1][1][0]
            end_frame = data[-1][2][0]
            label = data[1]
            inputs = data[0].to(device)

            # print("Label: ", [x[0] for x in groupby(label[0].numpy().astype(int))])
            print(video_file_name)
            test_gt_paths.append(f'./dataset/MSSL_dataset/VALIDATION/MSSL_VAL_SET_GT_TXT/{video_file_name}.txt')
            save_path = os.path.join(save_root, video_file_name)
            test_pred_paths.append(save_path)

            results = []
            with torch.no_grad():
                outputs, feats = model_list[mod_idx](inputs, video=True)
                if len(outputs.shape) == 2:
                    outputs = outputs[:, None]
                recog_result, probs = decoder. \
                    decode_with_probs(outputs, start_frame, end_frame, save_path)
                if not os.path.exists(f"./features_old/{modality[mod_idx]}"):
                    os.makedirs(f"./features_old/{modality[mod_idx]}")
                np.save(f"./features_old/{modality[mod_idx]}/{video_file_name}.npy", feats[:, :, 0].cpu().detach().numpy())
                split_results = evaluator.generate_labels_start_end_time(recog_result)
                print(modality[mod_idx], [x[0] for x in groupby(recog_result.astype(int))])
                results.append([split_results, probs])
            split_results = results[0][0]
            pred_pickle_file[video_file_name] = []
            for idx, item in enumerate(split_results[0]):
                if item != -1:
                    pred_pickle_file[video_file_name].append(
                        [item, split_results[1][idx] * 40, split_results[2][idx] * 40]
                    )

        # gt_pkl_path = "./dataset/MSSL_dataset/VALIDATION/MSSL_VAL_SET_GT.pkl"
        # gt_pkl_path = os.path.abspath(gt_pkl_path)
        # if not os.path.exists(f"{save_root}/ref"):
        #     os.makedirs(f"{save_root}/ref")
        # if not os.path.exists(f"{save_root}/res"):
        #     os.makedirs(f"{save_root}/res")
        # with open(f"{save_root}/res/predictions.pkl", 'wb') as handle:
        #     pickle.dump(pred_pickle_file, handle, protocol=4)
        # if not os.path.exists(f"{save_root}/ref/ground_truth.pkl"):
        #     os.system(f"ln -s {gt_pkl_path} {save_root}/ref/ground_truth.pkl")
        # print("Official evaluation results: ")
        # official_evaluator(folder_in=save_root, folder_out=save_root)
