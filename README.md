# This repo holds codes of the solution for the [Sign Spotting Challenge](https://chalearnlap.cvc.uab.cat/challenge/49/description/) at ECCV (Multi-Shot Supervised Learning track)
Our team ranked 3rd in the final test phase. 

## Reproduce Result and Conduct Relevant Experiments
We produce the extracted features from multuple modalities for sign spotting, which can be trained in ten minutes and achieve acceptable performance. To reproduce the result, you need:

### 1. Preparation
Our solution is based on the basic opencv and pytorch and provide requirements for conda environment:
```bash
conda create --name <env> --file requirements.txt
```
 
The [extracted features](https://drive.google.com/file/d/1fJGWMbelVOy5Em-dxa6E3mHxXQpH_7OU/view?usp=sharing) and the [trained model](https://drive.google.com/file/d/1gSf3MNYrmS-n4dwtooO9TNakKc7Er8dG/view?usp=sharing) can be downloaded from Google Drive. After download the extracted features, unzip them in to the dataset folder:
```bash
unzip extracted features.zip -d ./dataset/
```
The main desired directory tree is expected as follows,
```
.
├── configs
│   ├── test
│   └── train
├── dataset
│   ├── data_preprocess.sh
│   └── MSSL_dataset
│        ├── final_train_input.txt
│        ├── train_input.txt
│        └── valid_input.txt
│        ├── test_input.txt
│        ├── TRAIN
│        │   ├── MSSL_TRAIN_SET_GT.pkl
│        │   └── MSSL_TRAIN_SET_GT_TXT
│        ├── VALIDATION
│        │   ├── MSSL_VAL_SET_GT.pkl
│        │   └── MSSL_VAL_SET_GT_TXT
│        └── processed
│             ├── features
│             │   ├── flow
│             │   ├── mask_video
│             │   ├── skeleton
│             │   └── video
│             ├── test
│             │   ├── clipwise_label
│             │   └── framewise_label
│             ├── train
│             │   ├── clipwise_label
│             │   └── framewise_label
│             └── valid
│                 ├── clipwise_label
│                 └── framewise_label
├── weights 
│       ├──final_model.pth
└── submission
         ├── ref
         │   └── ground_truth.pk
         └── res
              └── predictions.pkl
```
### 2. Evaluation
For evaluation with the provided model (./final_model.pth), simply run:
```
python generate_predictions
``` 
The final prediction can be found in `submission/prediction_validate/res/predictions.pkl`.

For training the final spotting model with the extracted features, simply run:
```
python main.py --config ./configs/train/fusion_detector.yml
```

### 3. Feature Extraction and Training (if needed)
The Feature Extraction process generate cropped video, optical flow, skeleton and masked video. To obtain skeleton data, we adopt [mediapipe](https://github.com/google/mediapipe) for pose and hands estimation, which should be installed first.   

Download the data set provided in challenge and put them in `./dataset/`, then run the script:
```
cd dataset
bash data_preprocess.sh
```

The organization is expected as follows:
```
dataset
├── data_preprocess.sh
└── MSSL_dataset
   ├── final_train_input.txt
   ├── train_input.txt
   ├── valid_input.txt
   ├── test_input.txt
   ├── TRAIN
   ├── VALIDATION
   ├── MSSL_TEST_SET_VIDEOS
   └── processed
        ├── train_pose.pkl
        ├── valid_pose.pkl
        ├── test_pose.pkl
        ├── train
        │   ├── original_video
        │   ├── video
        │   ├── flow
        │   ├── pose
        │   ├── clipwise_label
        │   └── framewise_label
        ├── valid
        │   ├── original_video
        │   ├── video
        │   ├── flow
        │   ├── pose
        │   ├── clipwise_label
        │   └── framewise_label
        └── test
             ├── original_video
             ├── video
             ├── flow
             ├── pose
             ├── clipwise_label
             └── framewise_label
```

We adopt a two-round training scheme for feature extraction, in the first round, 
only a subset that contains clips of query signs is built to increase the 
discriminative ability of the backbone. On the second round, 
all clips are used for training.  
 
For the first round training, run the command:
```bash
python main.py --config ./configs/train/video_config.yml
python main.py --config ./configs/train/mask_video_config.yml
python main.py --config ./configs/train/skeleton_config.yml
python main.py --config ./configs/train/skeleton_config.yml
```

Then select the best (validation) or the last (test) weight for the next round training, by modifying the `remove_bg=False` and `weights=<path_to_best_weight>`in the config files, and run the above command again.

For the feature extraction, modify the weight path in `feature_extraction`, and run
```bash
python feature_extraction.py
``` 
which will generate feats as step 1 shown.

Team leader: Xilin Chen 

Team member: Yuecong Min, [Peiqi Jiao](https://github.com/Page-Jiao), [Aiming Hao](https://github.com/hamcoder)

#### Relevant Repos
[I3D code and pretrained model](https://github.com/Tushar-N/pytorch-resnet3d)   
[P3D code and pretrained model](https://github.com/qijiezhao/pseudo-3d-pytorch)    
[ST-GCN](https://github.com/yysijie/st-gcn)

For more information, please contact Yuecong Min (yuecong.min [AT] vipl.ict.ac.cn)