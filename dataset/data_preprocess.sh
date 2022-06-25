#! /bin/bash

# unzip files to the target folder
unzip -P SEtNRzeHCHRA9xr6 ./MSSL_dataset/MSSL_Train_Set.zip -d ./MSSL_dataset/
unzip -P P4washbCad9FWJHB ./MSSL_dataset/MSSL_Val_Set.zip -d ./MSSL_dataset/
unzip -P PxHyvFux ./MSSL_dataset/MSSL_Test_set.zip -d ./MSSL_dataset/
unzip -P Eu8NdJDL ./MSSL_dataset/MSSL_Valid_gt.zip -d ./MSSL_dataset/

# step 1. extract frames from video
python ./preprocess/1_frame_extraction.py

# step 2. pose estimation
python ./preprocess/2_pose_estimation.py

# step 3. cropped video and input lists generation
python ./preprocess/3_input_list_generation.py

# step 4. generate skeleton data
python ./preprocess/4_skeleton_process.py

# step 5. optical flow estimation
python ./preprocess/5_optical_flow_extraction.py

#step 6. generate frame-wise and clip-wise label
python ./preprocess/6_label_generation.py
