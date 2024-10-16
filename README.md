# NGSIM-data-process-for-trajectory-prediction-planning
This is a NGSIM data process method for trajectory prediction and planning. The original NGSIM data is transformed to numpy frames (.npz). Each frame contains the ego vehicles' history trajectory, the nearby N agents' history trajectories, the map features and the ground truth of future trajectories.

## Step1. Data decompression
Firstly, you need to decompress pre-processed data in folder "data/". There are three csvs and they are collected from https://github.com/Rim-El-Ballouli/NGSIM-US-101-trajectory-dataset-smoothing.

## Step2. Installing dependency
You can check the python dependencies in scripts.

## Step3 Run "data_process.py"
In "data_process.py", you can choose to input any csv among folder "data/" and a saving path. The processed numpy frames will be saved automatically. Furthermore, you can choose if you need visualization.

## An visualization example
Here is an visualization example of a random numpy frame.
![图片](https://github.com/user-attachments/assets/ee8b9755-c7ed-495c-beb8-0d411dc265bf)

