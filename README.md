# NGSIM-data-process-for-trajectory-prediction-planning
This is a NGSIM data process method for trajectory prediction and planning. The original NGSIM data is transformed to numpy frames (.npz). Each frame contains the ego vehicles' history trajectory, the nearby N agents' history trajectories, the map features and the ground truth of future trajectories.

## Step1. Data decompression
Firstly, you need to decompress pre-processed data in folder "data/". There are three csvs and they are collected from https://github.com/Rim-El-Ballouli/NGSIM-US-101-trajectory-dataset-smoothing.

##Step2. Installing dependency
