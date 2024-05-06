 #!/bin/bash

export MODEL_NAME="resnet34"
python3 lightning_train.py \
    --train-data /home/tda/CARLA/TrafficGeneration/Datasets/Waymo_tf_example/Waymo_Prerendered_train \
    --val-data /home/tda/CARLA/TrafficGeneration/Datasets/Waymo_tf_example/Waymo_Prerendered_valid \
    --model ${MODEL_NAME}  \
    --save_path ./logs/${MODEL_NAME} \
    --img-res 224 \
    --in-channels 25 \
    --time-limit 80 \
    --n-traj 6 \
    --lr 0.001 \
    --batch-size 64 \
    --n-epochs 10
    
