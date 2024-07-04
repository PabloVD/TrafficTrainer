 #!/bin/bash

export MODEL_NAME="resnet18"
#export MODEL_NAME="xception71"

python3 lightning_train.py \
    --train-data /home/tda/CARLA/TrafficGeneration/Datasets/Waymo_tf_example/rendered_train_fixed \
    --val-data /home/tda/CARLA/TrafficGeneration/Datasets/Waymo_tf_example/rendered_valid_fixed \
    --model ${MODEL_NAME}  \
    --save_path ./logs/fixed_${MODEL_NAME} \
    --in-channels 25 \
    --time-limit 10 \
    --n-traj 6 \
    --lr 0.001 \
    --batch-size 64 \
    --n-epochs 10 \
    --num-devices 1
    
