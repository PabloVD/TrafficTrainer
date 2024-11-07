 #!/bin/bash

#export DATAPATH="/home/tda/CARLA/TrafficGeneration/Datasets/Waymo_tf_example/"
export DATAPATH="/data2/122-2/Datasets/Waymo_tf_example/"

export DATATRAIN=${DATAPATH}rendered_train_fixed_closedloop
export DATAVALID=${DATAPATH}rendered_train_fixed_closedloop

#export MODEL_NAME="resnet18"
#export MODEL_NAME="xception71"
#export MODEL_NAME="vit_large_patch32_224"
export MODEL_NAME="resnet18"

export LR=0.0001

export SCHEDULER="multistep"

export TIME_LIMIT=20

export LOSS="NLL"
#export LOSS="L2"
#export LOSS="L1"

export BATCHSIZE=50


python3 lightning_train.py \
    --train-data ${DATATRAIN} \
    --val-data ${DATAVALID} \
    --model ${MODEL_NAME}  \
    --loss ${LOSS} \
    --save_path ./logs/${LOSS}_${TIME_LIMIT}_${MODEL_NAME}_gru \
    --in-channels 23 \
    --time-limit ${TIME_LIMIT} \
    --n-traj 6 \
    --lr ${LR} \
    --batch-size $BATCHSIZE \
    --n-epochs 200 \
    --devices 0 1 \
    --scheduler ${SCHEDULER} \
    --wd 0.
    


