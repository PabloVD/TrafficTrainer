 #!/bin/bash

export DATAPATH="/home/tda/CARLA/TrafficGeneration/Datasets/Waymo_tf_example/"
#export DATAPATH="/data2/122-2/Datasets/Waymo_tf_example/"

#export DATATRAIN=${DATAPATH}tests_prerendered
#export DATAVALID=${DATAPATH}tests_prerendered
export DATATRAIN=${DATAPATH}rendered_train_fixed
export DATAVALID=${DATAPATH}rendered_train_fixed

export MODEL_NAME="resnet18"
export MODEL_NAME="xception71"
export MODEL_NAME="vit_base_patch16_224"

#export LR=0.0005
export LR=0.0001

export SCHEDULER="multistep"

#export FRAME="moving"
export FRAME="fixed"

export TIME_LIMIT=20

export LOSS="NLL"
#export LOSS="L2"
#export LOSS="L1"

export BATCHSIZE=20


python3 lightning_train.py \
    --train-data ${DATATRAIN} \
    --val-data ${DATAVALID} \
    --model ${MODEL_NAME}  \
    --loss ${LOSS} \
    --save_path ./logs/smalldata_${LOSS}_${TIME_LIMIT}_${MODEL_NAME} \
    --in-channels 23 \
    --time-limit ${TIME_LIMIT} \
    --n-traj 6 \
    --lr ${LR} \
    --batch-size $BATCHSIZE \
    --n-epochs 200 \
    --devices 0 \
    --scheduler ${SCHEDULER} \
    --wd 0.
    


