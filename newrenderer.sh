rm /home/tda/CARLA/TrafficGeneration/Datasets/Waymo_tf_example/tests_prerendered/*

python3 newrenderer.py \
    --data /home/tda/CARLA/TrafficGeneration/Datasets/Waymo_tf_example/tests_raw \
    --out /home/tda/CARLA/TrafficGeneration/Datasets/Waymo_tf_example/tests_prerendered \
    --n-jobs 1 \
    #--use-vectorize \

