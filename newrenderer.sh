rm /home/tda/CARLA/TrafficGeneration/Datasets/Waymo_tf_example/tests_prerendered/*

#python3 prerender.py \
python3 newrenderer.py \
    --data /home/tda/CARLA/TrafficGeneration/Datasets/Waymo_tf_example/tests_raw \
    --out /home/tda/CARLA/TrafficGeneration/Datasets/Waymo_tf_example/tests_prerendered \
    --use-vectorize \

python3 newrenderer.py \
    --data "/media/tda/Crucial X6/PabloComputer/TrafficGenerationDatasets/Waymo_tf_example_train" \
    --out /home/tda/CARLA/TrafficGeneration/Datasets/Waymo_tf_example/rendered_train_fixed
    
python3 newrenderer.py \
    --data "/media/tda/Crucial X6/PabloComputer/TrafficGenerationDatasets/Waymo_tf_example_valid" \
    --out /home/tda/CARLA/TrafficGeneration/Datasets/Waymo_tf_example/rendered_valid_fixed \

