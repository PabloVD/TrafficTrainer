python3 prerender.py \
    --data "/media/tda/Crucial X6/PabloComputer/TrafficGenerationDatasets/Waymo_tf_example_train" \
    --out /home/tda/CARLA/TrafficGeneration/Datasets/Waymo_tf_example/rendered_train_moving
    
python3 prerender.py \
    --data "/media/tda/Crucial X6/PabloComputer/TrafficGenerationDatasets/Waymo_tf_example_valid" \
    --out /home/tda/CARLA/TrafficGeneration/Datasets/Waymo_tf_example/rendered_valid_moving \
    --use-vectorize \
