rm /home/tda/CARLA/TrafficGeneration/Datasets/Waymo_tf_example/tests_prerendered/*


python3 data_utils/closedlooprenderer.py \
    --data /home/tda/CARLA/TrafficGeneration/Datasets/Waymo_tf_example/tests_raw \
    --out /home/tda/CARLA/TrafficGeneration/Datasets/Waymo_tf_example/tests_prerendered \
    #--use-vectorize \

#python3 data_utils/closedlooprenderer.py \
#    --data "/media/tda/Crucial X6/PabloComputer/TrafficGenerationDatasets/Waymo_tf_example_train" \
#    --out "/media/tda/Crucial X6/PabloComputer/TrafficGenerationDatasets/WaymoPrerendered/rendered_train_fixed"
    
#python3 data_utils/closedlooprenderer.py \
#    --data "/media/tda/Crucial X6/PabloComputer/TrafficGenerationDatasets/Waymo_tf_example_valid" \
#    --out "/media/tda/Crucial X6/PabloComputer/TrafficGenerationDatasets/WaymoPrerendered/rendered_valid_fixed" \
#    --use-vectorize

