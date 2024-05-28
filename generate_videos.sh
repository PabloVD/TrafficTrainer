rm -r scenarios/*

python3 store_test_data.py \
	--model resnet18 \
	--data /home/tda/CARLA/TrafficGeneration/Datasets/Waymo_tf_example/tests_prerendered	\
	--use-top1 \
	--save ./viz \
	--n-samples 2

rm -r frames/*
rm -r videos/*

python3 generate_videos.py
