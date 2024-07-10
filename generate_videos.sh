rm -r scenarios/*

python3 test_model.py \
	-m model_nll_80 \
	--data /home/tda/CARLA/TrafficGeneration/Datasets/Waymo_tf_example/tests_prerendered	\
	--use-top1 \
	--save ./viz \
	--n-samples 20

rm -r frames/*
rm -r videos/*

python3 generate_videos.py
