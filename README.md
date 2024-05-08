# TrafficTrainer

Predict trajectories of vehicles in the Waymo Open Dataset Motion based on rasterized inputs.

Based on [MotionCNN](https://arxiv.org/abs/2206.02163), see [this repo](https://github.com/kbrodt/waymo-motion-prediction-2021) and [this repo](https://github.com/stepankonev/MotionCNN-Waymo-Open-Motion-Dataset).

To install the required packages, just run:

```
pip install -r requirements.txt
```

To train a model, run:

```
sh train.sh
```