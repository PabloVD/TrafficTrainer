
import torch
from model import LightningModel
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=str, required=True, help="Model name")
    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    print(args)

    # Model params
    IN_CHANNELS = 25
    TL = 20
    N_TRAJS = 6
    model_name = "resnet18"
    sched="multistep"

    checkpoint = "models/"+args.m+".ckpt"

    print("Loading from checkpoint",checkpoint)
    model = LightningModel.load_from_checkpoint(checkpoint_path=checkpoint, model_name=model_name, in_channels=IN_CHANNELS, time_limit=TL, n_traj=N_TRAJS, lr=1.e-3, weight_decay=0., sched=sched,map_location='cuda:0').cuda().eval()
  
    script = model.to_torchscript()

    # save for use in production environment
    torch.jit.save(script, "model.pt")


if __name__ == "__main__":
    main()
