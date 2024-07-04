
import torch
from model import LightningModel
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=str, required=True, help="Checkpoint model name")
    parser.add_argument("--loss", type=str, required=True, help="Model name")
    parser.add_argument("--tl", type=str, required=True, help="Time limit")
    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    print(args)

    # Model Hyperparameters
    hparams = {}
    hparams["model_name"] = "resnet18"
    hparams["loss"] = args.loss
    hparams["in_channels"] = 25
    hparams["time_limit"] = args.tl
    hparams["n_traj"] = 6
    hparams["lr"] = 1.e-3
    hparams["sched"] = "multistep"
    hparams["weight_decay"] = 0.

    checkpoint = "models/"+args.m+".ckpt"

    print("Loading from checkpoint",checkpoint)
    model = LightningModel.load_from_checkpoint(checkpoint_path=checkpoint, hparams=hparams, map_location='cuda:0').cuda().eval()
  
    script = model.to_torchscript()

    # save for use in production environment
    torch.jit.save(script, "model.pt")


if __name__ == "__main__":
    main()
