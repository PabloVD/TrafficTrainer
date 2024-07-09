
import torch
from model import LightningModel
import argparse
import yaml
import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=str, required=True, help="Folder model name")
    args = parser.parse_args()
    return args

def main():

    args = parse_args()

    checkpoint_folder = "models/"+args.m

    with open(checkpoint_folder+"/hparams.yaml") as yamlfile:
        hparams = yaml.safe_load(yamlfile)

    checkpoints = sorted(glob.glob(checkpoint_folder+"/checkpoints/*ckpt"))
    checkpoint = checkpoints[-1]

    print("Loading from checkpoint",checkpoint)
    model = LightningModel.load_from_checkpoint(checkpoint_path=checkpoint, hparams=hparams, map_location='cuda:0').cuda().eval()
  
    script = model.to_torchscript()

    torch.jit.save(script, "model_"+hparams["loss"]+"_"+str(hparams["time_limit"])+".pt")


if __name__ == "__main__":
    main()
