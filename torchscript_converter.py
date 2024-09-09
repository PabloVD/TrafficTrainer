
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

def export_model(modelname):

    checkpoint_folder = "checkpoints/"+modelname

    with open(checkpoint_folder+"/hparams.yaml") as yamlfile:
        hparams = yaml.safe_load(yamlfile)

    checkpoints = sorted(glob.glob(checkpoint_folder+"/checkpoints/*ckpt"))
    checkpoint = checkpoints[-1]    # Get the last checkpoint among the stored ones

    print("Loading from checkpoint",checkpoint)
    model = LightningModel.load_from_checkpoint(checkpoint_path=checkpoint, hparams=hparams, map_location='cuda:0').cuda().eval()
  
    script = model.to_torchscript()

    torch.jit.save(script, "models/"+modelname+".pt")

    model_loaded = torch.jit.load("models/"+modelname+".pt")
    # print(model_loaded)

def main():

    args = parse_args()

    export_model(args.m)


if __name__ == "__main__":
    main()
