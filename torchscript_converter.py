
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

    pos, yaw = torch.rand((10,2)), torch.rand((10))
    confs, logits = torch.rand((10,6)), torch.rand((10,6,20,3))
  
    script = model.to_torchscript()

    #print(script.next_step(pos, yaw, confs, logits))
    # print(script.lr)

    torch.jit.save(script, "models/"+modelname+".pt")
    print("Model correctly exported to: "+"models/"+modelname+".pt")

    model_loaded = torch.jit.load("models/"+modelname+".pt")

    # Test that some methods are correctly exported
    # print(model_loaded)
    # print(model_loaded.lr)
    print(model_loaded.next_step(pos, yaw, confs, logits))
    # print(model.rasterizer)

def main():

    args = parse_args()

    export_model(args.m)


if __name__ == "__main__":
    main()
