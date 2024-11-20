
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
import torch
from torch.utils.data import DataLoader
from data_utils.waymo_dataset import WaymoLoader
import random

device="cuda"

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

raster_size = 224
displacement = torch.tensor([[raster_size // 4, raster_size // 2]]).to(device)
zoom_fact = 3.
inchannels = 10

samplfrec = 1
showpred = True

datafolder = "/home/tda/CARLA/TrafficGeneration/Datasets/Waymo_tf_example/rendered_valid_fixed"

# Compute average and final displacement error
# Expects arrays of shape (T,2)
def displacement_error(x_true, x_pred):

    # x_true = x_true[:time_limit]
    # x_pred = x_pred[:time_limit]

    ade = torch.sqrt(torch.sum((x_pred[::samplfrec] - x_true[::samplfrec])**2.,axis=-1)).mean()
    fde = torch.sqrt(torch.sum((x_pred[-1] - x_true[-1])**2.,axis=-1))

    return ade, fde

# raster expected in shape (imgsize,imgsize,channels)
def raster2rgb(raster, i):

    road = raster[:,:,0:3]
    img = np.copy(road)
    ego = raster[:,:,3+i]
    others = raster[:,:,3+inchannels+i]

    ego = ego[:,:,None]
    others = others[:,:,None]

    zeros = np.zeros_like(ego)
    ego_pos = np.concatenate([ego,ego,ego],axis=-1)
    ego = np.concatenate([zeros,zeros,ego],axis=-1)/2.
    others_pos = np.concatenate([others,others,others],axis=-1)
    others = np.concatenate([zeros,others,zeros],axis=-1)/2.

    img[ego_pos!=0]=ego[ego_pos!=0]
    img[others_pos!=0]=others[others_pos!=0]

    return img

def render_predictions(y, pred, raster, ind):

    y, pred = y.cpu().detach().numpy(), pred.cpu().detach().numpy()

    raster = raster.squeeze(0).cpu().detach().numpy()
    raster = raster.transpose(2, 1, 0)

    img = raster2rgb(raster, inchannels-1)
    plt.imshow(img)

    plt.scatter(y[:,0],y[:,1],color="b",s=10,alpha=0.5)
    plt.scatter(pred[:,0],pred[:,1],color="r",s=10,alpha=0.5)
    plt.savefig(outpath+"/frame_"+str(ind)+".png")
    plt.close()
    # plt.show()


def run_model(model, data, ind, showpred=True):

    # Get input data
    raster = data["raster"].to(device)
    is_available = data["future_val_marginal"].to(device)
    y = data["gt_marginal"].to(device)

    # Run model
    confidences, logits = model(raster)

    # Process prediction
    confidences, logits = confidences.squeeze(0), logits.squeeze(0)
    indmax_batch = confidences.argmax(0)
    pred = logits[indmax_batch]
    pred = pred[:,:2]*zoom_fact + displacement

    # Process ground truth
    y = y*is_available.view(-1,20,1)
    y = y*zoom_fact + displacement
    y = y.squeeze(0)

    # Compute metrics
    ade, fde = displacement_error(y, pred)

    # Render predictions
    if showpred:
        render_predictions(y, pred, raster, ind)
        
    return ade, fde

def test_model(modelname, numsamples):

    # Load model
    print("Using model",modelname)
    model = torch.jit.load(modelname)
    model = model.to(device)

    ade_list, fde_list = [], []

    for j, data in enumerate(tqdm(dataloader)):

        ade, fde = run_model(model, data, j, showpred=showpred)
        ade_list.append(ade.item()); fde_list.append(fde.item())

        if j==numsamples:
            break

    ade_list, fde_list = np.array(ade_list), np.array(fde_list)
    ade_mean, ade_std, fde_mean, fde_std = ade_list.mean(),ade_list.std(), fde_list.mean(), fde_list.std()
    print("ADE={:.2f}+-{:.2f}, FDE={:.2f}+-{:.2f}".format(ade_mean, ade_std, fde_mean, fde_std))


if __name__=="__main__":

    if showpred:

        outpath = "/home/tda/CARLA/TrafficGeneration/vis_tests/"
        outpath += "compareframes"

        if not os.path.exists(outpath):
            os.system("mkdir "+outpath)
        else:
            os.system("rm "+outpath+"/*")

    dataloader = DataLoader(
            WaymoLoader(datafolder),
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            persistent_workers=True,
        )

    models = [
        # "models/NLL_20_vit_large_patch32_224_closedloop_0.pt",
        # "models/NLL_20_vit_base_patch16_224_predyaw_5.pt",
        "models/NLL_20_vit_large_patch32_224_cumsum_2.pt"
            ]

    for modelname in models:

        test_model(modelname, numsamples=300)