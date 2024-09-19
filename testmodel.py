
import numpy as np
from matplotlib import pyplot as plt
import glob
from tqdm import tqdm
import os
import torch
from torch.utils.data import DataLoader
from data_utils.waymo_dataset import WaymoLoader
import random

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

raster_size = 224
displacement = torch.tensor([[raster_size // 4, raster_size // 2]])
zoom_fact = 3.
inchannels = 10

device="cpu"

modelname = "models/NLL_20_vit_base_patch16_224_predyaw_5.pt"

# Load model
print("Using model",modelname)
model = torch.jit.load(modelname)
model = model.to(device)


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


filenames = sorted(glob.glob("/home/tda/CARLA/TrafficGeneration/Datasets/Waymo_tf_example/tests_prerendered/vehicle_*"))
#filenames = filenames[:10]

outpath = "/home/tda/CARLA/TrafficGeneration/vis_tests/"
outpath += "compareframes"

if not os.path.exists(outpath):
    os.system("mkdir "+outpath)
else:
    os.system("rm "+outpath+"/*")

dataloader = DataLoader(
        WaymoLoader("/home/tda/CARLA/TrafficGeneration/Datasets/Waymo_tf_example/rendered_train_fixed/"),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
    )



def show_gt(data, ind):

    raster = data["raster"]
    is_available = data["future_val_marginal"]

    # np.save("rendertests/batch_torch"+str(ind)+"_time_"+str(0),raster[0].cpu().detach().numpy())

    confidences, logits = model(raster)
    confidences, logits = confidences.squeeze(0), logits.squeeze(0)
        
    indmax_batch = confidences.argmax(0)
    pred = logits[indmax_batch]
    pred = pred[:,:2]*zoom_fact + displacement
    pred = pred.cpu().detach().numpy()

    raster = raster.squeeze(0).numpy()
    raster = raster.transpose(2, 1, 0)

    img = raster2rgb(raster, inchannels-1)
    plt.imshow(img)

    y = data["gt_marginal"]
    y = y*is_available.view(-1,20,1)
    y = y*zoom_fact + displacement

    y = y.squeeze(0)
    plt.scatter(y[:,0],y[:,1],color="b",s=10,alpha=0.5)
    plt.scatter(pred[:,0],pred[:,1],color="r",s=10,alpha=0.5)
    plt.savefig(outpath+"/frame_"+str(ind)+".png")
    plt.close()
    # plt.show()

for j, data in enumerate(tqdm(dataloader)):

    show_gt(data, j)
    if j==50:
        break