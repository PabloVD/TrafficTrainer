
import numpy as np
from matplotlib import pyplot as plt
import glob
from tqdm import tqdm
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data_utils.waymo_dataset import WaymoLoader
import random
from data_utils.rasterizer_torch import get_rotation_matrix, zoom_fact, raster_size

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

displacement = torch.tensor([[raster_size // 4, raster_size // 2]])

inchannels = 10 #11

device="cpu"

modelname = "models/NLL_20_vit_base_patch16_224_16.pt"

# Load model
print("Using model",modelname)
model = torch.jit.load(modelname)
model = model.to(device)


# def update_step(self, XY, YAW, confidences, logits, agind, currind):

#     # Extract batch size
#     batch_size = XY.shape[0]

#     arr = torch.arange(batch_size)
    
#     # Gather the corresponding xy and yaw from each ego agent
#     xy_ag = XY[arr, agind]
#     yaw_ag = YAW[arr, agind]
    
#     # Get the index of the maximum confidence for each row in the batch
#     indmax_batch = confidences.argmax(dim=1)
    
#     # Gather the logits based on the indices obtained from the maximum confidences
#     pred = logits[arr, indmax_batch]
    
#     # Get current position and yaw
#     currpos = xy_ag[:, currind]
#     curryaw = yaw_ag[:, currind]
    
#     # Calculate rotation matrix for each batch
#     rot_matrix = get_rotation_matrix(-curryaw)
    
#     # Rotating and translating prediction
#     pred_rotated = torch.bmm(pred, rot_matrix) + currpos.unsqueeze(1)  # shape: (batch_size, 10, 2)
    
#     # Displace directly the vehicle to the predicted position
#     nextpos = pred_rotated[:, 0]  # Take the first position from the prediction
    
#     # Estimate orientation
#     diffpos = nextpos - currpos
#     newyaw = torch.atan2(diffpos[:, 1], diffpos[:, 0])
    
#     # Update XY and YAW for the next step
#     XY[arr, agind, currind + 1] = nextpos
#     YAW[arr, agind, currind + 1] = newyaw

#     return XY, YAW



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

fixed_frame = True
#fixed_frame = False

filenames = sorted(glob.glob("/home/tda/CARLA/TrafficGeneration/Datasets/Waymo_tf_example/tests_prerendered/vehicle_*"))
#filenames = sorted(glob.glob("/home/tda/CARLA/TrafficGeneration/Datasets/Waymo_tf_example/rendered_valid_fixed/vehicle_*"))
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




print("Rendering frames")
# for j, filename in enumerate(tqdm(filenames)):
# for j, data in enumerate(tqdm(dataloader)):

# data = next(iter(dataloader))

# y = data["gt_marginal"]
# is_available = data["future_val_marginal"]

# # y = y[is_available]
# XY = data["XY"]
# YAWS = data["YAWS"]
# agind = data["agent_ind"]
# batchsize = y.shape[0]
# btchrng = torch.arange(batchsize)
# yaw = YAWS[btchrng, agind]
# yaw = yaw[:,10:-1]
# is_available = is_available[:,:-1]



# diffpos = y[:,1:]-y[:,:-1]

# newyaw = torch.atan2(diffpos[:, :, 1], diffpos[:, :, 0])

# print(y.shape, is_available.shape, yaw.shape, newyaw.shape)

# # yaw = yaw[is_available]

# # newyaw = newyaw[is_available]

# print(yaw.shape, newyaw.shape)

# for b in range(batchsize):
#     y_true = yaw[b]
#     y_pred = newyaw[b]
#     # y_true = y_true[is_available[b]]
#     # y_pred = y_pred[is_available[b]]
#     print("True",y_true)
#     print("Pred",y_pred+np.pi)
#     #plt.plot(y_true,y_pred,color="b")
    
#     plt.plot(torch.cos(y_true),color="b")
#     plt.plot(torch.cos(y_pred),color="r")

# plt.grid()
# plt.show()

def show_gt(data, ind):

    raster = data["raster"]
    is_available = data["future_val_marginal"]

    confidences, logits = model(raster)
    confidences, logits = confidences.squeeze(0), logits.squeeze(0)
        
    indmax_batch = confidences.argmax(0)
    pred = logits[indmax_batch]
    pred = pred*zoom_fact + displacement
    pred = pred.cpu().detach().numpy()

    raster = raster.squeeze(0).numpy()
    raster = raster.transpose(2, 1, 0)

    img = raster2rgb(raster, inchannels-1)
    plt.imshow(img)

    y = data["gt_marginal"]#.squeeze(0)
    # posego = XY[btchrng, agind]
    # currpos = posego[:,inchannels-1]
    # yawego = YAWS[btchrng, agind]
    # yaw_curr = yawego[:,inchannels-1]
    # rot_matrix = get_rotation_matrix(-yaw_curr)
    y = y*is_available.view(-1,20,1)

    # print(y.shape, is_available.shape)

    # y = torch.bmm(y, rot_matrix)*zoom_fact + displacement # shape: (batch_size, 10, 2)
    y = y*zoom_fact + displacement

    y = y.squeeze(0)
    plt.scatter(y[:,0],y[:,1],color="b",s=10,alpha=0.5)
    plt.scatter(pred[:,0],pred[:,1],color="r",s=10,alpha=0.5)
    plt.savefig(outpath+"/frame_"+str(ind)+".png")
    plt.close()
    # plt.show()

for j, data in enumerate(tqdm(dataloader)):

    show_gt(data, j)
    if j==30:
        break