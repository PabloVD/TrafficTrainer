
import numpy as np
from matplotlib import pyplot as plt
import glob
from tqdm import tqdm
import os
import torch
import torchvision.transforms as transforms

IMG_RES = 224
center_ego = [IMG_RES//4, IMG_RES//2]
noise_pos_std = 2.
noise_ang_std = 10.
noise_ang_std2 = 20.
ego_rotator = transforms.RandomRotation(noise_ang_std2, center=(center_ego[1],center_ego[0]))

transf = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(size=(IMG_RES, IMG_RES),scale=(0.95,1.)),
        ])

def ego_loc(img):

    tmp = img.reshape(img.shape[0],img.shape[1],-1)
    indices = torch.argmax(tmp,dim=-1)
    row = indices // IMG_RES
    column = indices - IMG_RES*row

    locs = torch.cat([row.unsqueeze(-1),column.unsqueeze(-1)],dim=-1)

    return locs

def ego_transform(x):

    ego = x[:,3:3+11]

    timeframes = ego.shape[1]

    noise_pos = noise_pos_std*torch.randn((timeframes,2))
    noise_pos = torch.cumsum(noise_pos, dim=1)
    noise_pos = torch.flip(noise_pos,dims=(0,))

    noise_ang = noise_ang_std*torch.randn((timeframes))
    noise_ang = torch.cumsum(noise_ang, dim=0)
    noise_ang = torch.flip(noise_ang,dims=(0,))

    # Random rotation around end position of the ego vehicle
    ego = ego_rotator(ego)

    locs = ego_loc(ego)

    for i in range(timeframes):
        for b in range(ego.shape[0]):
            
            translation = [noise_pos[i,0],noise_pos[i,1]]
            center_rot = (locs[b,i,1], locs[b,i,0])
            angle = noise_ang[i].item()

            # Random translation and rotation around vehicle
            ego[b:b+1,i] = transforms.functional.affine(ego[b:b+1,i], translate=translation, angle=angle, scale=1, shear=0, center=center_rot)
        
    x[:,3:3+11] = ego

    return x



fixed_frame = True
#fixed_frame = False

filenames = sorted(glob.glob("/home/tda/CARLA/TrafficGeneration/Datasets/Waymo_tf_example/tests_prerendered/vehicle_*"))
#filenames = sorted(glob.glob("/home/tda/CARLA/TrafficGeneration/Datasets/Waymo_tf_example/rendered_valid_fixed/vehicle_*"))
filenames = filenames[:10]

outpath = "/home/tda/CARLA/TrafficGeneration/vis_tests/"
outpath += "frames"

if not os.path.exists(outpath):
    os.system("mkdir "+outpath)
else:
    os.system("rm "+outpath+"*")

print("Rendering frames")
for j, filename in enumerate(tqdm(filenames)):

    if j>2:
        continue

    data = np.load(filename, allow_pickle=True)

    raster = data["raster"].astype("float32") / 255

    if fixed_frame:
        raster = raster.transpose(2, 1, 0)

    raster = ego_transform(torch.Tensor(raster).unsqueeze(0))
    #raster = transf(raster)
    raster = raster.squeeze(0).numpy()

    for i in range(11):

        if fixed_frame:
            rast = raster[0:3].transpose(1, 2, 0).mean(-1)
            rast += raster[3+i]/2. + raster[3+11+i]  
        else:
            rast = raster[i]

        plt.imshow(rast)

        plt.savefig(outpath+"/vehicle_"+str(j)+"_"+str(i)+".png")


    # for i in range(11):
    # #for i in range(1):
    #     #rast = raster[0] +raster[1] + raster[2] + 2*raster[3+i] + raster[3+11+i]
    #     #rast = raster[i*3] + raster[i*3+1] +raster[i*3+2] + 2*raster[33+i] + raster[33+11+i]
    #     #rast = data["raster"][:,:,0:3]
    #     rast = raster[i]
    #     plt.imshow(rast)
    #     plt.savefig(outpath+"frames/vehicle_"+str(j)+"_"+str(i)+".png")


