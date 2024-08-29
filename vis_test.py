
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

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

inchannels = 10 #11
# inchannels = 30

IMG_RES = 224
center_ego = [IMG_RES//4, IMG_RES//2]
noise_pos_std = 2.
noise_ang_std = 10.
noise_ang_std2 = 20.
ego_rotator = transforms.RandomRotation(noise_ang_std2, center=(center_ego[1],center_ego[0]))
others_rotator = transforms.RandomRotation(noise_ang_std2)

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

    # print(img.shape, locs.shape)
    # print(locs[0,-1])

    # inds = (img>0.).nonzero()
    # print(inds.shape)

    # exit()

    return locs

# def ego_transform(x):

#     ego = x[:,3:3+inchannels]

#     timeframes = ego.shape[1]

#     # noise_pos = noise_pos_std*torch.randn((timeframes,2))
#     # noise_pos = torch.cumsum(noise_pos, dim=1)
#     # noise_pos = torch.flip(noise_pos,dims=(0,))

#     # noise_ang = noise_ang_std*torch.randn((timeframes))
#     # noise_ang = torch.cumsum(noise_ang, dim=0)
#     # noise_ang = torch.flip(noise_ang,dims=(0,))

#     # Random rotation around end position of the ego vehicle
#     #ego = ego_rotator(ego)

#     locs = ego_loc(ego)

#     noise_trans = noise_pos_std*torch.randn((2))
#     noise_angle = noise_ang_std*torch.randn((1))

#     for i in range(timeframes):
#         for b in range(ego.shape[0]):
            
#             #translation = [noise_pos[i,0],noise_pos[i,1]]
#             translation = [noise_trans[0],noise_trans[1]]
#             center_rot = (locs[b,i,1], locs[b,i,0])
#             #angle = noise_ang[i].item()
#             angle = noise_angle.item()

#             # Random translation and rotation around vehicle
#             ego[b:b+1,i] = transforms.functional.affine(ego[b:b+1,i], translate=translation, angle=angle, scale=1, shear=0, center=center_rot)
        
#     x[:,3:3+inchannels] += ego

#     return x

def ego_transform(x):

    ego = x[:,3:3+inchannels]

    n_timeframes = ego.shape[1]
    center_rot = [ center_ego[1], center_ego[0] ]

    # # option 1
    # K = 10.

    # angle = 0.
    # angle_incr = np.random.uniform(-3,3)
    # for i in reversed(range(n_timeframes-1)):

    #     # option 1
    #     angle += angle_incr

    #     translation = [ -K*np.sin(angle*np.pi/180.) , -K*(1.-np.cos(angle*np.pi/180.)) ]
    #     ego[:,i] = transforms.functional.affine(ego[:,i], translate=translation, angle=angle, scale=1, shear=0, center=center_rot)
    # # end option 1

    # option 2

    locs = ego_loc(ego)
    vel = locs[:,1:]-locs[:,:-1]
    vel = torch.sqrt(vel[:,:,0]**2. + vel[:,:,1]**2.)
    end_locs = locs[:,:-1]
    vel[torch.logical_and(end_locs[:,:,0]==0. , end_locs[:,:,1]==0.)]=0.

    vel = torch.clamp(vel, 0, 3)

    # angle_incr = 0.3#random.choice([-0.2,0.2])
    angle_incr = np.random.uniform(-0.3,0.3)
    angle_incr = random.choice([-1,1])

    for b in range(ego.shape[0]):
        angle = 0.
        # translation = [0.,0.]

        for i in reversed(range(n_timeframes-1)):

            angle += vel[b,i].item()*angle_incr
            # angle += np.clip(vel[b,i].item()*angle_incr, a_min=-0.6, a_max=0.6)

            # if vel[b,i].item()>0:
            #     angle += angle_incr
            # else:
            #     angle 

            translation = [ -vel[b,i]*np.sin(angle*np.pi/180.) ,  -vel[b,i]*np.cos(angle*np.pi/180.) ]
            # translation = [ translation[0] + -vel[b,i]*np.sin(angle*np.pi/180.) , translation[1] + -vel[b,i]*np.cos(angle*np.pi/180.) ]

            # ego[b:b+1,i] = transforms.functional.affine(ego[b:b+1,-1], translate=translation, angle=angle, scale=1, shear=0, center=center_rot)
            ego[b:b+1,i] = transforms.functional.affine(ego[b:b+1,i+1], translate=translation, angle=angle, scale=1, shear=0, center=center_rot)
            # ego[b:b+1,i] = transforms.functional.affine(ego[b:b+1,i+1], translate=translation, angle=vel[b,i].item()*angle_incr, scale=1, shear=0, center=center_rot)


    # end option 2
        
    x[:,3:3+inchannels] = ego

    return x

def others_transform(x):

    others = x[:,3+inchannels:3+inchannels*2]
    x[:,3+inchannels:3+inchannels*2] = others_rotator(others)
    return x

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
outpath += "frames"

if not os.path.exists(outpath):
    os.system("mkdir "+outpath)
else:
    os.system("rm "+outpath+"/*")

dataloader = DataLoader(
        WaymoLoader("/home/tda/CARLA/TrafficGeneration/Datasets/Waymo_tf_example/tests_prerendered/"),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
    )




print("Rendering frames")
# for j, filename in enumerate(tqdm(filenames)):
for j, data in enumerate(tqdm(dataloader)):

    if j>20:
        continue

    # if j!=5:
    #     continue

    # data = np.load(filename, allow_pickle=True)
    # raster = data["raster"].astype("float32") / 255

    raster = data[0]

    #print(raster.shape)

    # if fixed_frame:
    #     raster = raster.transpose(2, 1, 0)
    #     #raster = raster.transpose(1, 0, 2)

    # raster = ego_transform(raster)
    #raster = others_transform(raster)

    #print(raster.shape)
    
    raster = raster.squeeze(0).numpy()
    raster = raster.transpose(1, 2, 0)

    for i in range(inchannels):
        # for i in [10,0]:

        if fixed_frame:
            rast = raster2rgb(raster, i)
        else:
            rast = raster[i]

        rast[rast>1]=1

        plt.imshow(rast)
        # plt.axis('off')

        plt.savefig(outpath+"/vehicle_"+str(j)+"_"+str(i)+".png", bbox_inches='tight')
