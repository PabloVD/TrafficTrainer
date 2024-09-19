
import numpy as np
from matplotlib import pyplot as plt
import glob
from tqdm import tqdm
import os
import torch
import random
from natsort import natsorted

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

inchannels = 10 #11




# raster expected in shape (imgsize,imgsize,channels)
def raster2rgb(raster, i):

    road = raster[:,:,0:3]
    img = np.copy(road)
    ego = raster[:,:,3+i]
    others = raster[:,:,3+inchannels+i]

    ego = ego[:,:,None]
    others = others[:,:,None]

    zeros = np.zeros_like(ego)
    # ego = zeros
    ego_pos = np.concatenate([ego,ego,ego],axis=-1)
    ego = np.concatenate([zeros,zeros,ego],axis=-1)/2.
    others_pos = np.concatenate([others,others,others],axis=-1)
    others = np.concatenate([zeros,others,zeros],axis=-1)/2.

    img[ego_pos!=0]=ego[ego_pos!=0]
    img[others_pos!=0]=others[others_pos!=0]

    return img



outpath = "/home/tda/CARLA/TrafficGeneration/vis_tests/"
outpath += "renderouts"

if not os.path.exists(outpath):
    os.system("mkdir "+outpath)
else:
    os.system("rm "+outpath+"/*")

def plot_frames(mode):

    filenames = natsorted(glob.glob("/home/tda/CARLA/TrafficGeneration/TrafficTrainer/rendertests/batch_"+mode+"*_time*"))
    # filenames = filenames[:10]

    
    print(filenames)


    print("Rendering frames")
    # for j, filename in enumerate(tqdm(filenames)):
    for j, filename in enumerate(tqdm(filenames)):

        data = np.load(filename, allow_pickle=True)

        raster = data#[0]
        raster = raster.transpose(1, 2, 0)

        # print(raster.shape)


        for i in range(inchannels):


            rast = raster2rgb(raster, i)
            

            rast[rast>1]=1

            plt.imshow(rast)
            # plt.axis('off')

            plt.savefig(outpath+"/"+mode+"_currtime_"+str(j)+"_"+str(i)+".png", bbox_inches='tight')



plot_frames("torch")
plot_frames("debug")