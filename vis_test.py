
import numpy as np
from matplotlib import pyplot as plt
import glob
from tqdm import tqdm
import os

#fixed_frame = True
fixed_frame = False
#use_rgb = False
use_rgb = True

filenames = sorted(glob.glob("/home/tda/CARLA/TrafficGeneration/Datasets/Waymo_tf_example/tests_prerendered/vehicle_*"))
filenames = filenames[:100]
outpath = "/home/tda/CARLA/TrafficGeneration/vis_tests/"

os.system("rm "+outpath+"frames/*")

print("Rendering frames")
for j, filename in enumerate(tqdm(filenames)):

    if j>2:
        continue

    data = np.load(filename, allow_pickle=True)

    raster = data["raster"].astype("float32") / 255

    if not use_rgb:
        raster = raster.transpose(2, 1, 0)

    for i in range(11):

        if fixed_frame:
            rast = raster[0:3].transpose(1, 2, 0).mean(-1)
            rast += raster[3+i]/2. + raster[3+11+i]  
        else:
            rast = raster[i]

        plt.imshow(rast)

        plt.savefig(outpath+"frames/vehicle_"+str(j)+"_"+str(i)+".png")


    # for i in range(11):
    # #for i in range(1):
    #     #rast = raster[0] +raster[1] + raster[2] + 2*raster[3+i] + raster[3+11+i]
    #     #rast = raster[i*3] + raster[i*3+1] +raster[i*3+2] + 2*raster[33+i] + raster[33+11+i]
    #     #rast = data["raster"][:,:,0:3]
    #     rast = raster[i]
    #     plt.imshow(rast)
    #     plt.savefig(outpath+"frames/vehicle_"+str(j)+"_"+str(i)+".png")


