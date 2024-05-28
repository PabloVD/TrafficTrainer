import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import glob
from natsort import natsorted
from moviepy.video.io import ImageSequenceClip
from matplotlib.patches import Circle

margin = 10
dpi = 100
# Frame rate
fps = 10
sizeagent = 200
min_agents = 5
fontsize = 20

def generate_video(sce_id):

    # Get all frames in folder
    image_files = natsorted(glob.glob("frames/"+sce_id+"/*.png"))

    # Create video and save
    clip = ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile("videos/"+sce_id+".mp4", verbose=False, logger=None)


customlegend = []
customlegend.append( Circle((0,0), color="blue", label="Ground truth") )
customlegend.append( Circle((0,0), color="red", label="Prediction") )
customlegend.append( Circle((0,0), color="green", label="Other agents") )


def main():

    print("Generating videos")

    os.system("mkdir frames")
    os.system("mkdir videos")


    scenarios = glob.glob("scenarios/*")

    for scenario in tqdm(scenarios):

        sce_id = scenario.replace("scenarios/","")

        os.system("mkdir frames/"+sce_id)

        assets = natsorted(glob.glob(scenario+"/asset*"))

        preds = natsorted(glob.glob(scenario+"/agent_pred*"))
        trues = natsorted(glob.glob(scenario+"/agent_true*"))

        gt_all = np.load(scenario+"/gt_all.npy")

        agents_true = [np.load(file) for file in trues]
        agents_pred = [np.load(file) for file in preds]

        
        bboxarr = np.array([[ag[:,0].min(), ag[:,0].max(), ag[:,1].min(), ag[:,1].max()] for ag in agents_pred])

        max_frames = min([a.shape[0] for a in agents_true])

        #print("Num agents:",len(gt_all), len(agents_true))
        if len(agents_true)<3:
            #print("Few agents, continue")
            continue
        
        for t in range(max_frames):
      
            plt.figure(figsize=(15, 15))#, dpi=dpi)
            
            for asset in assets:

                _X = np.load(asset)
                
                if _X[:, 5:12].sum() > 0:
                    # plt.plot(_X[:, 0], _X[:, 1], linewidth=20, color="green")
                    pass
                else:
                    plt.plot(_X[:, 0], _X[:, 1], color="black", alpha=0.2)

            for i in range(len(gt_all)):

                true = gt_all[i]
                plt.scatter(true[t,0],true[t,1],color="green",s=sizeagent,alpha=0.5)

            for true in agents_true:
                for true2 in gt_all:
                    if true.shape[0]==true2.shape[0]:
                        if (true==true2).sum()>0:
                            
                            print("ey")

            for i in range(len(agents_true)):

                true = agents_true[i]
                pred = agents_pred[i]
                # plt.scatter(true[t,0],true[t,1],color="blue",s=sizeagent,alpha=0.7)
                # plt.scatter(pred[t,0],pred[t,1],color="red",s=sizeagent,alpha=0.7)
                plt.plot(true[:t,0],true[:t,1],color="blue",linewidth=2,alpha=0.7)
                plt.plot(pred[:t,0],pred[:t,1],color="red",linewidth=2,alpha=0.7)


            plt.xlim(bboxarr[:,0].min()-margin, bboxarr[:,1].max()+margin)
            plt.ylim(bboxarr[:,2].min()-margin, bboxarr[:,3].max()+margin)
            plt.xticks([])
            plt.yticks([])
            plt.legend(handles=customlegend, loc = "upper right", fontsize=fontsize)

            plt.savefig("frames/"+sce_id+"/"+str(t)+".png",bbox_inches='tight')
            plt.close()

        generate_video(sce_id)

            
            


if __name__ == "__main__":
    main()
