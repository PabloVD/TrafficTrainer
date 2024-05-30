import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import glob
from natsort import natsorted
from moviepy.video.io import ImageSequenceClip
from matplotlib.patches import Circle
from scipy.spatial.distance import cdist

margin = 10
dpi = 100
# Frame rate
fps = 10
sizeagent = 100
min_agents = 5
fontsize = 20
linewidth = 5
alpha = 0.5

# Colors
col_gt = "blue"
col_pred1 = "red"
col_pred2 = "orange"
col_others = "green"

def generate_video(sce_id):

    # Get all frames in folder
    image_files = natsorted(glob.glob("frames/"+sce_id+"/*.png"))

    # Create video and save
    clip = ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile("videos/"+sce_id+".mp4", verbose=False, logger=None)


customlegend = []
customlegend.append( Circle((0,0), color=col_gt, label="Ground truth") )
customlegend.append( Circle((0,0), color=col_pred1, label="Prediction 1st") )
customlegend.append( Circle((0,0), color=col_pred2, label="Prediction 2nd") )
customlegend.append( Circle((0,0), color=col_others, label="Other agents") )

def find_similar_arrays(A, B, threshold=1e-3):
    similar_pairs = []
    #print(A.shape, B.shape)
    for i, a in enumerate(A):
        # Calculate the pairwise distances between array 'a' and all arrays in 'B'
        distances = cdist([a.reshape(-1)], B.reshape(B.shape[0], -1), metric='euclidean').flatten()
        # Find the indices of 'B' where the distance is below the threshold
        similar_indices = np.where(distances < threshold)[0]
        for j in similar_indices:
            similar_pairs.append((i, j))
    return similar_pairs

def find_same_agent(agents, agent, threshold=1e-3):

    distances = cdist([agent.reshape(-1)], agents.reshape(agents.shape[0], -1), metric='euclidean').flatten()
    similar_indices = np.where(distances < threshold)[0]
    
    return (len(similar_indices)>0)




def main():

    print("Generating videos")

    os.system("mkdir frames")
    os.system("mkdir videos")


    scenarios = glob.glob("scenarios/*")

    for scenario in tqdm(scenarios):

        sce_id = scenario.replace("scenarios/","")

        os.system("mkdir frames/"+sce_id)

        assets = natsorted(glob.glob(scenario+"/asset*"))

        trues = natsorted(glob.glob(scenario+"/agent_true*"))
        preds1 = natsorted(glob.glob(scenario+"/agent_pred*traj_0*"))
        preds2 = natsorted(glob.glob(scenario+"/agent_pred*traj_1*"))

        gt_all = np.load(scenario+"/gt_all.npy")

        agents_true = [np.load(file) for file in trues]
        agents_pred1 = [np.load(file) for file in preds1]
        agents_pred2 = [np.load(file) for file in preds2]

        # print(len(agents_true))
        # print(agents_true[0].shape)
        
        bboxarr = np.array([[ag[:,0].min(), ag[:,0].max(), ag[:,1].min(), ag[:,1].max()] for ag in agents_true])

        max_frames = min([a.shape[0] for a in agents_true])
        agents_true = np.array([a[:max_frames] for a in agents_true])

        gt_all = gt_all[:,:max_frames]

        # ground truth agents tot rack are also in the ground truth array of all agents, find those which are the same to not repeat them
        repeated_agents = find_similar_arrays(agents_true, gt_all)
        repeated_agents = np.array(repeated_agents)[:,1]

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
                    plt.plot(_X[:, 0], _X[:, 1], color="black", linewidth=1, alpha=0.2)

            for i in range(len(gt_all)):

                if i in repeated_agents:
                    continue

                true = gt_all[i]

                # if find_same_agent(agents_true[:,t], true[t]):
                #     continue


                plt.scatter(true[t,0],true[t,1],color=col_others,s=sizeagent,alpha=alpha)

            for i in range(len(agents_true)):

                true = agents_true[i]
                pred1 = agents_pred1[i]
                pred2 = agents_pred2[i]
                # plt.scatter(true[t,0],true[t,1],color="blue",s=sizeagent,alpha=0.7)
                # plt.scatter(pred[t,0],pred[t,1],color="red",s=sizeagent,alpha=0.7)
                plt.plot(true[:t,0],true[:t,1],color=col_gt,linewidth=linewidth,alpha=alpha)
                plt.plot(pred1[:t,0],pred1[:t,1],color=col_pred1,linewidth=linewidth,alpha=alpha)
                plt.plot(pred2[:t,0],pred2[:t,1],color=col_pred2,linewidth=linewidth*0.7,alpha=alpha)


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
