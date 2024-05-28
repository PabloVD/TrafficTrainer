import argparse
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from torch.utils.data import DataLoader

from waymo_dataset import WaymoLoader
from tqdm import tqdm
from model import LightningModel

IMG_RES = 224
IN_CHANNELS = 25
TL = 80
N_TRAJS = 6

margin = 50
dpi = 100

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--save", type=str, required=True)
    parser.add_argument("--n-samples", type=int, required=False, default=50)
    parser.add_argument("--use-top1", action="store_true")

    args = parser.parse_args()

    return args

def init_figure(idx):

    figure(figsize=(15, 15), dpi=dpi)
    patch_id = idx
    bboxarr = []

    return patch_id, bboxarr


def main():
    args = parse_args()
    if not os.path.exists(args.save):
        os.mkdir(args.save)

    model_name = args.model
    checkpoint = model_name+"-bestmodel.ckpt"
    print("Loading from checkpoint",checkpoint)
    model = LightningModel.load_from_checkpoint(checkpoint_path=checkpoint, model_name=model_name, in_channels=IN_CHANNELS, time_limit=TL, n_traj=N_TRAJS, lr=1.e-3).cuda().eval()
    
    loader = DataLoader(
        WaymoLoader(args.data, return_vector=True),
        batch_size=1,
        num_workers=1,
        shuffle=False,
    )

    scenario_id = None

    os.system("mkdir scenarios")

    iii = 0
    num_sce = 0


    with torch.no_grad():
        for x, y, is_available, vector_data, center, shift, yaw, scenario, gt_all in tqdm(loader):
            x, y, is_available = map(lambda x: x.cuda(), (x, y, is_available))

            center = np.array(center)[0]

            gt_all = gt_all[0]

            for j in range(len(gt_all)):
                gt_all[j] = gt_all[j] - center

            V = vector_data[0]

            X, idx = V[:, :44], V[:, 44].flatten()

            if scenario!=scenario_id:

                if num_sce >= args.n_samples:
                    break

                num_sce += 1

                folder = "scenarios/"+scenario[0]

                os.system("mkdir "+folder)

                np.save(folder+"/gt_all",gt_all)

                # if iii>0:
            
                #     bboxarr = np.array(bboxarr)
                #     plt.xlim(bboxarr[:,0].min()-margin, bboxarr[:,1].max()+margin)
                #     plt.ylim(bboxarr[:,2].min()-margin, bboxarr[:,3].max()+margin)
                #     plt.savefig(os.path.join(args.save, f"{iii:0>2}.png"))
                #     plt.close()
            
                # figure(figsize=(15, 15), dpi=dpi)
                bboxarr = []
                scenario_id=scenario
            
                for i in np.unique(idx):
                    _X = X[idx == i]
                    np.save(folder+"/asset_"+str(int(i)),_X)
                    
                    # if _X[:, 5:12].sum() > 0:                        
                    #     plt.plot(_X[:, 0], _X[:, 1], linewidth=4, color="red")
                    # else:
                    #     plt.plot(_X[:, 0], _X[:, 1], color="black", alpha=0.2)

                
            
            y = y.squeeze(0).cpu().numpy()
            is_available = is_available.squeeze(0).long().cpu().numpy()

            gt = y[is_available > 0]

            yaw = yaw.item()
            rot_matrix = np.array(
            [
                [np.cos(-yaw), -np.sin(-yaw)],
                [np.sin(-yaw), np.cos(-yaw)],
            ]
            )
            shift = np.array(shift)[0]

            gt = gt@rot_matrix + shift

            gt = gt - center
            #plt.plot(gt[::10, 0],gt[::10, 1],"-o",color="blue")

            # gt2 = yy.squeeze(0).cpu().numpy()[is_available > 0]
            # gt2 = gt2 - center
            # plt.plot(gt2[::10, 0],gt2[::10, 1],"-o",color="green")

            bbox = [gt[:,0].min(0), gt[:,0].max(0), gt[:,1].min(0), gt[:,1].max(0)]
            bboxarr.append(bbox)

            # Model

            confidences_logits, logits = model(x)

            argmax = confidences_logits.argmax()
            if args.use_top1:
                confidences_logits = confidences_logits[:, argmax].unsqueeze(1)
                logits = logits[:, argmax].unsqueeze(1)

            # loss = pytorch_neg_multi_log_likelihood_batch(
            #     y, logits, confidences_logits, is_available
            # )
            confidences = torch.softmax(confidences_logits, dim=1)

            logits = logits.squeeze(0).cpu().numpy()
            confidences = confidences.squeeze(0).cpu().numpy()

            pred = logits[confidences.argmax()][is_available > 0]
            pred = pred@rot_matrix + shift 
            pred = pred - center

            #plt.plot(pred[::10, 0],pred[::10, 1],"-o",color="green",label="pred top 1")

            np.save(folder+"/agent_pred_"+str(iii),pred)
            np.save(folder+"/agent_true_"+str(iii),gt)
            

            iii += 1
            

            
            


if __name__ == "__main__":
    main()
