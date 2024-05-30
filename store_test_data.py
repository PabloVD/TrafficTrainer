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
ntrajs = 2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--save", type=str, required=True)
    parser.add_argument("--n-samples", type=int, required=False, default=50)
    parser.add_argument("--use-top1", action="store_true")

    args = parser.parse_args()

    return args


def last_index(valid):

    last_one_index = np.where(valid == 1)[0][-1]#len(valid) - 1 - valid[::-1].index(1)
    #print(valid.shape,np.where(valid == 1).shape)

    return last_one_index

def interpolate_trajectories(traj, valid):

    int_traj = traj.copy()
    traj_val = int_traj[valid>0]

    idxs = np.array(range(len(int_traj)))

    int_traj[valid==0,0] = np.interp(idxs[valid==0], idxs[valid>0], traj_val[:,0])
    int_traj[valid==0,1] = np.interp(idxs[valid==0], idxs[valid>0], traj_val[:,1])

    return int_traj


def main():
    args = parse_args()
    if not os.path.exists(args.save):
        os.mkdir(args.save)

    model_name = args.model
    checkpoint = model_name+"-bestmodel.ckpt"
    print("Loading from checkpoint",checkpoint)
    model = LightningModel.load_from_checkpoint(checkpoint_path=checkpoint, model_name=model_name, in_channels=IN_CHANNELS, time_limit=TL, n_traj=N_TRAJS, lr=1.e-3, weight_decay=1.e-3).cuda().eval()
    
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
        for x, y, is_available, vector_data, center, shift, yaw, scenario, gt_all, val_all in tqdm(loader):
            x, y, is_available = map(lambda x: x.cuda(), (x, y, is_available))

            center = np.array(center)[0]

            gt_all = gt_all[0]
            val_all= val_all[0]

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

                gt_all = gt_all.cpu().numpy()
                val_all = val_all.cpu().numpy()

                for ag in range(len(gt_all)):

                    gt_all[ag] = interpolate_trajectories(gt_all[ag], val_all[ag])

                np.save(folder+"/gt_all",gt_all)

                scenario_id=scenario
            
                for i in np.unique(idx):
                    _X = X[idx == i]
                    np.save(folder+"/asset_"+str(int(i)),_X)
                    

            y = y.squeeze(0).cpu().numpy()
            is_available = is_available.squeeze(0).long().cpu().numpy()

            last_idx = last_index(is_available)
            is_available = is_available[:last_idx+1]
            y = y[:last_idx+1]

            #print(remove_last_zeros(is_available))

            # gt = y[is_available > 0]
            gt = interpolate_trajectories(y, is_available)

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

            # gt2 = yy.squeeze(0).cpu().numpy()[is_available > 0]
            # gt2 = gt2 - center

            np.save(folder+"/agent_true_"+str(iii),gt)

            # Model

            confidences_logits, logits = model(x)

            confidences = torch.softmax(confidences_logits, dim=1)

            logits = logits.squeeze(0).cpu().numpy()
            confidences = confidences.squeeze(0).cpu().numpy()

            for traj in range(ntrajs):
            
                indmax = confidences.argmax()

                pred = logits[confidences.argmax()][:last_idx+1]#[is_available > 0]
                pred = pred@rot_matrix + shift 
                pred = pred - center

                np.save(folder+"/agent_pred_"+str(iii)+"_traj_"+str(traj),pred)

                confidences = np.delete(confidences, indmax)
                logits = np.delete(logits, indmax, axis=0)
            

            iii += 1
            
            

            
            


if __name__ == "__main__":
    main()
