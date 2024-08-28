import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from waymo_dataset import WaymoLoader
from tqdm import tqdm
from torchscript_converter import export_model

N_TRAJS = 6
ntrajs = 2

device = "cuda"

samplfrec = 1 # sample at 2Hz for testing
time_limit = 20

def eleventoten(x):
    roads = x[:,:3]
    ego = x[:,4:3+11]
    others = x[:,4+11:]
    return torch.cat([roads,ego,others],dim=1)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=str, required=True, help="Model name")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--n-samples", type=int, required=False, default=50)
    parser.add_argument("--store-data", action="store_true")
    parser.add_argument("--export", action="store_true")

    args = parser.parse_args()

    return args


def last_index(valid):

    last_one_index = np.where(valid == 1)[0][-1]#len(valid) - 1 - valid[::-1].index(1)

    return last_one_index

def interpolate_trajectories(traj, valid):

    int_traj = traj.copy()
    traj_val = int_traj[valid>0]

    idxs = np.array(range(len(int_traj)))

    int_traj[valid==0,0] = np.interp(idxs[valid==0], idxs[valid>0], traj_val[:,0])
    int_traj[valid==0,1] = np.interp(idxs[valid==0], idxs[valid>0], traj_val[:,1])

    return int_traj

# Compute average and final displacement error
# Expects arrays of shape (T,2)
def displacement_error(x_true, x_pred):

    x_true = x_true[:time_limit]
    x_pred = x_pred[:time_limit]

    ade = np.sqrt(np.sum((x_pred[::samplfrec] - x_true[::samplfrec])**2.,axis=-1)).mean()
    fde = np.sqrt(np.sum((x_pred[-1] - x_true[-1])**2.,axis=-1))

    return ade, fde

def main():
    args = parse_args()

    store = args.store_data
    print("Store data for videos:",store)

    if args.export:
        print("Exporting model")
        export_model(args.m)

    print("Testing model",args.m)

    # Load model
    model = torch.jit.load("models/"+args.m+".pt")
    model = model.to(device)

    loader = DataLoader(
        WaymoLoader(args.data, return_vector=True),
        batch_size=1,
        num_workers=1,
        shuffle=False,
    )

    scenario_id = None

    if store:
        os.system("mkdir scenarios")

    iii = 0
    num_sce = 0

    ade_list, fde_list = [], []

    with torch.no_grad():
        for x, y, is_available, vector_data, center, shift, yaw, scenario, gt_all, val_all in tqdm(loader):
            x, y, is_available = map(lambda x: x.cuda(), (x, y, is_available))

            x = eleventoten(x)

            center = np.array(center)[0]

            gt_all = gt_all[0]
            val_all= val_all[0]

            for j in range(len(gt_all)):
                gt_all[j] = gt_all[j] - center

            V = vector_data[0]

            X, idx = V[:, :44], V[:, 44].flatten()

            if store:
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

            # Model
            confidences_logits, logits = model(x)

            confidences = torch.softmax(confidences_logits, dim=1)

            logits = logits.squeeze(0).cpu().numpy()
            confidences = confidences.squeeze(0).cpu().numpy()
            y = y.squeeze(0).cpu().numpy()
            is_available = is_available.squeeze(0).long().cpu().numpy()

            y = y[:time_limit]
            is_available = is_available[:time_limit]

            # Test results
            if y[is_available > 0].shape[0]>samplfrec:

                ade, fde = np.zeros(N_TRAJS), np.zeros(N_TRAJS)
                for j in range(N_TRAJS):

                    #print(y.shape, logits.shape, is_available.shape)
                    ade[j], fde[j] = displacement_error(y[is_available > 0][:, :2], logits[j][is_available > 0][:, :2])
                    
                ade, fde = ade.min(), fde.min()
                ade_list.append(ade); fde_list.append(fde)

            if store:

                last_idx = last_index(is_available)
                is_available = is_available[:last_idx+1]
                y = y[:last_idx+1]

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


                np.save(folder+"/agent_true_"+str(iii),gt)
        
                # Save trajectories in order of confidence
                for traj in range(ntrajs):
                
                    indmax = confidences.argmax()

                    pred = logits[confidences.argmax()][:last_idx+1]#[is_available > 0]
                    pred = pred@rot_matrix + shift 
                    pred = pred - center

                    np.save(folder+"/agent_pred_"+str(iii)+"_traj_"+str(traj),pred)

                    confidences = np.delete(confidences, indmax)
                    logits = np.delete(logits, indmax, axis=0)
            
            iii += 1

    ade_list, fde_list = np.array(ade_list), np.array(fde_list)
    print("minADE={:.2f}+-{:.2f}, minFDE={:.2f}+-{:.2f}".format(ade_list.mean(),ade_list.std(), fde_list.mean(), fde_list.std()))
            
            

            
            


if __name__ == "__main__":
    main()
