import argparse
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from torch.utils.data import DataLoader

from waymo_dataset import WaymoLoader
from model import pytorch_neg_multi_log_likelihood_batch
from natsort import natsorted
import glob
from model import LightningModel
from tqdm import tqdm

IMG_RES = 224
IN_CHANNELS = 25
TL = 80
N_TRAJS = 6

samplfrec = 5 # sample at 2Hz

make_plot = True

# Compute average and final displacement error
# Expects arrays of shape (T,2)
def displacement_error(x_true, x_pred):

    ade = np.sqrt(np.sum((x_pred[::samplfrec] - x_true[::samplfrec])**2.,axis=-1)).mean()
    fde = np.sqrt(np.sum((x_pred[-1] - x_true[-1])**2.,axis=-1))

    return ade, fde

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--save", type=str, required=True)
    parser.add_argument("--n-samples", type=int, required=False, default=50)
    parser.add_argument("--use-top1", action="store_true")

    args = parser.parse_args()

    return args


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

    ade_list, fde_list = [], []

    iii = 0
    with torch.no_grad():
        for x, y, is_available, vector_data in tqdm(loader):
            x, y, is_available = map(lambda x: x.cuda(), (x, y, is_available))

            confidences_logits, logits = model(x)

            argmax = confidences_logits.argmax()
            if args.use_top1:
                confidences_logits = confidences_logits[:, argmax].unsqueeze(1)
                logits = logits[:, argmax].unsqueeze(1)

            loss = pytorch_neg_multi_log_likelihood_batch(
                y, logits, confidences_logits, is_available
            )
            confidences = torch.softmax(confidences_logits, dim=1)
            V = vector_data[0]

            X, idx = V[:, :44], V[:, 44].flatten()

            if make_plot:
                figure(figsize=(15, 15), dpi=80)
                for i in np.unique(idx):
                    _X = X[idx == i]
                    if _X[:, 5:12].sum() > 0:
                        plt.plot(_X[:, 0], _X[:, 1], linewidth=4, color="red")
                    else:
                        plt.plot(_X[:, 0], _X[:, 1], color="black")
                    plt.xlim([-224 // 4, 224 // 4])
                    plt.ylim([-224 // 4, 224 // 4])

            logits = logits.squeeze(0).cpu().numpy()
            y = y.squeeze(0).cpu().numpy()
            is_available = is_available.squeeze(0).long().cpu().numpy()
            confidences = confidences.squeeze(0).cpu().numpy()

            if y[is_available > 0].shape[0]>samplfrec:

                ade, fde = np.zeros(N_TRAJS), np.zeros(N_TRAJS)
                for j in range(N_TRAJS):

                    #print(y.shape, logits.shape, is_available.shape)
                    ade[j], fde[j] = displacement_error(y[is_available > 0][:, :2], logits[j][is_available > 0][:, :2])
                    
                ade, fde = ade.min(), fde.min()
                ade_list.append(ade); fde_list.append(fde)

            if make_plot:
                plt.plot(
                    y[is_available > 0][::10, 0],
                    y[is_available > 0][::10, 1],
                    "-o",
                    label="gt",
                )

                plt.plot(
                    logits[confidences.argmax()][is_available > 0][::10, 0],
                    logits[confidences.argmax()][is_available > 0][::10, 1],
                    "-o",
                    label="pred top 1",
                )

                if not args.use_top1:
                    for traj_id in range(len(logits)):
                        if traj_id == argmax:
                            continue

                        alpha = confidences[traj_id].item()
                        plt.plot(
                            logits[traj_id][is_available > 0][::10, 0],
                            logits[traj_id][is_available > 0][::10, 1],
                            "-o",
                            label=f"pred {traj_id} {alpha:.3f}",
                            alpha=alpha,
                        )


                #plt.title(loss.item())
                plt.title("ADE={:.2f}, FDE={:.2f}".format(ade,fde))
                plt.legend()
                plt.savefig(os.path.join(args.save, f"{iii:0>2}_{loss.item():.3f}.png"),bbox_inches='tight')
                plt.close()

            iii += 1
            if iii == args.n_samples:
                break

    ade_list, fde_list = np.array(ade_list), np.array(fde_list)
    print("minADE={:.2f}+-{:.2f}, minFDE={:.2f}+-{:.2f}".format(ade_list.mean(),ade_list.std(), fde_list.mean(), fde_list.std()))


if __name__ == "__main__":
    main()
