import argparse
import torch
from torch.utils.data import DataLoader
import lightning as L
from model import LightningModel
from waymo_dataset import WaymoDataset
import glob
from natsort import natsorted
#from lightning.pytorch.loggers import WandbLogger


torch.set_float32_matmul_precision('medium')

IMG_RES = 224
IN_CHANNELS = 25
TL = 80
N_TRAJS = 6
DEVICE = "gpu"
load_checkoint = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data", type=str, required=True, help="Path to rasterized data"
    )
    parser.add_argument(
        "--val-data", type=str, required=True, help="Path to rasterized data"
    )
    parser.add_argument(
        "--img-res",
        type=int,
        required=False,
        default=IMG_RES,
        help="Input images resolution",
    )
    parser.add_argument(
        "--in-channels",
        type=int,
        required=False,
        default=IN_CHANNELS,
        help="Input raster channels",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        required=False,
        default=TL,
        help="Number time step to predict",
    )
    parser.add_argument(
        "--n-traj",
        type=int,
        required=False,
        default=N_TRAJS,
        help="Number of trajectories to predict",
    )
    parser.add_argument(
        "--save_path", type=str, required=True, help="Path to save model and logs"
    )

    parser.add_argument(
        #"--model", type=str, required=False, default="xception71", help="CNN model name"
        "--model", type=str, required=False, default="resnet34", help="CNN model name"
    )
    parser.add_argument("--lr", type=float, required=False, default=1e-3)
    parser.add_argument("--batch-size", type=int, required=False, default=48)
    parser.add_argument("--n-epochs", type=int, required=False, default=60)

    parser.add_argument(
        "--n-monitor-validate",
        type=int,
        required=False,
        default=1,
        help="Validate model each n epochs",
    )

    args = parser.parse_args()

    return args




def main():

    args = parse_args()

    # Data parameters
    train_path = args.train_data
    val_path = args.val_data
    batch_size = args.batch_size
    num_workers = 12

    # Model parameters
    model_name = args.model
    in_channels=args.in_channels
    time_limit = args.time_limit
    n_traj = args.n_traj
    lr = args.lr

    # Training parameters
    n_epochs = args.n_epochs
    save_path = args.save_path

    # WandB logger
    # wandb_logger = WandbLogger(project='TrafficTrainer')
    # wandb_logger.experiment.config["batch_size"] = batch_size
    # wandb_logger.experiment.config["model_name"] = model_name

    # Training dataloader
    train_dataset = WaymoDataset(train_path)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        #persistent_workers=True,
    )

    # Validation dataloader
    val_dataset = WaymoDataset(val_path)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        #persistent_workers=True,
    )

    # Load checkpoint
    if load_checkoint:
        lastcheckpointdir = natsorted(glob.glob("./logs/"+model_name+"/lightning_logs/version_*"))[-1]
        checkpoint = glob.glob(lastcheckpointdir+"/checkpoints/*.ckpt")[0]
        print("Loading from checkpoint",checkpoint)
        model = LightningModel.load_from_checkpoint(checkpoint_path=checkpoint, model_name=model_name, in_channels=in_channels, time_limit=time_limit, n_traj=n_traj, lr=lr)
    else:
        model = LightningModel(model_name=model_name, in_channels=in_channels, time_limit=time_limit, n_traj=n_traj, lr=lr)

    print("Initializing trainer")
    trainer = L.Trainer(max_epochs=n_epochs, val_check_interval=0.5, accelerator=DEVICE, default_root_dir=save_path,  precision="16-mixed")#, logger=wandb_logger)

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    main()
