
import numpy as np
import timm
import torch
from torch import optim, nn
import lightning as L
import torchvision.transforms as transforms
from losses import NLL_loss, L2_loss, L1_loss
from data_utils.rasterizer import rasterizer

IMG_RES = 224
center_ego = [IMG_RES//4, IMG_RES//2]

import os
outpath = "rendertests"
if not os.path.exists(outpath):
    os.system("mkdir "+outpath)
else:
    os.system("rm "+outpath+"/*")

# CNN model
class Model(nn.Module):
    def __init__(self, model_name, in_channels, time_limit, n_traj):
        super().__init__()

        self.n_traj = n_traj
        self.time_limit = time_limit

        self.n_hidden = 2**11
        self.n_out = self.n_traj * 2 * self.time_limit + self.n_traj

        self.model = timm.create_model(
            model_name,
            pretrained=True,
            in_chans=in_channels,
            num_classes=self.n_out,
        )

        # self.head = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(self.n_hidden, self.n_hidden),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(self.n_hidden, self.n_out)
        # )

        # self.model = nn.Sequential(
        #     self.backbone,
        #     self.head
        # )


    def forward(self, x):

        outputs = self.model(x)

        confidences_logits, logits = (
            outputs[:, : self.n_traj],
            outputs[:, self.n_traj :],
        )

        logits = logits.view(-1, self.n_traj, self.time_limit, 2)

        return confidences_logits, logits


# Lightning Module
class LightningModel(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        model_name = hparams["model_name"]
        in_channels = hparams["in_channels"]
        time_limit = hparams["time_limit"]
        n_traj = hparams["n_traj"]
        self.history = 10

        self.model = Model(model_name, in_channels=in_channels, time_limit=time_limit, n_traj=n_traj)
        
        self.lr = hparams["lr"]
        self.weight_decay = hparams["weight_decay"]
        self.sched = hparams["scheduler"]
        self.time_limit = hparams["time_limit"]
        self.loss_type = hparams["loss"]
        
        self.transforms = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(size=(IMG_RES, IMG_RES),scale=(0.95,1.)),
        ])

        if self.loss_type=="NLL":
            self.loss = NLL_loss()
        elif self.loss_type=="L2":
            self.loss = L2_loss()
        elif self.loss_type=="L1":
            self.loss = L1_loss()
        else:
            print("Loss not valid")

        self.save_hyperparameters(hparams)
    

    def update_step(self, XY, YAW, confidences, logits, agind, currind):

        # TO DO improve with array multiplication
        for j in range(XY.shape[0]):

            xy_ag = XY[j, agind[j]]
            yaw_ag = YAW[j, agind[j]]

            indmax = confidences[j].argmax()
            # sortedtens, indices = torch.sort(confidences[j])
            # indmax = indices[-2]

            pred = logits[j,indmax]

            currpos, curryaw = xy_ag[currind], yaw_ag[currind]*np.pi/180.

            c = torch.cos(-curryaw)
            s = torch.sin(-curryaw)
            rot_matrix = torch.stack([torch.stack([c, -s]),torch.stack([s, c])])
        
            pred = pred@rot_matrix + currpos 

            # Displace directly the vehicle to the predicted position
            nextpos =  pred[0]

            # Estimate orientation
            diffpos = nextpos - currpos
            newyaw = torch.arctan2(diffpos[1], diffpos[0])*180./np.pi

            XY[j, agind[j],currind+1] = nextpos
            YAW[j, agind[j],currind+1] = newyaw

        return XY, YAW
    
    def get_raster_input(self, x, batch, XY, YAW, tind):

        x = torch.zeros_like(x)

        for b in range(x.shape[0]):

            agents_data = {"agents_ids":batch["agents_ids"][b].cpu().detach().numpy(),
                            "agents_valid":batch["agents_valid"][b].cpu().detach().numpy(),
                            "XY":XY[b].cpu().detach().numpy(),
                            "YAWS":YAW[b].cpu().detach().numpy(),
                            "lengths":batch["lengths"][b].cpu().detach().numpy(),
                            "widths":batch["widths"][b].cpu().detach().numpy()}

            roads_data = {"roads_ids":batch["roads_ids"][b].cpu().detach().numpy(),
                            "roads_valid":batch["roads_valid"][b].cpu().detach().numpy(),
                            "roads_coords":batch["roads_coords"][b].cpu().detach().numpy()}
            
            tl_data = {"tl_ids":batch["tl_ids"][b].cpu().detach().numpy(),
                        "tl_valid":batch["tl_valid"][b].cpu().detach().numpy(),
                        "tl_states":batch["tl_states"][b].cpu().detach().numpy()}
            
            raster_dict = rasterizer(batch["agent_ind"][b].cpu(),
                                        tind,
                                        agents_data,
                                        roads_data,
                                        tl_data,
                                        self.history,
                                        prerender=False)
            
            x[b] = torch.Tensor(raster_dict["raster"])

        return x


    def training_step(self, batch, batch_idx):
        
        x = batch["raster"]
        XY = batch["XY"]
        YAW = batch["YAWS"]

        loss = 0.

        for tind in range(self.history-1,self.history-1+self.time_limit-1):

            print(tind)

            y = batch["gt_marginal"][:,tind-(self.history-1):]
            is_available = batch["future_val_marginal"][:,tind-(self.history-1):]

            if tind>self.history-1:

                x = self.get_raster_input(x, batch, XY, YAW, tind)                

            np.save(outpath+"/batch_"+str(batch_idx)+"_time_"+str(tind),x.cpu().detach().numpy())

            confidences_logits, logits = self.model(x)
            logits = logits[:,:,tind-(self.history-1):]
            loss += self.loss(y, logits, confidences_logits, is_available)

            XY, YAW = self.update_step(XY, YAW, confidences_logits, logits, batch["agent_ind"], tind)


        self.log("train_loss", loss)
        lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        self.log("lr",lr)

        return loss


    # def validation_step(self, batch, batch_idx):

    #     x, y, is_available = batch
    #     y = y[:,:self.time_limit]
    #     is_available = is_available[:,:self.time_limit]
    #     confidences_logits, logits = self.model(x)
    #     loss = self.loss(y, logits, confidences_logits, is_available)
    #     self.log("val_loss", loss, sync_dist=True)

    #     return loss
    

    def test_step(self, batch, batch_idx):

        x, y, is_available = batch
        y = y[:,:self.time_limit]
        is_available = is_available[:,:self.time_limit]
        confidences_logits, logits = self.model(x)
        loss = self.loss(y, logits, confidences_logits, is_available)
        self.log("test_loss", loss, sync_dist=True)

        return loss


    def configure_optimizers(self):

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.sched=="multistep":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(50, 200, 50)), gamma=0.1)
        elif self.sched=="cyclic":
            scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=1.e-2*self.lr, max_lr=self.lr, step_size_up=20,cycle_momentum=False)
        elif self.sched=="onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, total_steps=200)
        elif self.sched=="cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=20,eta_min=max(1e-2*self.lr, 1e-6))
        else:
            return optimizer

        return [optimizer], [scheduler]
        

    def forward(self, x):

        return self.model(x)