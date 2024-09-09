
import numpy as np
import timm
import torch
from torch import optim, nn
import lightning as L
import torchvision.transforms as transforms
from losses import NLL_loss, L2_loss, L1_loss
# from data_utils.rasterizer import rasterizer
from data_utils.rasterizer_torch import rasterizer_torch
from data_utils.rasterizer_torch import get_rotation_matrix

import time

IMG_RES = 224
center_ego = [IMG_RES//4, IMG_RES//2]

# import os
# outpath = "rendertests"
# if not os.path.exists(outpath):
#     os.system("mkdir "+outpath)
# else:
#     os.system("rm "+outpath+"/*")

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

        self.model = Model(model_name, in_channels=in_channels, time_limit=time_limit, n_traj=n_traj)
        
        self.lr = hparams["lr"]
        self.weight_decay = hparams["weight_decay"]
        self.sched = hparams["scheduler"]
        self.time_limit = hparams["time_limit"]
        self.loss_type = hparams["loss"]

        self.history = 10
        self.future_window = 10#self.time_limit
        
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

        # Extract batch size
        batch_size = XY.shape[0]

        arr = torch.arange(batch_size)
        
        # Gather the corresponding xy and yaw from each ego agent
        xy_ag = XY[arr, agind]
        yaw_ag = YAW[arr, agind]
        
        # Get the index of the maximum confidence for each row in the batch
        indmax_batch = confidences.argmax(dim=1)
        
        # Gather the logits based on the indices obtained from the maximum confidences
        pred = logits[arr, indmax_batch]
        
        # Get current position and yaw
        currpos = xy_ag[:, currind]
        curryaw = yaw_ag[:, currind]*np.pi/180.0
        
        # Calculate rotation matrix for each batch
        rot_matrix = get_rotation_matrix(-curryaw)
        
        # Rotating and translating prediction
        pred_rotated = torch.bmm(pred, rot_matrix) + currpos.unsqueeze(1)  # shape: (batch_size, 10, 2)
        
        # Displace directly the vehicle to the predicted position
        nextpos = pred_rotated[:, 0]  # Take the first position from the prediction
        
        # Estimate orientation
        diffpos = nextpos - currpos
        newyaw = torch.atan2(diffpos[:, 1], diffpos[:, 0])*180.0/np.pi
        
        # Update XY and YAW for the next step
        XY[arr, agind, currind + 1] = nextpos
        YAW[arr, agind, currind + 1] = newyaw

        return XY, YAW
    
    def get_raster_input(self, batch, XY, YAW, tind):

        x = torch.zeros_like(batch["raster"])

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
            
            x[b] = torch.tensor(raster_dict["raster"])

        return x
    
    def get_raster_input_torch(self, batch, XY, YAW, tind):
        
        raster_dict = rasterizer_torch(tind,
                                 batch,
                                 XY,
                                 YAW,
                                 self.history,
                                 device=self.device)
        
        x = torch.tensor(raster_dict["raster"], device=self.device)

        return x


    def training_step(self, batch, batch_idx):

        # starttime = time.time()
        # rastertime = 0.

        x = batch["raster"]
        XY = batch["XY"]
        YAW = batch["YAWS"]

        loss = 0.

        for tind in range(self.history-1,self.history-1+self.future_window-1):

            y = batch["gt_marginal"][:,tind-(self.history-1):]
            is_available = batch["future_val_marginal"][:,tind-(self.history-1):]

            if tind>self.history-1:

                # rasterstarttime = time.time()

                # x = self.get_raster_input(batch, XY, YAW, tind)           
                x = self.get_raster_input_torch(batch, XY, YAW, tind)  
                
                # rastertime += time.time()-rasterstarttime

            # np.save(outpath+"/batch_"+str(batch_idx)+"_time_"+str(tind),x.cpu().detach().numpy())

            confidences_logits, logits = self.model(x)
            logits = logits[:,:,tind-(self.history-1):]
            loss += self.loss(y, logits, confidences_logits, is_available)

            XY, YAW = self.update_step(XY, YAW, confidences_logits, logits, batch["agent_ind"], tind)


        self.log("train_loss", loss)
        lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        self.log("lr",lr)

        # tottime = time.time()-starttime
        # print("Times [s]: Rasterization, Model, Total {:.2f}, {:.2f}, {:.2f}".format(rastertime,tottime-rastertime,tottime))

        return loss


    def validation_step(self, batch, batch_idx):

        x = batch["raster"]
        XY = batch["XY"]
        YAW = batch["YAWS"]

        loss = 0.

        for tind in range(self.history-1,self.history-1+self.future_window-1):

            y = batch["gt_marginal"][:,tind-(self.history-1):]
            is_available = batch["future_val_marginal"][:,tind-(self.history-1):]

            if tind>self.history-1:

                # x = self.get_raster_input(batch, XY, YAW, tind)           
                x = self.get_raster_input_torch(batch, XY, YAW, tind)            

            confidences_logits, logits = self.model(x)
            logits = logits[:,:,tind-(self.history-1):]
            loss += self.loss(y, logits, confidences_logits, is_available)

            XY, YAW = self.update_step(XY, YAW, confidences_logits, logits, batch["agent_ind"], tind)

        self.log("val_loss", loss)

        return loss
    

    # def test_step(self, batch, batch_idx):

    #     x, y, is_available = batch
    #     y = y[:,:self.time_limit]
    #     is_available = is_available[:,:self.time_limit]
    #     confidences_logits, logits = self.model(x)
    #     loss = self.loss(y, logits, confidences_logits, is_available)
    #     self.log("test_loss", loss, sync_dist=True)

    #     return loss


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