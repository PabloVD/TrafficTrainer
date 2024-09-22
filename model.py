
import numpy as np
import timm
import torch
from torch import optim, nn
from lightning import LightningModule
from losses import NLL_loss, L2_loss, L1_loss
from data_utils.rasterizer_torch import Rasterizer
from data_utils.rasterizer_torch import get_rotation_matrix
from agents_module import AgentsModule

import time


debug = False

if debug:
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

        self.n_out_map = 1024
        self.n_out_agents = 1024
        self.n_hidden = self.n_out_map + self.n_out_agents
        self.n_out = self.n_traj * 3 * self.time_limit + self.n_traj

        self.map_module = timm.create_model(
            model_name,
            pretrained=True,
            in_chans=3,
            num_classes=self.n_out_map,
        )

        self.agents_module = AgentsModule(self.n_out_agents)

        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.n_hidden, self.n_hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.n_hidden//2, self.n_hidden//4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.n_hidden, self.n_out)
        )



    def forward(self, x_map, x_ego, bb_ego, x_agents, bb_agents):

        out_map = self.map_module(x_map)

        out_agents = self.agents_module(x_ego, bb_ego, x_agents, bb_agents)

        out_tot = torch.cat([out_map, out_agents],dim=-1)

        outputs = self.head(out_tot)

        confidences_logits, logits = (
            outputs[:, : self.n_traj],
            outputs[:, self.n_traj :],
        )

        logits = logits.view(-1, self.n_traj, self.time_limit, 3)
        logits = logits.cumsum(dim=2)

        return confidences_logits, logits


# Lightning Module
class LightningModel(LightningModule):
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

        if self.loss_type=="NLL":
            self.loss = NLL_loss()
        elif self.loss_type=="L2":
            self.loss = L2_loss()
        elif self.loss_type=="L1":
            self.loss = L1_loss()
        else:
            print("Loss not valid")

        self.save_hyperparameters(hparams)

        self.rasterizer = Rasterizer(zoom_fact = 3.)
    
    @torch.jit.export
    def next_step(self, currpos, curryaw, confidences, logits):

        # Extract batch size
        batch_size = currpos.shape[0]
        arr = torch.arange(batch_size)
        
        # Get the index of the maximum confidence for each row in the batch
        indmax_batch = confidences.argmax(dim=1)
        
        # Gather the logits based on the indices obtained from the maximum confidences
        pred = logits[arr, indmax_batch]

        # Calculate rotation matrix for each batch
        rot_matrix = get_rotation_matrix(-curryaw)
        
        # Rotating and translating prediction
        pred_rotated = torch.bmm(pred[:,:,:2], rot_matrix) + currpos.unsqueeze(1)  # shape: (batch_size, 10, 2)
        
        # Displace directly the vehicle to the predicted position
        nextpos = pred_rotated[:,0]  # Take the first position from the prediction
        
        # Estimate orientation
        # diffpos = nextpos - currpos
        # nextyaw = torch.atan2(diffpos[:, 1], diffpos[:, 0])
        nextyaw = pred[:,0,2] + curryaw

        return nextpos, nextyaw
    
    def update_step(self, XY, YAW, confidences_logits, logits, agind, tind):
        
        batch_size = XY.shape[0]
        btchrng = torch.arange(batch_size)
        currpos = XY[btchrng, agind, tind]
        curryaw = YAW[btchrng, agind, tind]
        nextpos, nextyaw = self.next_step(currpos, curryaw, confidences_logits, logits)
        XY[btchrng, agind, tind + 1] = nextpos
        YAW[btchrng, agind, tind + 1] = nextyaw

        return XY, YAW
    
    
    def get_raster_input_torch(self, batch, XY, YAW, tind):
        
        raster_dict = self.rasterizer.get_training_dict(tind,
                                 batch,
                                 XY,
                                 YAW,
                                 self.history,
                                 device=self.device)
        
        return raster_dict


    def training_step(self, batch, batch_idx):

        # starttime = time.time()
        # rastertime = 0.

        
        XY = batch["XY"]
        YAW = batch["YAWS"]

        loss = 0.

        for tind in range(self.history-1,self.history-1+self.future_window-1):

            # if tind>self.history-1:

            # rasterstarttime = time.time()

            raster_dict = self.get_raster_input_torch(batch, XY, YAW, tind)
            x_map = raster_dict["raster"]
            y = raster_dict["gt_marginal"]
            is_available = raster_dict["future_val_marginal"]
            x_ego = raster_dict["x_ego"]
            bb_ego = raster_dict["bb_ego"]
            x_agents = raster_dict["x_agents"]
            bb_agents = raster_dict["bb_agents"]
            
            # rastertime += time.time()-rasterstarttime

            # else:
            #     # first batch, no need to rasterize
            #     x = batch["raster"]
            #     y = batch["gt_marginal"]
            #     is_available = batch["future_val_marginal"]

            #     batchsize = XY.shape[0]
            #     btchrng = torch.arange(batchsize)
            #     yaw_ego = YAW[btchrng, batch["agent_ind"]]
            #     future_yaw = yaw_ego[:,tind+1:]
            #     current_yaw = yaw_ego[:,tind]
            #     future_yaw = future_yaw - current_yaw.view(-1,1)
            #     y = torch.cat([y,future_yaw.unsqueeze(-1)],dim=-1)
  
            if debug: np.save(outpath+"/batch_torch"+str(batch_idx)+"_time_"+str(tind),x[0].cpu().detach().numpy())

            confidences_logits, logits = self.model(x_map, x_ego, bb_ego, x_agents, bb_agents)
            logits = logits[:,:,:self.time_limit-(tind-(self.history-1))]   # Take those available within the future window
            loss += self.loss(y, logits, confidences_logits, is_available)

            XY, YAW = self.update_step(XY, YAW, confidences_logits, logits, batch["agent_ind"], tind)


        self.log("train_loss", loss)
        lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        self.log("lr",lr)

        # tottime = time.time()-starttime
        # print("Times [s]: Rasterization, Model, Total {:.2f}, {:.2f}, {:.2f}".format(rastertime,tottime-rastertime,tottime))

        return loss


    def validation_step(self, batch, batch_idx):

        
        XY = batch["XY"]
        YAW = batch["YAWS"]

        loss = 0.

        for tind in range(self.history-1,self.history-1+self.future_window-1):

            # if tind>self.history-1:

            # rasterstarttime = time.time()

            raster_dict = self.get_raster_input_torch(batch, XY, YAW, tind)
            x_map = raster_dict["raster"]
            y = raster_dict["gt_marginal"]
            is_available = raster_dict["future_val_marginal"]
            x_ego = raster_dict["x_ego"]
            bb_ego = raster_dict["bb_ego"]
            x_agents = raster_dict["x_agents"]
            bb_agents = raster_dict["bb_agents"]
            
            # rastertime += time.time()-rasterstarttime

            # else:
            #     # first batch, no need to rasterize
            #     x = batch["raster"]
            #     y = batch["gt_marginal"]
            #     is_available = batch["future_val_marginal"]

            #     batchsize = XY.shape[0]
            #     btchrng = torch.arange(batchsize)
            #     yaw_ego = YAW[btchrng, batch["agent_ind"]]
            #     future_yaw = yaw_ego[:,tind+1:]
            #     current_yaw = yaw_ego[:,tind]
            #     future_yaw = future_yaw - current_yaw.view(-1,1)
            #     y = torch.cat([y,future_yaw.unsqueeze(-1)],dim=-1)

            confidences_logits, logits = self.model(x_map, x_ego, bb_ego, x_agents, bb_agents)
            logits = logits[:,:,:self.time_limit-(tind-(self.history-1))]   # Take those available within the future window
            loss += self.loss(y, logits, confidences_logits, is_available)

            XY, YAW = self.update_step(XY, YAW, confidences_logits, logits, batch["agent_ind"], tind)

        self.log("val_loss", loss)

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