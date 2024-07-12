
import numpy as np
import timm
import torch
from torch import optim, nn
import lightning as L
import torchvision.transforms as transforms
from losses import NLL_loss, L2_loss, L1_loss

IMG_RES = 224
center_ego = [IMG_RES//4, IMG_RES//2]

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

        # self.noise_pos_std = 1#2.
        # self.noise_ang_std = 5#10.
        # self.noise_ang_std2 = 10#30.
        # self.ego_rotator = transforms.RandomRotation(self.noise_ang_std2, center=(center_ego[1],center_ego[0]))
        noise_ang_std2 = 20.
        self.others_rotator = transforms.RandomRotation(noise_ang_std2)

        self.save_hyperparameters(hparams)
    

    def ego_loc(self, img):

        tmp = img.reshape(img.shape[0],img.shape[1],-1)
        indices = torch.argmax(tmp,dim=-1)
        row = indices // IMG_RES
        column = indices - IMG_RES*row

        locs = torch.cat([row.unsqueeze(-1),column.unsqueeze(-1)],dim=-1)

        return locs



    def ego_transform(self, x):

        ego = x[:,3:3+11]

        n_timeframes = ego.shape[1]
        center_rot = [ center_ego[1], center_ego[0] ]

        # # option 1
        # K = 10.

        # angle = 0.
        # angle_incr = np.random.uniform(-3,3)
        # for i in reversed(range(n_timeframes-1)):

        #     # option 1
        #     angle += angle_incr

        #     translation = [ -K*np.sin(angle*np.pi/180.) , -K*(1.-np.cos(angle*np.pi/180.)) ]
        #     ego[:,i] = transforms.functional.affine(ego[:,i], translate=translation, angle=angle, scale=1, shear=0, center=center_rot)
        # # end option 1

        # option 2

        locs = self.ego_loc(ego)
        vel = locs[:,1:]-locs[:,:-1]
        vel = torch.sqrt(vel[:,:,0]**2. + vel[:,:,1]**2.)
        end_locs = locs[:,:-1]
        vel[torch.logical_and(end_locs[:,:,0]==0. , end_locs[:,:,1]==0.)]=0.

        vel = torch.clamp(vel, 0, 3)

        # angle_incr = 0.3#random.choice([-0.2,0.2])
        angle_incr = np.random.uniform(-0.3,0.3)
        # angle_incr = 1

        for b in range(ego.shape[0]):
            angle = 0.
            # translation = [0.,0.]

            for i in reversed(range(n_timeframes-1)):

                angle += vel[b,i].item()*angle_incr
                # angle += np.clip(vel[b,i].item()*angle_incr, a_min=-0.6, a_max=0.6)

                # if vel[b,i].item()>0:
                #     angle += angle_incr
                # else:
                #     angle 

                translation = [ -vel[b,i]*np.sin(angle*np.pi/180.) ,  -vel[b,i]*np.cos(angle*np.pi/180.) ]
                # translation = [ translation[0] + -vel[b,i]*np.sin(angle*np.pi/180.) , translation[1] + -vel[b,i]*np.cos(angle*np.pi/180.) ]

                # ego[b:b+1,i] = transforms.functional.affine(ego[b:b+1,-1], translate=translation, angle=angle, scale=1, shear=0, center=center_rot)
                ego[b:b+1,i] = transforms.functional.affine(ego[b:b+1,i+1], translate=translation, angle=angle, scale=1, shear=0, center=center_rot)
                # ego[b:b+1,i] = transforms.functional.affine(ego[b:b+1,i+1], translate=translation, angle=vel[b,i].item()*angle_incr, scale=1, shear=0, center=center_rot)


        # end option 2
            
        x[:,3:3+11] = ego

        return x
    
    def others_transform(self, x):

        others = x[:,3+11:3+11+11]
        x[:,3+11:3+11+11] = self.others_rotator(others)
        return x


    def training_step(self, batch, batch_idx):
        
        x, y, is_available = batch
        y = y[:,:self.time_limit]
        is_available = is_available[:,:self.time_limit]
        # x = self.ego_transform(x)
        # x = self.others_transform(x)
        x = self.transforms(x)
        confidences_logits, logits = self.model(x)
        loss = self.loss(y, logits, confidences_logits, is_available)
        self.log("train_loss", loss)
        lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        self.log("lr",lr)

        return loss


    def validation_step(self, batch, batch_idx):

        x, y, is_available = batch
        y = y[:,:self.time_limit]
        is_available = is_available[:,:self.time_limit]
        confidences_logits, logits = self.model(x)
        loss = self.loss(y, logits, confidences_logits, is_available)
        self.log("val_loss", loss, sync_dist=True)

        return loss
    

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