
import numpy as np
import timm
import torch
from torch import optim, nn
import lightning as L
import torchvision.transforms as transforms

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


# Loss function
def pytorch_neg_multi_log_likelihood_batch(gt, logits, confidences, avails):
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        logits (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """

    # convert to (batch_size, num_modes, future_len, num_coords)
    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords

    # error (batch_size, num_modes, future_len)
    error = torch.sum(
        ((gt - logits) * avails) ** 2, dim=-1
    )  # reduce coords and use availability

    with np.errstate(
        divide="ignore"
    ):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error = nn.functional.log_softmax(confidences, dim=1) - 0.5 * torch.sum(
            error, dim=-1
        )  # reduce time

    # error (batch_size, num_modes)
    error = -torch.logsumexp(error, dim=-1, keepdim=True)

    return torch.mean(error)


# Lightning Module
class LightningModel(L.LightningModule):
    def __init__(self, model_name, in_channels, time_limit, n_traj, lr, weight_decay, sched):
        super().__init__()

        self.model = Model(model_name, in_channels=in_channels, time_limit=time_limit, n_traj=n_traj)
        self.lr = lr
        self.weight_decay = weight_decay
        self.sched = sched
        self.time_limit = time_limit
        
        self.transforms = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(size=(IMG_RES, IMG_RES),scale=(0.95,1.)),
        ])

        self.noise_pos_std = 2.
        self.noise_ang_std = 10.
        self.noise_ang_std2 = 30.
        self.ego_rotator = transforms.RandomRotation(self.noise_ang_std2, center=(center_ego[1],center_ego[0]))

    
    def ego_loc(self, img):

        tmp = img.reshape(img.shape[0],img.shape[1],-1)
        indices = torch.argmax(tmp,dim=-1)
        row = indices // IMG_RES
        column = indices - IMG_RES*row

        locs = torch.cat([row.unsqueeze(-1),column.unsqueeze(-1)],dim=-1)

        return locs

    def ego_transform(self, x):

        ego = x[:,3:3+11]

        timeframes = ego.shape[1]

        noise_pos = self.noise_pos_std*torch.randn((timeframes,2))
        noise_pos = torch.cumsum(noise_pos, dim=1)
        noise_pos = torch.flip(noise_pos,dims=(0,))

        noise_ang = self.noise_ang_std*torch.randn((timeframes))
        noise_ang = torch.cumsum(noise_ang, dim=0)
        noise_ang = torch.flip(noise_ang,dims=(0,))

        # Random rotation around end position of the ego vehicle
        ego = self.ego_rotator(ego)

        locs = self.ego_loc(ego)

        for i in range(timeframes):
            for b in range(ego.shape[0]):
                
                translation = [noise_pos[i,0],noise_pos[i,1]]
                center_rot = (locs[b,i,1], locs[b,i,0])
                angle = noise_ang[i].item()

                # Random translation and rotation around vehicle
                ego[b:b+1,i] = transforms.functional.affine(ego[b:b+1,i], translate=translation, angle=angle, scale=1, shear=0, center=center_rot)
            
        x[:,3:3+11] = ego

        return x
            
    def training_step(self, batch, batch_idx):
        
        x, y, is_available = batch
        y = y[:,:self.time_limit]
        is_available = is_available[:,:self.time_limit]
        x = self.ego_transform(x)
        x = self.transforms(x)
        confidences_logits, logits = self.model(x)
        loss = pytorch_neg_multi_log_likelihood_batch(y, logits, confidences_logits, is_available)
        self.log("train_loss", loss)
        lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        self.log("lr",lr)

        return loss

    def validation_step(self, batch, batch_idx):

        x, y, is_available = batch
        y = y[:,:self.time_limit]
        is_available = is_available[:,:self.time_limit]
        confidences_logits, logits = self.model(x)
        loss = pytorch_neg_multi_log_likelihood_batch(y, logits, confidences_logits, is_available)
        self.log("val_loss", loss)

        return loss
    
    def test_step(self, batch, batch_idx):

        x, y, is_available = batch
        confidences_logits, logits = self.model(x)
        loss = pytorch_neg_multi_log_likelihood_batch(y, logits, confidences_logits, is_available)
        self.log("test_loss", loss)

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