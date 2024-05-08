
import numpy as np
import timm
import torch
from torch import optim, nn
import pytorch_lightning as L

# CNN model
class Model(nn.Module):
    def __init__(self, model_name, in_channels, time_limit, n_traj):
        super().__init__()

        self.n_traj = n_traj
        self.time_limit = time_limit
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            in_chans=in_channels,
            num_classes=self.n_traj * 2 * self.time_limit + self.n_traj,
        )


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
    def __init__(self, model_name, in_channels, time_limit, n_traj, lr=1.e-3):
        super().__init__()

        self.model = Model(model_name, in_channels=in_channels, time_limit=time_limit, n_traj=n_traj)
        self.lr = lr
    
    def training_step(self, batch, batch_idx):
        
        x, y, is_available = batch
        confidences_logits, logits = self.model(x)
        loss = pytorch_neg_multi_log_likelihood_batch(y, logits, confidences_logits, is_available)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):

        x, y, is_available = batch
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
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=1.e-3*self.lr, max_lr=self.lr, cycle_momentum=False)

        return [optimizer], [scheduler]
        #return optimizer