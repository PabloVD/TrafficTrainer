
import torch
from torch import nn


# Negative Log-likelihood 
class NLL_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gt, logits, confidences, avails):
        
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
        error = torch.sum(((gt - logits) * avails)**2, dim=-1)  # reduce coords and use availability

        # with np.errstate(
        #     divide="ignore"
        # ):  # when confidence is 0 log goes to -inf, but we're fine with it
        #     # error (batch_size, num_modes)
        #     error = nn.functional.log_softmax(confidences, dim=1) - 0.5 * torch.sum(error, dim=-1)  # reduce time

        # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error = nn.functional.log_softmax(confidences, dim=1) - 0.5 * torch.sum(error, dim=-1)  # reduce time

        # error (batch_size, num_modes)
        error = -torch.logsumexp(error, dim=-1, keepdim=True)

        return torch.mean(error)


# L2 loss
class L2_loss(nn.Module):
    def __init__(self):
        super().__init__()

        self.l2loss = nn.MSELoss(reduction="none")
        self.softmax = nn.Softmax(dim=1)

    # gt: (batch,tl,2), logits: (batch,trajs,tl,2), confidences: (batch,trajs), avails: (batch, tl)
    def forward(self, gt, logits, confidences, avails):

        # Expand confidences to match the shape of logits and gt
        confidences_expanded = confidences[:, :, None, None]  # Shape: (batch, trajs, 1, 1)
        probs = self.softmax(confidences_expanded)

        # Apply avails mask to gt and logits
        gt_masked = gt * avails[:, :, None]  # Shape: (batch, tl, 2)
        logits_masked = logits * avails[:, None, :, None]  # Shape: (batch, trajs, tl, 2)

        # Compute the L2 loss
        loss = probs*self.l2loss( logits_masked, gt_masked[:, None, :, :] ) # Shape: (batch, trajs, tl, 2)

        return torch.mean(loss)
        
# L1 loss
class L1_loss(nn.Module):
    def __init__(self):
        super().__init__()

        self.l1loss = nn.L1Loss(reduction="none")
        self.softmax = nn.Softmax(dim=1)

    # gt: (batch,tl,2), logits: (batch,trajs,tl,2), confidences: (batch,trajs), avails: (batch, tl)
    def forward(self, gt, logits, confidences, avails):

        # Expand confidences to match the shape of logits and gt
        confidences_expanded = confidences[:, :, None, None]  # Shape: (batch, trajs, 1, 1)
        probs = self.softmax(confidences_expanded)

        # Apply avails mask to gt and logits
        gt_masked = gt * avails[:, :, None]  # Shape: (batch, tl, 2)
        logits_masked = logits * avails[:, None, :, None]  # Shape: (batch, trajs, tl, 2)

        # Compute the L2 loss
        loss = probs*self.l1loss( logits_masked, gt_masked[:, None, :, :] ) # Shape: (batch, trajs, tl, 2)

        return torch.mean(loss)