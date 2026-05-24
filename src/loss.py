import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim 

class MixLoss(nn.Module):
    def __init__(self, alpha=0.84, data_range=1.0):
        """
        Mix Loss combining MS-SSIM and L1 loss.
        As proposed in "Loss Functions for Image Restoration with Neural Networks" 
        by Zhao et al. (2017).
        
        Args:
            alpha (float): Weighting parameter. The authors empirically set it to 0.84 
                           to balance the two losses.
            data_range (float): The dynamic range of the input images 
                                (e.g., 1.0 for [0, 1] normalized tensors, 255 for [0, 255]).
        """
        super(MixLoss, self).__init__()
        self.alpha = alpha
        self.data_range = data_range

    def forward(self, processed_img, reference_img):
        # 1. MS_SSIM loss as explained in paper
        ms_ssim_val = ms_ssim(processed_img, reference_img, 
                              data_range=self.data_range, size_average=True)
        loss_ms_ssim = 1.0 - ms_ssim_val

        # 2. L1 Loss
        loss_l1 = F.l1_loss(processed_img, reference_img, reduction='mean')
        
        # 3. Combine them into the Mix Loss
        mix_loss = (self.alpha * loss_ms_ssim) + ((1.0 - self.alpha) * loss_l1)
        
        return mix_loss

# --- Example Usage ---
# Ensure your images are in the shape (Batch_Size, Channels, Height, Width)
# processed = torch.rand(4, 3, 256, 256)
# ground_truth = torch.rand(4, 3, 256, 256)

# criterion = MixLoss(alpha=0.84, data_range=1.0)
# loss = criterion(processed, ground_truth)
# loss.backward()