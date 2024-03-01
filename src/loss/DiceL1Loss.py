import torch
from monai.losses import DiceLoss


class DiceL1Loss(torch.nn.Module):
    def __init__(self, 
        alpha=1, 
        beta=0.1, 
        smooth_nr=0, 
        smooth_dr=1e-5, 
        squared_pred=True, 
        to_onehot_y=False, 
        sigmoid=True):
        super(DiceL1Loss, self).__init__()
        self.Dice = DiceLoss(smooth_nr=smooth_nr, smooth_dr=smooth_dr, squared_pred=squared_pred, to_onehot_y=to_onehot_y, sigmoid=sigmoid)
        self.L1 = torch.nn.L1Loss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred_mask, pred_image, mask, image):
        dice = self.Dice(pred_mask, mask)
        l1   = self.L1(pred_image, image)
        return self.alpha * dice + self.beta * l1


