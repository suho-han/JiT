import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Flatten label and prediction tensors
        inputs = torch.sigmoid(logits)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / \
            (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice


class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return bce_loss + dice_loss


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        numerator = 2 * torch.sum(probs * targets)
        denominator = torch.sum(probs * probs) + torch.sum(targets * targets)
        dice = (numerator + self.smooth) / (denominator + self.smooth)
        return 1 - dice


class WeightedBCEDiceLoss(nn.Module):
    def __init__(self, w_bce=1.0, w_dice=1.0, smooth=1.0):
        super(WeightedBCEDiceLoss, self).__init__()
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.w_bce * bce_loss + self.w_dice * dice_loss
