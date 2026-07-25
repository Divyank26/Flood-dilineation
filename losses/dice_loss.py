import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self, smooth=1):

        super().__init__()

        self.smooth = smooth

    def forward(self, prediction, target):

        prediction = torch.sigmoid(prediction)

        prediction = prediction.view(-1)

        target = target.view(-1)

        intersection = (prediction * target).sum()

        dice = (
            2 * intersection + self.smooth
        ) / (
            prediction.sum()
            + target.sum()
            + self.smooth
        )

        return 1 - dice