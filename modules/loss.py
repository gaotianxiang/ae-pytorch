import torch
import torch.nn as nn


class AELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_function = nn.MSELoss()

    def forward(self, inputs, reconstructions):
        loss = self.loss_function(reconstructions, inputs)
        return loss
