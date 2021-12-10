import torch.nn as nn
import torch.nn.functional as F


class Simple(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.backbone = nn.Conv2d(in_channels=3, kernel_size=(3, 3), out_channels=int(out_channels))
        self.backbone.weight.data.normal_(0, 0.01)
        self.backbone.bias.data.zero_()

    def forward(self, x):
        return F.relu(self.backbone(x))
