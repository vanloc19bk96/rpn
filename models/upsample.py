import torch.nn as nn


class Upsampling(nn.Module):
    def __init__(self):
        super().__init__()
        self.top_layer = nn.Conv2d(in_channels=512, out_channels=256,
                                   kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.up_sampling_4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up_sampling_3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up_sampling_2 = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x):
        p5 = self.top_layer(x)
        p4 = self.up_sampling_4(p5)
        p3 = self.up_sampling_3(p4)
        p2 = self.up_sampling_2(p3)

        return [p2, p3, p4, p5]
