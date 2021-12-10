import torch
import torch.nn as nn
from models.backbone.resnet import Resnet
from models.upsample import Upsampling


class FPN(nn.Module):
    def __init__(self, num_layer, in_channels=3):
        super(FPN, self).__init__()
        self.bottom_up_pathway = Resnet(num_layer=num_layer, in_channels=in_channels)
        self.top_down_pathway = Upsampling()
        self.intermediate_layer_4 = nn.Conv2d(in_channels=256, out_channels=256,
                                              kernel_size=(1, 1), stride=(1, 1))
        self.intermediate_layer_3 = nn.Conv2d(in_channels=128, out_channels=256,
                                              kernel_size=(1, 1), stride=(1, 1))
        self.intermediate_layer_2 = nn.Conv2d(in_channels=64, out_channels=256,
                                              kernel_size=(1, 1), stride=(1, 1))

        self.final_layer_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1))
        self.final_layer_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1))
        self.final_layer_4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1))

    def forward(self, x):
        c2, c3, c4, c5 = self.bottom_up_pathway(x)
        p2, p3, p4, p5 = self.top_down_pathway(c5)

        intermediate_feature_4 = self.intermediate_layer_4(c4)
        intermediate_feature_3 = self.intermediate_layer_3(c3)
        intermediate_feature_2 = self.intermediate_layer_2(c2)
        f4 = self.final_layer_4(intermediate_feature_4 + p4)
        f3 = self.final_layer_3(intermediate_feature_3 + p3)
        f2 = self.final_layer_2(intermediate_feature_2 + p2)
        f5 = p5

        return [f2, f3, f4, f5]


if __name__ == '__main__':
    fpn = FPN(num_layer=18, in_channels=3)
    input = torch.rand((10, 3, 224, 224))
    output = fpn(input)
    print(output[-1].size())
