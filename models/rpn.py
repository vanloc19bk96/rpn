import torch.nn as nn
import torch


class RPN(nn.Module):
    def __init__(self, out_channels=512, num_anchors=9):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=512, out_channels=out_channels,
                               kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.reg_layer = nn.Conv2d(in_channels=out_channels, out_channels=num_anchors * 4, kernel_size=(1, 1),
                                   stride=(1, 1), padding=0)
        self.cls_layer = nn.Conv2d(in_channels=out_channels, out_channels=num_anchors * 1, kernel_size=(1, 1),
                                   stride=(1, 1), padding=0)

        # self.conv1.weight.data.normal_(0, 0.01)
        # self.conv1.bias.data.zero_()
        # self.reg_layer.weight.data.normal_(0, 0.01)
        # self.reg_layer.bias.data.zero_()
        # self.cls_layer.weight.data.normal_(0, 0.01)
        # self.cls_layer.bias.data.zero_()

    def forward(self, input_feature_map):
        x = torch.relu(self.conv1(input_feature_map))
        pred_anchor_locs = self.reg_layer(x)
        pred_cls_scores = torch.sigmoid(self.cls_layer(x))

        return pred_anchor_locs, pred_cls_scores
