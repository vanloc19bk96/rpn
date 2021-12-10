import torch
import torch.nn as nn
import torch.nn.functional as F

supported_layers = [18, 34, 50, 101, 152]


class Block(nn.Module):
    def __init__(self, num_layer, in_channels, intermediate_channel, stride=1):
        super().__init__()

        # check whether backbone supported or not
        assert num_layer in supported_layers, 'Backbone was not supported'
        self.num_layer = num_layer

        expansion = 1 if num_layer < 50 else 4

        # For resnet: 50 101 152
        if num_layer > 34:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=intermediate_channel, kernel_size=(1, 1),
                                   stride=(stride, stride), padding=0)
            self.conv3 = nn.Conv2d(in_channels=intermediate_channel, out_channels=intermediate_channel * expansion, kernel_size=(1, 1),
                                   stride=(1, 1), padding=0)
            self.bn3 = nn.BatchNorm2d(intermediate_channel * expansion)
        else:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=intermediate_channel, kernel_size=(3, 3),
                                   stride=(stride, stride), padding=1)
        self.conv2 = nn.Conv2d(in_channels=intermediate_channel, out_channels=intermediate_channel, kernel_size=(3, 3),
                               stride=(1, 1), padding=1)

        self.bn1 = nn.BatchNorm2d(intermediate_channel)

        self.bn2 = nn.BatchNorm2d(intermediate_channel)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=intermediate_channel * expansion, kernel_size=(1, 1),
                      stride=(stride, stride)),
            nn.BatchNorm2d(intermediate_channel * expansion)
        )

    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        if self.num_layer > 34:
            x = F.relu(self.bn3(self.conv3(x)))
        identity = self.shortcut(identity)
        x += identity

        return x


class Resnet(nn.Module):
    def __init__(self, num_layer, in_channels):
        super().__init__()

        assert num_layer in supported_layers, 'Backbone was not supported'

        self.num_layer = num_layer
        self.in_channels = in_channels

        if num_layer == 18:
            num_repeats = [2, 2, 2, 2]
        elif num_layer == 34 or num_layer == 50:
            num_repeats = [3, 4, 6, 3]
        elif num_layer == 101:
            num_repeats = [3, 4, 23, 3]
        else:
            num_repeats = [3, 8, 36, 3]

        self.layer1 = nn.Conv2d(in_channels=in_channels, out_channels=64,
                                kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer2 = self._make_layer(num_repeats[0], 64, 64, 1)
        self.layer3 = self._make_layer(num_repeats[1], 64, 128, 2)
        self.layer4 = self._make_layer(num_repeats[2], 128, 256, 2)
        self.layer5 = self._make_layer(num_repeats[3], 256, 512, 2)

    def forward(self, x):
        x1 = F.relu(self.pool1(self.bn1(self.layer1(x))))
        c2 = self.layer2(x1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)

        return [c2, c3, c4, c5]


    def _make_layer(self, num_repeats, in_channels, intermediate_channel, strides):
        layers = []
        for i, num_repeat in enumerate(range(num_repeats)):
            if i > 0:
                strides = 1
            block = Block(self.num_layer, in_channels, intermediate_channel, strides)
            in_channels = intermediate_channel
            layers.append(block)
        return nn.Sequential(*layers)

# import torchvision.models as models
# resnet18 = models.resnet18()
#
# resnet = Resnet(num_layer=18, in_channels=3)
# input = torch.rand((10, 3, 224, 224))
# output = resnet(input)
# print(output[-1].size())
#
# print(18*"*")
# new_model = nn.Sequential(*list(resnet18.children())[:-2])
# print(new_model(input).size())