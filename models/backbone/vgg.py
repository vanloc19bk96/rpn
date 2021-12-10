import torch.nn as nn
import torchvision.models as models


class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.vgg16(pretrained=True)
        model.eval()
        for param in model.features.parameters():
            param.requires_grad = False
        self.features = model.features[0:24]

    def forward(self, x):
        for f in self.features:
            x = f(x)
        return x

from cv2 import cv2
import torch
# vgg = VGG()
# img = cv2.imread('../../data/test/images/road1.png')
# img = cv2.resize(img, (800, 800))
# img = torch.transpose(torch.tensor(img).float(), 2, 0)
# prediction = vgg(torch.unsqueeze(img, 0))
# print(prediction.size())
