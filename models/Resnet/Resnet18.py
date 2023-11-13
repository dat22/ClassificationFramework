import torch
import torch.nn as nn
from torchvision import models

from Framework.ModelBase import ModelBase

class Resnet18(ModelBase):
    def __init__(self, in_channels, num_classes) -> None:
        super().__init__()
        self.network = models.resnet18(pretrained = True)
        self.network.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = self.network.fc.in_features
        self.network.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.network(x)