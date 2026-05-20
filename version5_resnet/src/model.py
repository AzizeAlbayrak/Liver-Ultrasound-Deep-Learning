import torch.nn as nn
from torchvision import models


class TransferResNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        #ResNet18'i indir
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        #dropout katmanı
        num_ftrs = self.resnet.fc.in_features

        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),  #her eğitim adımında nöronların %50'sini random kapanır
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)