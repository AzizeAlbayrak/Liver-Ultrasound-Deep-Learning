#ResNet18 Entegrasyonu
import torch.nn as nn
from torchvision import models

class TransferResNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        #önceden eğitilmiş resnet indirme
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        #resnetin gövdesini dondur özellik çıkarma
        for param in self.resnet.parameters():
            param.requires_grad = False

        #son katmana bizim 3 sınıfı ekle
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)