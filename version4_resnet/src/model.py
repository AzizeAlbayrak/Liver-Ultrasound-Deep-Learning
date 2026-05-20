#fine_tuning
import torch.nn as nn
from torchvision import models


class TransferResNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        #resneti indir
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # DİKKAT: V3'teki dondurma (requires_grad = False) kodunu sildik!
        # Artık tüm ağ tıbbi görüntüleri öğrenmek için serbest.

        #son katmanı (Classifier) söküp kendi 3 sınıflı katmanımızı takıyoruz
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)