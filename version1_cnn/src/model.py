import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
#input channels kanal sayısı rgb 3 kanal
#num channels kaç sınıf tahmini
    def __init__(self, input_channels=3, num_classes=3):
        super().__init__()
#Conv özell,k çıkarır kenar doku vs
#in_channels RGB
#out_channels= öğrenilecek katman sayısı
#kernel_size küçük kutu resmin üstünden kayar her yeri tarar
#ReLU neg. değerleri 0 yapar gerisini bırakır, modeli nonlinear yapar
#maxPool görüntüyü küçültür ama önemli bilgiyi korur özet gibi düşünebilirsin
#padding görüntü etrafına 0 eklemek boyutu korumak için
#flutten resmi vektöre çevirir
#linear sınıflandırma için
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16,3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(    
            nn.Flatten(),
            nn.Linear(64*16*16,num_classes) #input 128 3 maxpool da 16
        )
    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x
