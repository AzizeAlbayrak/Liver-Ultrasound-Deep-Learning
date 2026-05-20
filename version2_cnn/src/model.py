import torch
import torch.nn as nn

class CNN(nn.Module):

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
#BatchNorm her katmandan çıkan veriyi normalize eder düzenler değerleri düzenler
#Dropout training sırasında bazı nöronları rastgele kapatır
#Data Augmentation resmi çevirir döndürür değiştirir ve artmasını sağlar
        self.features = nn.Sequential( #görüntüden özellik çıkarmak
            nn.Conv2d(input_channels, 16,3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential( #bu görüntü hangi sınıf?
            nn.Flatten(),
            nn.Linear(64*16*16,128), #CNN’den çıkan veriyi sınıflandırmaya hazırlar
            nn.ReLU(),     #özellikleri sıkıştırır / öğrenir
            nn.Dropout(0.3), #%50 nöron kapatılır
            nn.Linear(128,num_classes) #128 feature
        )
    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x
