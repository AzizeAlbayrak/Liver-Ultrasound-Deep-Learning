import torch
import transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, batch_size=32):

    #transform:görüntüye uygulanacak işlemler
    transform = transforms.Compose([
        transforms.Resize((128,128)), #tüm resimler aynı boyuta
        transforms.ToTensor(),        #resmi sayısala çevir
         ])


    #ImageFolder klasörü labele çevirir
    dataset = datasets.ImageFolder(root=data_dir,transform=transform)

    #DataLoader veriyi modele parça parça verir batch
    dataloder = DataLoader(dataset,batch_size=batch_size,shuffle=True)

    return dataloder, dataset


