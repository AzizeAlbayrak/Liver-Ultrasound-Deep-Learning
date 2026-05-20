import torch
from numpy.ma.core import indices
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_dataloader(data_dir,batch_size=32):
    #train için dönüşüm veri arttırma
    train_transforms = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(), #rastgele yatay çevirme
        transforms.ToTensor(),
    ])

    #test için dönüşümler sadece boyutlandırma veri arttırma yok
    test_transforms = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
    ])

    #veriyi iki farklı transformla yükleme
    dataset_for_train = datasets.ImageFolder(root=data_dir,transform=train_transforms)
    dataset_for_test = datasets.ImageFolder(root=data_dir,transform=test_transforms)

    #veri ayrımı için rastgele index oluşturma
    dataset_size = len(dataset_for_train)
    train_size = int(0.8*dataset_size)

    generator = torch.Generator().manual_seed(42) #sabit rastgelelik
    indices = torch.randperm(dataset_size,generator=generator).tolist()


    #verileri ayırıyoruz
    train_dataset = Subset(dataset_for_train,indices)
    test_dataset = Subset(dataset_for_test,indices)

    #loaderları oluşturuyoruz
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

    return train_loader,test_loader,dataset_for_train.classes,dataset_for_test.class_to_idx
