import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_dataloader(data_dir, batch_size=32):
    #resnetin zorunlu kıldığı ImageNet renk standartları
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

    #train veri arttırma ve normalize
    train_transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    #test boyutlandırma ve normalize
    test_transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        normalize
    ])

    dataset_for_train = datasets.ImageFolder(root=data_dir,transform=train_transform)
    dataset_for_test = datasets.ImageFolder(root=data_dir, transform=test_transform)

    dataset_size = len(dataset_for_train)
    train_size = int(0.8 * dataset_size)

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(dataset_size, generator=generator).tolist()

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    #verileri ayırmak için
    train_dataset = Subset(dataset_for_train, train_indices)
    test_dataset = Subset(dataset_for_test, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, dataset_for_train.classes, dataset_for_train.class_to_idx