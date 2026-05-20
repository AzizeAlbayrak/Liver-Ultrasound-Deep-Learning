from src.dataset import get_dataloaders

data_dir ="data"

dataloder,dataset=get_dataloaders(data_dir)

from torch.utils.data import random_split, DataLoader
train_size = int(0.8*len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset,batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Sınıflar:" , dataset.classes)

#nasıl sayısala çevirmiş mapping
print("\n",dataset.class_to_idx)

for image,labels in train_loader:
    print("\nImage shape:" , image.shape)
    print("Labels:", labels)
    break

#model import etme
from src.model import SimpleCNN
model = SimpleCNN(input_channels=3, num_classes=3)

#loss ve optimizer
import torch
import torch.nn as nn
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

#training loop
correct = 0
total = 0

epochs = 10
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")

    running_loss = 0
    correct = 0
    total = 0 #her epoch da reset

    for images, labels in train_loader:

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #LOSS
        running_loss += loss.item()

        #Accury
        _, preds = torch.max(outputs, 1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(dataloder)
    accuracy = correct / total

    print("Loss:", epoch_loss)
    print("Accuracy:", accuracy)

#TEST
model.eval()

test_accuracy = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        test_total += labels.size(0)
        test_accuracy += (preds == labels).sum().item()

test_acc = test_accuracy / test_total
print("Test Accuracy:", test_acc)

model.train()

from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import torch

#Tahmin toplama
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

#confisuan matris
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:\n", cm)

print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=dataset.classes))

