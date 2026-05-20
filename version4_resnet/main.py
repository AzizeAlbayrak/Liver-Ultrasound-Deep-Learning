import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
from src.model import TransferResNet
from src.dataset import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

data_dir = "../data"

train_loader, test_loader, classes, class_to_idx = get_dataloaders(data_dir, batch_size=32)

print("Sınıflar:", classes)

model = TransferResNet(num_classes=len(classes)).to(device)

#loss
weights = torch.tensor([1.5, 1.0, 1.5], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

#training loss
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    print(f"Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")

print("\n--- Eğitim Tamamlandı. Test Aşamasına Geçiliyor ---\n")

#test
model.eval()

all_preds = []
all_labels = []
test_correct = 0
test_total = 0

with torch.no_grad():
    for test_images, test_labels in test_loader:
        test_images = test_images.to(device)

        outputs = model(test_images)
        _, preds = torch.max(outputs, 1)

        test_total += test_labels.size(0)
        test_correct += (preds.cpu() == test_labels).sum().item()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(test_labels.numpy())

test_acc = test_correct / test_total
print(f"Test Accuracy: {test_acc:.4f}\n")

#metrikler
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:\n", cm)

print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=classes))