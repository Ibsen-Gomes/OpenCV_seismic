# train.py — Treinamento da CNN para feições sísmicas

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from model import CNNSeismicClassifierV4
from utils import plot_loss_accuracy

import time

# Início da contagem
start_time = time.time()
# ------------------------------------------------------------
# 🧩 Capítulo 1: Transformações no dataset
# ------------------------------------------------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

# ------------------------------------------------------------
# 🧩 Capítulo 2: Carregamento do dataset
# ------------------------------------------------------------
dataset = datasets.ImageFolder(root='database', transform=transform)
print(f"🧾 Classes detectadas: {dataset.classes}")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ------------------------------------------------------------
# 🧩 Capítulo 3: Inicialização do modelo
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNSeismicClassifierV4(num_classes=len(dataset.classes)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)

# ------------------------------------------------------------
# 🧩 Capítulo 4: Loop de treinamento
# ------------------------------------------------------------
num_epochs = 300
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # 🧪 Validação
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100 * correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"📊 Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

# ------------------------------------------------------------
# 🧩 Capítulo 5: Salvamento e visualização
# ------------------------------------------------------------
torch.save(model.state_dict(), "cnn_seismic_model_30_05.pth")
print("✅ Modelo salvo como cnn_seismic_model_30_05.pth")

plot_loss_accuracy(train_losses, val_losses, train_accuracies, val_accuracies)

# Fim da contagem
end_time = time.time()

# Cálculo do tempo decorrido
elapsed_time = end_time - start_time

# Mostra no terminal
print(f"Tempo de execução do treinamento: {elapsed_time:.4f} segundos")