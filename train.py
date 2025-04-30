# train.py – CNN aprimorada para classificação de feições sísmicas: falha, dobra e fundo

# ------------------------------------------------------------
# 🧩 Capítulo 1: Importação de bibliotecas
# ------------------------------------------------------------
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# 🧩 Capítulo 2: Definição da CNN mais robusta
# ------------------------------------------------------------
class CNNSeismicClassifier(nn.Module):
    def __init__(self):
        super(CNNSeismicClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 3)  # 3 classes: background, fault, fold

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 8 * 8)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ------------------------------------------------------------
# 🧩 Capítulo 3: Configuração do dataset + Data Augmentation
# ------------------------------------------------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root='database', transform=transform)

# Verifica nomes das classes
print(f"🧾 Classes detectadas: {dataset.classes}")

# Split treino/validação
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# ------------------------------------------------------------
# 🧩 Capítulo 4: Treinamento com acurácia
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNSeismicClassifier().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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

    # Validação
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
# 🧩 Capítulo 5: Salvamento do modelo treinado
# ------------------------------------------------------------
torch.save(model.state_dict(), "cnn_seismic_model.pth")
print("✅ Modelo salvo como cnn_seismic_model.pth")

# ------------------------------------------------------------
# 🧩 Capítulo 6: Gráficos salvos em 2D_GFI_results
# ------------------------------------------------------------
os.makedirs("2D_GFI_results", exist_ok=True)

# 🎨 Estilo bonito com seaborn
sns.set(style="whitegrid")

# 📈 Loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss', linewidth=2, marker='o', color='steelblue')
plt.plot(val_losses, label='Val Loss', linewidth=2, marker='s', color='darkorange')
plt.title('Training and Validation Loss', fontsize=16)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig("2D_GFI_results/loss_curve.png", dpi=300)
print("📥 loss_curve.png salvo em 2D_GFI_results/")
plt.show()

# 📊 Accuracy
plt.figure(figsize=(10, 6))
plt.plot(train_accuracies, label='Train Accuracy', linewidth=2, marker='o', color='green')
plt.plot(val_accuracies, label='Val Accuracy', linewidth=2, marker='s', color='red')
plt.title('Training and Validation Accuracy', fontsize=16)
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.tight_layout()
plt.savefig("2D_GFI_results/accuracy_curve.png", dpi=300)
print("📥 accuracy_curve.png salvo em 2D_GFI_results/")
plt.show()

