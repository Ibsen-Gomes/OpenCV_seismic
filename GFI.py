# GFI - Geological Feature Identifier

# ------------------------------------------------------------
# 🧩 Capítulo 1: Importação de bibliotecas
# ------------------------------------------------------------
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 🧩 Capítulo 2: Definição da CNN (versão compatível com modelo treinado)
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
# 🧩 Capítulo 3: Carregamento do modelo
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNSeismicClassifier().to(device)
model.eval()  # Coloca o modelo em modo avaliação

model.load_state_dict(torch.load("cnn_seismic_model.pth", map_location=device))
print("✅ Modelo carregado com sucesso!")

# ------------------------------------------------------------
# 🧩 Capítulo 4: Carregamento e pré-processamento da imagem sísmica
# ------------------------------------------------------------
imagem = cv2.imread('seismic_2D/2D_008.png', cv2.IMREAD_GRAYSCALE)

# Equaliza histograma para melhorar contraste
imagem_eq = cv2.equalizeHist(imagem)

# Aplica suavização para reduzir ruído
imagem_blur = cv2.GaussianBlur(imagem_eq, (5,5), 0)

# ------------------------------------------------------------
# 🧩 Capítulo 5: Proposição de candidatos usando OpenCV
# ------------------------------------------------------------
# Detecta bordas na imagem pré-processada
edges = cv2.Canny(imagem_blur, 50, 150)

# Encontra contornos a partir das bordas
contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Copia da imagem original para desenhar os resultados
imagem_resultado = cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)

# ------------------------------------------------------------
# 🧩 Capítulo 6: Classificação das regiões detectadas
# ------------------------------------------------------------
for cnt in contornos:
    x, y, w, h = cv2.boundingRect(cnt)
    
    # Filtro para descartar regiões muito pequenas
    if w * h < 500:
        continue

    # Recorta o patch da região candidata
    patch = imagem[y:y+h, x:x+w]

    # Redimensiona o patch para 64x64 (entrada padrão da CNN)
    patch_resized = cv2.resize(patch, (64, 64))
    patch_normalized = patch_resized / 255.0  # Normaliza para [0,1]
    patch_tensor = torch.tensor(patch_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # Classifica o patch usando a CNN
    output = model(patch_tensor)
    pred_label = torch.argmax(output, dim=1).item()

    # Desenha retângulos coloridos conforme a classe prevista
    if pred_label == 1:  # Falha
        cv2.rectangle(imagem_resultado, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Vermelho
    elif pred_label == 2:  # Sal
        cv2.rectangle(imagem_resultado, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Verde
    # Se for fundo (classe 0), não desenha nada

# ------------------------------------------------------------
# 🧩 Capítulo 7: Visualização do resultado final
# ------------------------------------------------------------
# Exibe a imagem final com os retângulos desenhados
plt.figure(figsize=(14,8))
plt.imshow(cv2.cvtColor(imagem_resultado, cv2.COLOR_BGR2RGB))
plt.title('Detecção automática: Verde=Sal, Vermelho=Falhas')
plt.axis('off')
plt.show()

# ------------------------------------------------------------
# 🧩 Capítulo 8: Salvamento do resultado final
# ------------------------------------------------------------
import os

# Garante que a pasta existe
os.makedirs("2D_GFI_results", exist_ok=True)

# Define nome do arquivo de saída baseado no nome da entrada
nome_saida = os.path.basename('seismic_2D/2D_008.png').replace('.png', '_GFI.png')

# Caminho completo de saída
caminho_saida = os.path.join("2D_GFI_results", nome_saida)

# Salva a imagem com os retângulos desenhados
cv2.imwrite(caminho_saida, imagem_resultado)

print(f"✅ Resultado salvo em: {caminho_saida}")