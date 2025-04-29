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
# 🧩 Capítulo 2: Definição da CNN
# ------------------------------------------------------------
class CNNSeismicClassifier(nn.Module):
    def __init__(self):
        super(CNNSeismicClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # Corrigido: 8x8 após pooling
        self.fc2 = nn.Linear(128, 3)  # 3 classes: fundo, falha, sal

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(-1, 64 * 8 * 8)  # Corrigido: compatível com pooling
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ------------------------------------------------------------
# 🧩 Capítulo 3: Carregamento do modelo
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNSeismicClassifier().to(device)
model.eval()  # Coloca o modelo em modo avaliação

# ------------------------------------------------------------
# 🧩 Capítulo 4: Carregamento e pré-processamento da imagem sísmica
# ------------------------------------------------------------
imagem = cv2.imread('database/salt/salt_001.jpg', cv2.IMREAD_GRAYSCALE)

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


# A fazer:

### 0) adicionar esse projeto no GitHub adicioando um Pipeline [FEITO];
### 1) melhorar as descrições dos códigos [FEITO];
### 2) baixar imagens do link [FEITO]: 
###    https://www.kaggle.com/code/prateekvyas/seismic-classification-using-deep-learning/;
### 3) extrair o "fundo" das imagens cotendo [FEITO]:
###    2.1) Partes retas e contínuas, laminadas da sísmica
###    2.2) Onde não há falhas nem domos
### 4) montar as pastas do banco de dados (train, test, validation) contendo as classes fundo, sal e falha;
### 5) montar o código de treinamento (para sísmica 2D segmentada em feições);
### 6) "encaixar" o código de treinamento nesse código de identificação;
### 7) criar um main;

# Linkedin:

### postagem falar sobre as limitações de hardware, resultados não estão perfeitos, busca da melhora...