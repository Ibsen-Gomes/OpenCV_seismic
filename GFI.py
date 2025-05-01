# GFI.py – Geological Feature Identifier com OpenCV + CNN

# ------------------------------------------------------------
# 🧩 Capítulo 1: Importações
# ------------------------------------------------------------
import cv2
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model import CNNSeismicClassifier

# ------------------------------------------------------------
# 🧩 Capítulo 2: Carregamento do modelo treinado
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNSeismicClassifier().to(device)
model.load_state_dict(torch.load("cnn_seismic_model.pth", map_location=device))
model.eval()
print("✅ Modelo carregado com sucesso!")

# ------------------------------------------------------------
# 🧩 Capítulo 3: Carregamento da imagem sísmica
# ------------------------------------------------------------
input_path = 'seismic_2D/2D_008.png'
imagem = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
imagem_eq = cv2.equalizeHist(imagem)
imagem_blur = cv2.GaussianBlur(imagem_eq, (5,5), 0)

# ------------------------------------------------------------
# 🧩 Capítulo 4: Segmentação com OpenCV
# ------------------------------------------------------------
edges = cv2.Canny(imagem_blur, 50, 150)
contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
imagem_resultado = cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)

# ------------------------------------------------------------
# 🧩 Capítulo 5: Classificação com CNN
# ------------------------------------------------------------
for cnt in contornos:
    x, y, w, h = cv2.boundingRect(cnt)
    if w * h < 500:
        continue

    patch = imagem[y:y+h, x:x+w]
    patch_resized = cv2.resize(patch, (64, 64))
    patch_normalized = patch_resized / 255.0
    patch_tensor = torch.tensor(patch_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    output = model(patch_tensor)
    pred_label = torch.argmax(output, dim=1).item()

    if pred_label == 1:
        cv2.rectangle(imagem_resultado, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Falha
    elif pred_label == 2:
        cv2.rectangle(imagem_resultado, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Sal
    # Fundo = 0 → ignora

# ------------------------------------------------------------
# 🧩 Capítulo 6: Visualização e salvamento
# ------------------------------------------------------------
plt.figure(figsize=(14,8))
plt.imshow(cv2.cvtColor(imagem_resultado, cv2.COLOR_BGR2RGB))
plt.title('Detecção automática: Verde=Sal, Vermelho=Falhas')
plt.axis('off')
plt.show()

# Salvar
os.makedirs("2D_GFI_results", exist_ok=True)
nome_saida = os.path.basename(input_path).replace('.png', '_GFI.png')
caminho_saida = os.path.join("2D_GFI_results", nome_saida)
cv2.imwrite(caminho_saida, imagem_resultado)
print(f"✅ Resultado salvo em: {caminho_saida}")
