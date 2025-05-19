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
from model import CNNSeismicClassifierV3

# ------------------------------------------------------------
# 🧩 Capítulo 2: Carregamento do modelo treinado
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNSeismicClassifierV3(num_classes=3).to(device)

model.load_state_dict(torch.load("cnn_seismic_model.pth", map_location=device))
model.eval()
print("✅ Modelo carregado com sucesso!")

# ------------------------------------------------------------
# 🧩 Capítulo 3: Carregamento da imagem sísmica
# ------------------------------------------------------------
input_path = 'seismic_2D/2D_002_2.png'
imagem = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
imagem_eq = cv2.equalizeHist(imagem)
imagem_blur = cv2.GaussianBlur(imagem_eq, (5,5), 0)

# ------------------------------------------------------------
# 🧩 Capítulo 4: Segmentação com OpenCV
# ------------------------------------------------------------
#edges = cv2.Canny(imagem_blur, 300, 450)
#contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#imagem_resultado = cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)

# ------------------------------------------------------------
# 🧩 Capítulo 4: Segmentação com OpenCV (ajustada)
# ------------------------------------------------------------
#edges = cv2.Canny(imagem_blur, 420, 540)  #(400, 500) 

# Fecha bordas próximas e remove ruído
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 5)) #(4, 3) 
#edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=5) #5

#contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#imagem_resultado = cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)

# ------------------------------------------------------------
# 🧩 Capítulo 4: Segmentação com OpenCV + Análise de Textura
# ------------------------------------------------------------
from skimage.feature import graycomatrix, graycoprops
from skimage.segmentation import slic, mark_boundaries

# 🔹 Filtros de textura
laplacian = cv2.Laplacian(imagem_blur, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

sobelx = cv2.Sobel(imagem_blur, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(imagem_blur, cv2.CV_64F, 0, 1, ksize=5)
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.convertScaleAbs(sobely)
sobel_combined = cv2.addWeighted(sobelx, 5, sobely, 5, 0)

# 🔹 Gabor filters
def apply_gabor(img, theta):
    kernel = cv2.getGaborKernel((6, 6), .85, theta, 1.5, 0.23, ktype=cv2.CV_32F)
    return cv2.filter2D(img, cv2.CV_8UC3, kernel)

gabor_0 = apply_gabor(imagem_blur, 0)
gabor_90 = apply_gabor(imagem_blur, np.pi/2)

# Geração do mapa de textura mais suave
texture_map = cv2.addWeighted(laplacian, 0.6, sobel_combined, 0.6, 0) # 0.6,0.6
texture_map = cv2.addWeighted(texture_map, 0.6, gabor_0, 0.4, 0) # 0.6,0.4
texture_map = cv2.addWeighted(texture_map, 0.72, gabor_90, 0.3, 0) # 0.7,0.3

# Detecção de bordas
edges = cv2.Canny(texture_map, 250, 280) #200, 275
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 5)) #5,5
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=5)

# Extração de contornos
contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
imagem_resultado = cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)

# GLCM local para identificar regiões homogêneas
mapa_homogeneidade = np.zeros(imagem.shape)
step = 32
for y in range(0, imagem.shape[0] - step, step):
    for x in range(0, imagem.shape[1] - step, step):
        bloco = imagem_blur[y:y+step, x:x+step]
        glcm = graycomatrix(bloco, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
        hom = graycoprops(glcm, 'homogeneity')[0, 0]
        if hom > 0.92:
            mapa_homogeneidade[y:y+step, x:x+step] = 255

# Visualizar
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(texture_map, cmap='gray')
plt.title("Mapa de Textura")
plt.axis("off")


# ------------------------------------------------------------
# 🧩 Capítulo 5: Classificação com CNN e visualização
# ------------------------------------------------------------
limiar_confiança = 0.9  # Mostra apenas predições com 85% ou mais de confiança

# Dicionários de rótulo e cor: só mostramos Falha (1) e Sal (2)
rotulos = {1: "Falha", 2: "Sal"}
cores = {1: (0, 0, 255), 2: (0, 255, 0)}  # Vermelho: falha, Verde: sal

for c in contornos:
    x, y, w, h = cv2.boundingRect(c)
    if w < 20 or h < 20:  # Ignora regiões muito pequenas
        continue

    # Recorte e pré-processamento
    recorte = imagem[y:y+h, x:x+w]
    recorte_resized = cv2.resize(recorte, (64, 64))
    entrada_tensor = torch.tensor(recorte_resized / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # Predição
    with torch.no_grad():
        saida = model(entrada_tensor)
        probs = F.softmax(saida, dim=1)
        classe_predita = torch.argmax(probs, dim=1).item()
        confianca = probs[0, classe_predita].item()

    # Exibe apenas se a classe for falha ou sal E confiança >= limiar
    if classe_predita in rotulos and confianca >= limiar_confiança:
        label = rotulos[classe_predita]
        cor = cores[classe_predita]
        texto = f"{label}: {confianca*100:.1f}%"
        cv2.rectangle(imagem_resultado, (x, y), (x + w, y + h), cor, 2)
        cv2.putText(imagem_resultado, texto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1)


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


#################### ori ######################