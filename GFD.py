"""
📘 GEOLOGICAL FEATURES DETECTOR - GFD

Este script é uma pipeline completa para CLASSIFICAÇÃO DE IMAGENS SÍSMICAS 2D, 
identificando regiões que contenham estruturas geológicas como *falhas* (faults) e 
*domos de sal* (salts) a partir de uma imagem sísmica de entrada. 
A arquitetura central é baseada em uma **rede neural convolucional (CNN)** personalizada, 
chamada `CNNSeismicClassifierV3`, que é treinada para classificar pequenos blocos da imagem.

➡️ Principais componentes:

1. 🔧 CLASSES:
   - `SeismicBlock`: bloco convolucional básico com dilatação opcional, 
   ativação LeakyReLU e batch normalization.
   - `SpatialAttention`: mecanismo de atenção espacial que foca em
     regiões relevantes da imagem extraída pela CNN.
   - `CNNSeismicClassifierV3`: arquitetura principal da CNN composta por vários 
   `SeismicBlock`s + `SpatialAttention` + MLP para classificação.

2. 🧠 FUNÇÕES:
   - `sliding_window_classification`: recorta *janelas móveis* da imagem, 
   aplica a CNN e calcula *probabilidades* de cada classe para cada pixel.
   - `run_full_inference`: função principal que carrega a imagem, 
   aplica a `sliding_window_classification`, processa os *heatmaps* e salva os resultados.

3. 📷 USO DO OPENCV:
   - Leitura e conversão de imagens (`cv2.imread`, `cv2.cvtColor`);
   - Redimensionamento de patches (`cv2.resize`);
   - Extração de contornos (`cv2.findContours`) para desenhar bounding boxes
     ao redor de regiões classificadas;
   - Escrita de imagens processadas (`cv2.imwrite`).

➡️ RELAÇÃO ENTRE COMPONENTES:
- `run_full_inference` chama `sliding_window_classification`, 
que recebe como entrada uma imagem carregada com `OpenCV` e um modelo `CNNSeismicClassifierV3`.
- `sliding_window_classification` chama `model.forward` para obter probabilidades pixel a pixel.
- Os resultados (heatmaps) são processados e salvos em disco com OpenCV e Matplotlib.

------------------------------------------------------------
🔽 INÍCIO DO CÓDIGO
------------------------------------------------------------
"""

import os
import cv2                      # 📷 Manipulação de imagem (leitura, redimensionamento, contornos, escrita)
import numpy as np              # 🧮 Operações matriciais
import torch                    # 🔥 Framework de deep learning
import torch.nn.functional as F # 🔧 Funções auxiliares como softmax
import matplotlib.pyplot as plt # 📊 Geração de gráficos (heatmaps)
from torchvision import transforms
from torch import nn            # 🧱 Módulos da arquitetura de rede neural
import time

# Início da contagem
start_time = time.time()

# ------------------------------------------------------------
# 🔍 Bloco convolucional customizado (usado como encoder)
# ------------------------------------------------------------
class SeismicBlock(nn.Module):
    """Bloco básico de convolução para processamento sísmico
    - Combina: Conv2D + BatchNorm + LeakyReLU
    - Parâmetros dilatáveis para capturar features em diferentes escalas"""
    def __init__(self, in_channels, out_channels, dilation=1, kernel_size=3):
        super(SeismicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              padding=dilation, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        # Aplica convolução → normalização → ativação
        return self.act(self.bn(self.conv(x)))

# ------------------------------------------------------------
# 🔍 Módulo de atenção espacial
# Realça regiões com maior resposta (media + máximo) nos canais
# ------------------------------------------------------------
class SpatialAttention(nn.Module):
    """Mecanismo de Atenção Espacial
    - Foca nas regiões mais relevantes da imagem sísmica
    - Combina avg-pooling e max-pooling para gerar máscara de atenção"""
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn  # Multiplica a atenção pelo input original

# ------------------------------------------------------------
# 🧠 Arquitetura principal da CNN para classificação sísmica
# ------------------------------------------------------------
class CNNSeismicClassifierV4(nn.Module):
    """Arquitetura principal da CNN para classificação sísmica
    - Encoder: Extração de features hierárquicas
    - Classificador: MLP para decisão final"""
    def __init__(self, num_classes=3):
        super(CNNSeismicClassifierV4, self).__init__()
        self.encoder = nn.Sequential(
            SeismicBlock(1, 32),                # Entrada: 1 canal (imagem cinza), saída: 32
            nn.MaxPool2d(2),                    # Downsampling
            SeismicBlock(32, 64),
            nn.MaxPool2d(2),
            SeismicBlock(64, 128, dilation=2),  # Dilatação aumenta o campo receptivo
            nn.MaxPool2d(2),
            SeismicBlock(128, 256, dilation=2),
            SpatialAttention(),                 # Foco espacial
            nn.MaxPool2d(2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Reduz a saída para (batch, canais, 1, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),                       # Remove dimensões extras
            nn.Dropout(0.4),                    # Regularização
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)         # Saída: 3 classes (fundo, falha, sal)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

# ------------------------------------------------------------
# 🪟 Sliding Window: Classificação pixel a pixel via janelas
# ------------------------------------------------------------
def sliding_window_classification(img, model, device, window_sizes=[4, 8, 16, 32, 64], stride=4):
    """Processa a imagem sísmica em janelas deslizantes
    - Gera heatmaps de probabilidade para falhas e domos de sal
    - Combina resultados de múltiplas escalas (window_sizes)"""
    h, w = img.shape
    heatmaps = {
        'fault': np.zeros((h, w), dtype=np.float32),  # Mapa de probabilidades para falha
        'salt': np.zeros((h, w), dtype=np.float32),   # Mapa de probabilidades para sal
        'count': np.zeros((h, w), dtype=np.int32)     # Contador de sobreposições
    }

    norm_img = img.astype(np.float32) / 255.0  # Normaliza entre 0-1

    for win_size in window_sizes:
        for y in range(0, h - win_size + 1, stride):
            for x in range(0, w - win_size + 1, stride):
                patch = norm_img[y:y+win_size, x:x+win_size]
                patch_resized = cv2.resize(patch, (64, 64))  # 🔄 Redimensiona para entrada do modelo
                patch_tensor = torch.tensor(patch_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

                with torch.no_grad():
                    out = model(patch_tensor)           # ↩️ Forward pass no modelo
                    probs = F.softmax(out, dim=1)       # 📈 Converte logits em probabilidades
                    probs_np = probs[0].cpu().numpy()
                    fault_prob, salt_prob = probs_np[1], probs_np[2]

                # Atualiza os heatmaps com as probabilidades previstas
                heatmaps['fault'][y:y+win_size, x:x+win_size] += fault_prob
                heatmaps['salt'][y:y+win_size, x:x+win_size] += salt_prob
                heatmaps['count'][y:y+win_size, x:x+win_size] += 1

    # Evita divisão por zero
    count_safe = heatmaps['count'].copy()
    count_safe[count_safe == 0] = 1

    fault_avg = heatmaps['fault'] / count_safe
    salt_avg = heatmaps['salt'] / count_safe

    return fault_avg, salt_avg  # 🔁 Retorna os heatmaps médios por classe

# ------------------------------------------------------------
# 🏁 Função principal: pipeline completa
# ------------------------------------------------------------
def run_full_inference(image_path, model_path, save_dir, threshold=90):
    """Pipeline completo de inferência:
    1. Carrega imagem e modelo
    2. Gera heatmaps
    3. Pós-processa resultados
    4. Salva visualizações"""
    os.makedirs(save_dir, exist_ok=True)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 🖼️ Lê imagem sísmica
    h, w = img.shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carrega o modelo
    model = CNNSeismicClassifierV4(num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Executa a inferência por janela deslizante
    fault_map, salt_map = sliding_window_classification(img, model, device)

    # Gera máscaras binárias com limiar de decisão
    #mask_fault = (fault_map > threshold).astype(np.uint8) * 255
    #mask_salt = (salt_map > threshold).astype(np.uint8) * 255

    # Calcula os thresholds dinâmicos para top 25% dos valores
    top_fault = np.percentile(fault_map, threshold)
    top_salt = np.percentile(salt_map, threshold)

    # Cria máscaras baseadas nesses valores percentuais
    mask_fault = (fault_map >= top_fault).astype(np.uint8) * 255
    mask_salt = (salt_map >= top_salt).astype(np.uint8) * 255


    # Gera imagem base colorida (cinza → BGR)
    base_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Desenha contornos de domos de sal (verde)
    result_salt = base_img.copy()
    contours_salt, _ = cv2.findContours(mask_salt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_salt:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 200:  # Filtra regiões muito pequenas
            cv2.rectangle(result_salt, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Desenha contornos de falhas (vermelho)
    result_fault = base_img.copy()
    contours_fault, _ = cv2.findContours(mask_fault, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_fault:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 200:
            cv2.rectangle(result_fault, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Salva heatmaps com matplotlib
    plt.figure(figsize=(10, 4))
    plt.imshow(fault_map, cmap='hot', vmin=0, vmax=1)
    plt.colorbar(label='Probabilidade de Falha')
    plt.title('Heatmap - Falha')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/heatmap_fault_colored.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.imshow(salt_map, cmap='Greens', vmin=0, vmax=1)
    plt.colorbar(label='Probabilidade de Sal')
    plt.title('Heatmap - Sal')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/heatmap_salt_colored.png", dpi=300)
    plt.close()

    # Salva resultados com OpenCV
    cv2.imwrite(f"{save_dir}/mask_fault.png", mask_fault)
    cv2.imwrite(f"{save_dir}/mask_salt.png", mask_salt)
    cv2.imwrite(f"{save_dir}/classified_salt_result.png", result_salt)
    cv2.imwrite(f"{save_dir}/classified_fault_result.png", result_fault)

    print("✅ Resultados salvos em:", save_dir)

    # -----------------------------------------
    # 🔥 Sobreposição do heatmap de falha
    # -----------------------------------------
    plt.figure(figsize=(10, 4))
    plt.imshow(img, cmap='gray', interpolation='none')
    plt.imshow(fault_map, cmap='hot', alpha=0.5, vmin=0, vmax=1)
    plt.colorbar(label='Probabilidade de Falha')
    plt.title('Imagem Original com Heatmap de Falha')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/overlay_fault.png", dpi=300)
    plt.close()

    # -----------------------------------------
    # 🟢 Sobreposição do heatmap de sal
    # -----------------------------------------
    plt.figure(figsize=(10, 4))
    plt.imshow(img, cmap='gray', interpolation='none')
    plt.imshow(salt_map, cmap='Greens', alpha=0.5, vmin=0, vmax=1)
    plt.colorbar(label='Probabilidade de Sal')
    plt.title('Imagem Original com Heatmap de Sal')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/overlay_salt.png", dpi=300)
    plt.close()

# ------------------------------------------------------------
# ▶️ EXECUTA A INFERÊNCIA COM IMAGEM E MODELO ESPECÍFICO
# ------------------------------------------------------------
run_full_inference(
    image_path='seismic_2D/2D_002_3.png',
    model_path='cnn_seismic_model_06_05.pth',
    save_dir='GFD_results/window'
)

# Fim da contagem
end_time = time.time()

# Cálculo do tempo decorrido
elapsed_time = end_time - start_time

# Mostra no terminal
print(f"Tempo de execução do GFD: {elapsed_time:.4f} segundos")