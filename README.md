# 🧠 OpenCV_seismic — Geological Feature Identifier (GFI)

Projeto para **identificação automática de feições geológicas** em seções sísmicas 2D, combinando **visão computacional (OpenCV)** e **redes neurais convolucionais (CNN com PyTorch)**.

O sistema detecta automaticamente **falhas**, **dobras (anticlinais/sinclinais)** e **regiões de fundo**, usando técnicas modernas de pré-processamento e aprendizado profundo.

---

## 🧬 Estrutura Modular

- `model.py` → Define a arquitetura da CNN.
- `train.py` → Realiza o treinamento com patches rotulados (salt, fault, fold, background).
- `GFI.py` → Identifica feições automaticamente em imagens sísmicas completas.
- `utils.py` → Plota e salva gráficos de desempenho.

---

## 🧩 Pipeline de funcionamento

### ✅ 1. Treinamento da CNN
- Utiliza imagens recortadas e rotuladas de diferentes feições geológicas (patches).
- A CNN é treinada para diferenciar **fundo**, **falha** e **dobra**.
- Aplica **aumento de dados** (data augmentation) e salva os gráficos em `2D_GFI_results/`.

### ✅ 2. Inferência automática com OpenCV + CNN
- OpenCV identifica **regiões candidatas** com base em bordas e contornos (Canny + `findContours`).
- Cada patch extraído é classificado pela CNN:
  - **Falha** → retângulo vermelho
  - **Dobra** → retângulo azul (opcional, ajustar cor)
  - **Sal/Domo** (se adicionado) → retângulo verde
  - **Fundo** → descartado
- Resultado final é salvo e exibido com marcações visuais.

---

## 📂 Estrutura de pastas

OpenCV_seismic/                ← Pasta raiz do projeto
├── train.py                   ← Código para treinar a CNN
├── GFI.py                     ← Código para detectar feições automaticamente
├── model.py                   ← Arquitetura da rede neural CNN
├── utils.py                   ← Funções auxiliares (gráficos de loss e accuracy)
├── cnn_seismic_model.pth      ← Arquivo salvo com os pesos da CNN treinada
├── database/                  ← Base de dados segmentada por classe
│   ├── background/            ← Patches rotulados de fundo
│   ├── fault/                 ← Patches rotulados de falhas
│   └── fold/                  ← Patches rotulados de dobras
├── seismic_2D/                ← Imagens de seção sísmica 2D completas
│   └── 2D_008.png             ← Exemplos usados no GFI.py
└── 2D_GFI_results/            ← Resultados gerados pelo GFI.py e gráficos do treino
    ├── 2D_008_GFI.png         ← Imagem final com feições marcadas
    ├── loss_curve.png         ← Curva de perda
    └── accuracy_curve.png     ← Curva de acurácia
