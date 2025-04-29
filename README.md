# 🧠 OpenCV_seismic — Geological Feature Identifier (GFI)

Projeto de identificação automática de feições geológicas em seções sísmicas 2D, combinando visão computacional com inteligência artificial. Utiliza **OpenCV** para segmentar imagens sísmicas e uma **CNN treinada com PyTorch** para classificar regiões como **falhas**, **domos de sal** ou **fundo**.

---

## 📌 Visão geral do pipeline

### ✅ 1. Entrada:
- Carrega uma **imagem de seção sísmica 2D completa** (formato `.jpg`, `.png`, etc.).

### ✅ 2. Pré-processamento com OpenCV:
- OpenCV detecta padrões visuais típicos:
  - **Bordas abruptas** (possíveis falhas),
  - **Texturas caóticas** (possíveis domos de sal),
  - **Texturas suaves/listradas** (fundo).
- A imagem não é rotulada nessa etapa — o OpenCV apenas **propõe regiões candidatas** com base em contornos e gradientes.

### ✅ 3. Classificação com CNN:
- Cada região candidata (patch) é passada para uma **rede neural convolucional (CNN)** já treinada com dados segmentados.
- A CNN classifica:
  - **Falha** ➝ retângulo vermelho
  - **Sal** ➝ retângulo verde
  - **Fundo** ➝ ignorado

### ✅ 4. Resultado:
- A imagem original retorna com **as feições identificadas e marcadas** automaticamente.
- Ideal para automatizar tarefas repetitivas de interpretação sísmica.

---

## 🧪 Exemplo de aplicação

```bash
python GFI.py