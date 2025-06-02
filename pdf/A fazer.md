############################### QUESTIONAMENTOS: ###################################

### 0) Como os parâmetros do OpenCV criam aquele mapa de textura?
### 1) Com imagens de atributos sísmicos seria melhor para o OpenCV identificar padrões?
### 2) Tem como realizar um processo de atributos sísmicos no OpenCV? 
### 3) links:
###         https://medium.com/data-analysis-center/seismic-fault-detection-ef78c1f7307c
###         https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0271615
### 4)Como conseguir bons resultados para outra seção sísmicas diferentes?
###         4.1) Achar o mapa de textura usando outra CNN + OpenCV iterativo:
###             4.1.1) Criar um mapa de textura modelo para uma seção sísmica x, com valores específicos de cv.arguments;
###             4.1.2) Criar uma CNN para entender os padrões do mapa de textura da seção sísmica x;
###             4.1.3) Aplicar uma iteração com valores distintos dos cv.arguments, para uma seção sísmica y;
###             4.1.4) Usar a CNN para escolher os cv.arguments que reproduzem as caracteristicas de sismica x na sísmica y;
###         4.2) Achar o mapa de textura usando deep leanring generativo:
###             4.2.1) Criar um mapa de textura modelo para uma seção sísmica x, com valores específicos de cv.arguments;
###             4.2.2) Criar uma rede_a generativa para gerar um mapa de textura com valores de cv.arguments, usando uma noma imagem sísmica y;
###             4.2.3) Usar outra rede_b para avaliar o quão bom é o resultado da rede_a comparando as sísmicas x e y;
###             4.2.4) Fazer os passos 4.2.1 até 4.2.3 iterativamente até que os filtros aplicados na sísmica y tiver o mesmo padrão de textura da imagem sísmica x.



# A fazer:

### 0) adicionar esse projeto no GitHub adicioando um Pipeline [FEITO];
### 1) melhorar as descrições dos códigos [FEITO];
### 2) baixar imagens do link [FEITO]: 
###    https://www.kaggle.com/code/prateekvyas/seismic-classification-using-deep-learning/;
### 3) extrair o "fundo" das imagens cotendo [FEITO]:
###    2.0) Partes retas e contínuas, laminadas da sísmica;
###    2.1) Onde não há falhas nem domos;
### 4) montar as pastas do banco de dados (train, test, validation) contendo as classes [NÃO_NECESSÁRIO];
### 5) montar o código de treinamento (para sísmica 2D segmentada em feições) [FEITO];
### 6) "encaixar" o código de treinamento nesse código de identificação [FEITO];
### 7) melhorar o banco de dados [FEITO_PARCIALMENTE]:
###    7.0) imagens com as feições interna da duna;
###    7.1) imagens com "zoons" das falhas;
###    7.2) imagens "bg" variados como: lámina d'água, interfaçe do fundo do mar, entre outras feições...
### 8) pesquisar como deixar o modelo mais complexo[FEITO_PARCIALMENTE]...
### 9) criar arquivo model.py para hospedar o modelo [FEITO];
### 10) melhorar a captura de padrões do OpenCV [FEITO_PARCIALMENTE];
###    Obs: uma evolução muito grande na captação de feiçoes pelo OpenCV no dia 08/05/25
###    10.1) pesquisar os parametros de: 
            Canny, 
            getStructuringElement, 
            orphologyEx, 
            findContours, 
            getGaborKernel, 
            addWeighted,
            cvtColor,
            graycoprops
### 11) Terminar o pre_processing;
### 12) criar um main;

### Obs: [FEITO_PARCIALMENTE] = pode melhorar!

################################### LINKEDIN: ###################################

### postagem falar sobre as limitações de hardware, resultados não estão perfeitos, busca da melhora...


#################### Alguns bons resultados antigos + códigos ###################

### resultado para 002:
#edges = cv2.Canny(imagem_blur, 420, 540)  #(400, 500) # (420, 520) # (430, 530) #00

## Fecha bordas próximas e remove ruído
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 3)) #(4, 3) #(5, 4)
#edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=7) #5 #6 #7

#contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#imagem_resultado = cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)

### Registro de um bom resultado para o modelo V3:

# ------------------------------------------------------------
# 🧩 Capítulo 3: Carregamento da imagem sísmica
# ------------------------------------------------------------
input_path = 'seismic_2D/2D_002_3.png'
imagem = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
imagem_eq = cv2.equalizeHist(imagem)
imagem_blur = cv2.GaussianBlur(imagem_eq, (5,5), 0)


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
    kernel = cv2.getGaborKernel((6, 6), .8, theta, 1.5, 0.22, ktype=cv2.CV_32F)
    return cv2.filter2D(img, cv2.CV_8UC3, kernel)

gabor_0 = apply_gabor(imagem_blur, 0)
gabor_90 = apply_gabor(imagem_blur, np.pi/2)

# Geração do mapa de textura mais suave
texture_map = cv2.addWeighted(laplacian, 0.6, sobel_combined, 0.6, 0) # 0.6,0.6
texture_map = cv2.addWeighted(texture_map, 0.65, gabor_0, 0.4, 0) # 0.6,0.4
#texture_map = cv2.addWeighted(texture_map, 0.72, gabor_90, 0.3, 0) # 0.7,0.3

# Detecção de bordas
edges = cv2.Canny(texture_map, 200, 250) #200, 275
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 5)) #5,5
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=4)

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
plt.figure(figsize=(14,8))
plt.imshow(texture_map, cmap='gray')
plt.title("Mapa de Textura")
plt.axis("off")
plt.savefig('2D_GFI_results/texture_results.png', dpi=300, bbox_inches='tight')


"Domos de sal tendem a estar associados a zonas de falhamento na sobrecarga superior ou em áreas marginais — logo, padrões verticais de alta curvatura seguidos por fraturas no topo são indícios úteis para segmentação automática."