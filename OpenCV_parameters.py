# Reexecutar após reinicialização do estado

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Criar diretório de saída
output_dir = "2D_GFI_results/opencv_experimentos"
os.makedirs(output_dir, exist_ok=True)

# Carrega a imagem sísmica em escala de cinza
img_path = "seismic_2D/2D_002_3.png"
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
image_blur = cv2.GaussianBlur(image, (5, 5), 0)

# Parâmetros a testar
canny_params = [(50, 100), (100, 150), (100, 200), (150,200), (150, 250), (200, 250), (250, 350)]
kernel_sizes = [(1,1), (2, 2), (3, 3),(4, 4), (5, 5), (6, 6), (7, 7), (6, 5), (7,5)]
gabor_thetas = [0, np.pi/4, np.pi/2]
sobel_ksizes = [3, 5, 7]

# Função auxiliar para Gabor
def apply_gabor(img, theta):
    kernel = cv2.getGaborKernel((21, 21), 4.0, theta, 10.0, 0.5, ktype=cv2.CV_32F)
    return cv2.filter2D(img, cv2.CV_8UC3, kernel)

# Canny
for t1, t2 in canny_params:
    edges = cv2.Canny(image_blur, t1, t2)
    filename = f"canny_{t1}_{t2}.png"
    cv2.imwrite(os.path.join(output_dir, filename), edges)

# Morphology Close
for ksize in kernel_sizes:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    edges = cv2.Canny(image_blur, 100, 200)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    filename = f"morph_close_{ksize[0]}x{ksize[1]}.png"
    cv2.imwrite(os.path.join(output_dir, filename), closed)

# Gabor
for theta in gabor_thetas:
    gabor = apply_gabor(image_blur, theta)
    filename = f"gabor_theta_{int(theta * 100)}.png"
    cv2.imwrite(os.path.join(output_dir, filename), gabor)

# Sobel
for k in sobel_ksizes:
    sobelx = cv2.Sobel(image_blur, cv2.CV_64F, 1, 0, ksize=k)
    sobely = cv2.Sobel(image_blur, cv2.CV_64F, 0, 1, ksize=k)
    combined = cv2.addWeighted(cv2.convertScaleAbs(sobelx), 0.5,
                               cv2.convertScaleAbs(sobely), 0.5, 0)
    filename = f"sobel_k{k}.png"
    cv2.imwrite(os.path.join(output_dir, filename), combined)

output_dir
