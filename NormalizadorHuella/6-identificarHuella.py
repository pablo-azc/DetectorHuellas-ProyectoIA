import os
import sys
import argparse
import random
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ==============================================================================
# 1. ETAPAS DE PROCESAMIENTO (PASOS 2 A 4)
# ==============================================================================

def simular_captura(img, angulo_max=45.0, pad_porcentaje=0.35):
    """
    Paso 2: Añade borde blanco y aplica una rotación aleatoria
    simulando la captura en un escáner.
    """
    h, w = img.shape[:2]
    pad_h = int(h * pad_porcentaje)
    pad_w = int(w * pad_porcentaje)

    img_padded = cv2.copyMakeBorder(
        img,
        top=pad_h, bottom=pad_h, left=pad_w, right=pad_w,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255]
    )

    angulo = random.uniform(-angulo_max, angulo_max)
    hp, wp = img_padded.shape[:2]
    centro = (wp // 2, hp // 2)
    matriz_rotacion = cv2.getRotationMatrix2D(centro, angulo, scale=1.0)

    img_rotada = cv2.warpAffine(
        img_padded,
        matriz_rotacion,
        (wp, hp),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    return img_rotada, angulo

def leer_imagen_gris(ruta_imagen):
    """Lee una imagen en escala de grises manejando canales alfa sobre blanco."""
    img = cv2.imread(str(ruta_imagen), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"No se pudo cargar la imagen en: {ruta_imagen}")
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb = img[:, :, :3].astype(np.float32)
        img = (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def mascara_huella(gray, close_size=None):
    """Genera máscara binaria conectada de la huella."""
    h, w = gray.shape
    if close_size is None:
        close_size = max(9, int(0.035 * min(h, w)))
    close_size |= 1

    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.count_nonzero(bw) > 0.5 * bw.size:
        bw = cv2.bitwise_not(bw)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    mask = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == idx).astype(np.uint8) * 255

def angulo_para_vertical(mask):
    """Calcula el ángulo de alineación para que el eje mayor quede vertical."""
    ys, xs = np.nonzero(mask)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    pts -= pts.mean(axis=0)
    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    vx, vy = vecs[:, np.argmax(vals)]
    theta = -np.degrees(np.arctan2(vy, vx))
    delta = 90.0 - theta
    return (delta + 90.0) % 180.0 - 90.0

def recortar_y_centrar(gray, mask, size=200, margen=0.0):
    """Recorta la huella orientada y la centra en un lienzo cuadrado normalizado."""
    x, y, w, h = cv2.boundingRect(mask)
    m = int(round(margen * max(w, h)))
    gray_p = cv2.copyMakeBorder(gray, m, m, m, m, cv2.BORDER_CONSTANT, value=255)
    recorte = gray_p[y:y + h + 2 * m, x:x + w + 2 * m]

    lado = max(recorte.shape[0], recorte.shape[1])
    top = (lado - recorte.shape[0]) // 2
    left = (lado - recorte.shape[1]) // 2
    canvas = np.full((lado, lado), 255, np.uint8)
    canvas[top:top + recorte.shape[0], left:left + recorte.shape[1]] = recorte

    interp = cv2.INTER_AREA if size < lado else cv2.INTER_CUBIC
    return cv2.resize(canvas, (size, size), interpolation=interp)

def normalizar_huella_paso_3(gray, size=200):
    """
    Paso 3: Segmenta, orienta al eje vertical, recorta, aplica filtros y binariza.
    """
    mask = mascara_huella(gray)
    if mask is None:
        raise ValueError("No se pudo segmentar la región de la huella.")

    # Rotación a vertical
    ang = angulo_para_vertical(mask)
    h, w = gray.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw = int(np.ceil(h * sin + w * cos))
    nh = int(np.ceil(h * cos + w * sin))
    M[0, 2] += nw / 2.0 - cx
    M[1, 2] += nh / 2.0 - cy

    gray_rot = cv2.warpAffine(gray, M, (nw, nh), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    mask_rot = cv2.warpAffine(mask, M, (nw, nh), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Recorte y centrado
    out = recortar_y_centrar(gray_rot, mask_rot, size=size, margen=0.0)

    # Filtros: Suavizado y Unsharp Masking
    blur = cv2.GaussianBlur(out, (0, 0), sigmaX=1.2)
    unsharp = cv2.addWeighted(out, 2.0, blur, -1.0, 0)

    # Binarización Otsu estricta
    _, binarizada = cv2.threshold(unsharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binarizada

def maximo_minucia(lista_conteos):
    """Determina la minucia ganadora (1: fin/derecha, 2: bifurcación/izquierda, 3: horizontal)."""
    maximo = 0
    ganador = 0
    for i in range(3):
        if lista_conteos[i] > maximo:
            maximo = lista_conteos[i]
            ganador = i + 1
    return ganador

def extraer_matriz_8x8_paso_4(img_norm):
    """
    Paso 4: Aplica esqueletización, conteo de cruces direccionales (Crossing Number)
    y particiona la huella en una matriz de 8x8 con el tipo de minucia dominante.
    Retorna la matriz de (8, 8) y el vector aplanado de (64,).
    """
    _, bin_inv = cv2.threshold(img_norm, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    esqueleto = cv2.ximgproc.thinning(bin_inv)
    esqueleto_norm = esqueleto // 255

    kernel_derecho = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)
    kernel_izquierdo = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    kernel_horizontal = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)

    c_der = cv2.filter2D(esqueleto_norm, -1, kernel_derecho)
    c_izq = cv2.filter2D(esqueleto_norm, -1, kernel_izquierdo)
    c_hor = cv2.filter2D(esqueleto_norm, -1, kernel_horizontal)

    derechos = (esqueleto_norm == 1) & (c_der == 2)
    izquierdos = (esqueleto_norm == 1) & (c_izq == 2)
    horizontales = (esqueleto_norm == 1) & (c_hor == 3)
    totales = (esqueleto_norm == 1)

    y_d, x_d = np.where(derechos)
    y_i, x_i = np.where(izquierdos)
    y_h, x_h = np.where(horizontales)
    y_t, x_t = np.where(totales)

    if len(x_t) == 0 or len(y_t) == 0:
        raise ValueError("No se encontraron crestas de huella válidas tras la esqueletización.")

    lim_y_min, lim_y_max = y_t.min(), y_t.max()
    lim_x_min, lim_x_max = x_t.min(), x_t.max()
    deltay = max(1, int((lim_y_max - lim_y_min) / 8))
    deltax = max(1, int((lim_x_max - lim_x_min) / 8))

    huella_grid = np.zeros((8, 8, 3), dtype=int)

    for x, y in zip(x_d, y_d):
        bx = min(7, max(0, (x - lim_x_min) // deltax))
        by = min(7, max(0, (y - lim_y_min) // deltay))
        huella_grid[bx, by, 0] += 1

    for x, y in zip(x_i, y_i):
        bx = min(7, max(0, (x - lim_x_min) // deltax))
        by = min(7, max(0, (y - lim_y_min) // deltay))
        huella_grid[bx, by, 1] += 1

    for x, y in zip(x_h, y_h):
        bx = min(7, max(0, (x - lim_x_min) // deltax))
        by = min(7, max(0, (y - lim_y_min) // deltay))
        huella_grid[bx, by, 2] += 1

    matriz_8x8 = np.zeros((8, 8), dtype=int)
    for r in range(8):
        for c in range(8):
            matriz_8x8[r, c] = maximo_minucia(huella_grid[c, r])

    return matriz_8x8, matriz_8x8.flatten().astype(np.float64)

# ==============================================================================
# 2. CARGA DEL MODELO K-MEANS GUARDADO
# ==============================================================================

def cargar_modelo():
    """Carga los centroides y datos de referencia generados por 5-k_means_8x8.py."""
    dir_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_modelo = os.path.join(dir_actual, 'modelo_kmeans.npz')

    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError(
            f"No se encontró el modelo guardado en '{ruta_modelo}'. "
            "Por favor ejecute primero '5-k_means_8x8.py' para entrenar y guardar los centroides."
        )

    datos = np.load(ruta_modelo, allow_pickle=True)
    return {
        'X': datos['X'],
        'y_real': datos['y_real'],
        'etiquetas': datos['etiquetas'],
        'centroides': datos['centroides'],
        'nombres_clases': list(datos['nombres_clases']),
        'cluster_a_clase': list(datos['cluster_a_clase'])
    }

# ==============================================================================
# 3. IDENTIFICACIÓN Y GRAFICADO
# ==============================================================================

def graficar_identificacion(modelo, vector_externo, matriz_8x8, img_norm, 
                            cluster_cercano, distancia_min, distancia_max, veredicto, nombre_clase):
    """
    Muestra la proyección PCA con los clústeres y la huella externa destacada,
    junto con el heatmap de su matriz 8x8 y la imagen normalizada.
    """
    X = modelo['X']
    etiquetas = modelo['etiquetas']
    centroides = modelo['centroides']
    k = len(centroides)

    # Ajustar PCA con los datos de entrenamiento
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    centroides_pca = pca.transform(centroides)
    externo_pca = pca.transform(vector_externo.reshape(1, -1))[0]

    fig = plt.figure(figsize=(15, 6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.8, 1, 1])

    # 1. Gráfica PCA
    ax_pca = fig.add_subplot(gs[:, 0])
    colores = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']

    for i in range(k):
        pts = X_pca[etiquetas == i]
        etiqueta_nombre = modelo['cluster_a_clase'][i] if i < len(modelo['cluster_a_clase']) else f'Clúster {i}'
        ax_pca.scatter(pts[:, 0], pts[:, 1], c=colores[i % len(colores)],
                       label=f'{etiqueta_nombre} (Clúster {i})', alpha=0.55, s=40, edgecolors='none')

    ax_pca.scatter(centroides_pca[:, 0], centroides_pca[:, 1], c='black', marker='X', s=180,
                   label='Centroides K-Means', edgecolors='white', linewidth=1.5)

    # Huella Externa
    color_externo = '#00cc44' if veredicto else '#ee2222'
    simbolo = '*'
    ax_pca.scatter(externo_pca[0], externo_pca[1], c=color_externo, marker=simbolo, s=350,
                   edgecolors='black', linewidth=1.5, zorder=10,
                   label=f'Huella Externa: {"ACEPTADA" if veredicto else "RECHAZADA"}')

    # Línea que conecta la huella externa con su centroide más próximo
    c_prox_pca = centroides_pca[cluster_cercano]
    ax_pca.plot([externo_pca[0], c_prox_pca[0]], [externo_pca[1], c_prox_pca[1]],
                color=color_externo, linestyle='--', linewidth=1.8, alpha=0.8)

    ax_pca.set_title(
        f"Espacio PCA - Resultado: {'AUTORIZADA' if veredicto else 'RECHAZADA'} "
        f"({nombre_clase} | d = {distancia_min:.2f} / umbral: {distancia_max:.2f})",
        fontsize=11, fontweight='bold', color=('green' if veredicto else 'red')
    )
    ax_pca.set_xlabel('Componente Principal 1')
    ax_pca.set_ylabel('Componente Principal 2')
    ax_pca.grid(True, linestyle='--', alpha=0.5)
    ax_pca.legend(loc='best', fontsize=9)

    # 2. Imagen normalizada procesada
    ax_img = fig.add_subplot(gs[:, 1])
    ax_img.imshow(img_norm, cmap='gray')
    ax_img.set_title("Huella Preprocesada (200x200)", fontsize=11, fontweight='bold')
    ax_img.axis('off')

    # 3. Matriz 8x8 obtenida (Heatmap)
    ax_mat = fig.add_subplot(gs[:, 2])
    cax = ax_mat.imshow(matriz_8x8, cmap='viridis', interpolation='nearest', vmin=0, vmax=3)
    ax_mat.set_title("Matriz 8x8 de Minucias", fontsize=11, fontweight='bold')
    ax_mat.set_xticks(range(8))
    ax_mat.set_yticks(range(8))

    for r in range(8):
        for c in range(8):
            val = matriz_8x8[r, c]
            color_texto = "white" if val < 1.5 else "black"
            ax_mat.text(c, r, f"{val}", ha="center", va="center", color=color_texto, fontsize=9, fontweight='bold')

    fig.colorbar(cax, ax=ax_mat, orientation='horizontal', fraction=0.06, pad=0.1, label='Minucia Dominante (0-3)')
    fig.suptitle('Identificación de Huella Dactilar Externa', fontsize=14, fontweight='bold')

    plt.show()

def procesar_e_identificar(ruta_imagen, distancia_max=4.0, simular=False, mostrar_grafico=True):
    """
    Función principal que ejecuta los pasos 2 a 4 sobre una imagen externa
    y la compara contra los centroides guardados de K-Means.
    """
    print("=================================================================")
    print("           IDENTIFICACIÓN DE HUELLA EXTERNA                      ")
    print("=================================================================")
    print(f"Archivo de entrada : {ruta_imagen}")
    print(f"Umbral de distancia: {distancia_max}")
    print(f"Simular captura    : {simular}")

    # 1. Cargar el modelo guardado en 5-k_means_8x8.py
    modelo = cargar_modelo()
    centroides = modelo['centroides']

    # 2. Cargar imagen externa
    img_gris = leer_imagen_gris(ruta_imagen)

    # Paso 2 opcional: Simular captura
    if simular:
        print("[Paso 2] Aplicando simulación de captura con rotación aleatoria...")
        img_gris, ang = simular_captura(img_gris)
        print(f"         Ángulo aplicado: {ang:.2f}°")

    # Paso 3: Normalización
    print("[Paso 3] Normalizando huella (orientación vertical, recorte, filtros)...")
    img_norm = normalizar_huella_paso_3(img_gris, size=200)

    # Paso 4: Extracción de matriz 8x8
    print("[Paso 4] Esqueletización y generación de matriz 8x8...")
    matriz_8x8, vector_64 = extraer_matriz_8x8_paso_4(img_norm)

    # 3. Comparación de distancias contra los centroides
    print("\n[Clasificación K-Means] Evaluando distancias a centroides:")
    distancias = np.linalg.norm(centroides - vector_64, axis=1)

    cluster_cercano = int(np.argmin(distancias))
    distancia_min = distancias[cluster_cercano]
    nombre_clase = modelo['cluster_a_clase'][cluster_cercano]

    for i, d in enumerate(distancias):
        clase_i = modelo['cluster_a_clase'][i]
        indicador = " <-- MÁS CERCANO" if i == cluster_cercano else ""
        print(f"  Clúster {i} ({clase_i:10s}): distancia = {d:.3f}{indicador}")

    # Evaluación del umbral de distancia
    veredicto = (distancia_min <= distancia_max)

    print("\n------------------ RESULTADO DE IDENTIFICACIÓN ------------------")
    if veredicto:
        print(f"  ESTADO     : PERMITIDA / AUTORIZADA")
        print(f"  IDENTIDAD  : {nombre_clase}")
        print(f"  DISTANCIA  : {distancia_min:.3f} (<= umbral {distancia_max})")
    else:
        print(f"  ESTADO     : RECHAZADA / DESCONOCIDA")
        print(f"  CLÚSTER MÁS PRÓXIMO: {nombre_clase}")
        print(f"  DISTANCIA  : {distancia_min:.3f} (SUPERA el umbral {distancia_max})")
    print("-----------------------------------------------------------------\n")

    if mostrar_grafico:
        print("Desplegando gráfico de visualización...")
        graficar_identificacion(modelo, vector_64, matriz_8x8, img_norm,
                               cluster_cercano, distancia_min, distancia_max, veredicto, nombre_clase)

    return {
        'veredicto': veredicto,
        'clase_identificada': nombre_clase if veredicto else None,
        'cluster_cercano': cluster_cercano,
        'distancia_min': distancia_min,
        'vector_64': vector_64,
        'matriz_8x8': matriz_8x8,
        'img_norm': img_norm,
        'modelo': modelo
    }

# ==============================================================================
# MAIN CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Identificar huella dactilar externa con K-Means (8x8).")
    parser.add_argument("--img", type=str, default=None,
                        help="Ruta a la imagen de huella externa a identificar.")
    parser.add_argument("--distancia", type=float, default=4.0,
                        help="Distancia máxima euclídea del centroide para considerarse correcta (default: 4.0).")
    parser.add_argument("--simular", action="store_true",
                        help="Aplica rotación aleatoria y padding (Paso 2) antes de procesar.")

    args = parser.parse_args()

    # Si no se pasó ruta, usar una imagen de ejemplo del repositorio para prueba interactiva
    if not args.img:
        dir_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ruta_defecto = os.path.join(
            dir_base, "Huellas", "VariantesPermitidas_Tomadas", "huella_1", "0_1.png"
        )
        if os.path.exists(ruta_defecto):
            print(f"No se especificó --img. Usando imagen de prueba por defecto:\n  {ruta_defecto}\n")
            args.img = ruta_defecto
        else:
            print("[ERROR] Debe especificar una ruta con --img <ruta_huella>")
            sys.exit(1)

    procesar_e_identificar(args.img, distancia_max=args.distancia, simular=args.simular)

if __name__ == "__main__":
    main()

