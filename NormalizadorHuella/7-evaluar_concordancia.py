"""
7-evaluar_concordancia.py
-------------------------
Evalúa la concordancia y robustez del clasificador K-Means de huellas dactilares.
Procesa huellas permitidas (autorizadas) y huellas no permitidas de prueba (intrusos),
aplicando la simulación de captura en escáner al vuelo para las huellas de prueba.

Genera:
  1. Matriz de concordancia / asignación a clústeres.
  2. Matriz de confusión binaria (Aceptada / Rechazada) y métricas de seguridad (FAR, FRR, TAR, TRR).
  3. Registro detallado de Falsas Aceptaciones (archivo CSV y reporte TXT).
  4. Visualizaciones completas:
     - Proyección PCA 2D (con entrenamiento, centroides y muestras de prueba).
     - Heatmap de la matriz de concordancia.
     - Histograma y distribución de distancias frente al umbral euclídeo.
"""

import os
import sys
import glob
import argparse
import csv
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Importar funciones de 6-identificarHuella.py
DIR_ACTUAL = os.path.dirname(os.path.abspath(__file__))
DIR_PROYECTO = os.path.dirname(DIR_ACTUAL)

sys.path.insert(0, DIR_ACTUAL)
import importlib.util

_ruta_paso6 = os.path.join(DIR_ACTUAL, "6-identificarHuella.py")
if not os.path.exists(_ruta_paso6):
    raise FileNotFoundError(f"No se encontró '6-identificarHuella.py' en {_ruta_paso6}")

_spec6 = importlib.util.spec_from_file_location("identificar_huella", _ruta_paso6)
_mod_paso6 = importlib.util.module_from_spec(_spec6)
_spec6.loader.exec_module(_mod_paso6)

cargar_modelo = _mod_paso6.cargar_modelo
procesar_e_identificar = _mod_paso6.procesar_e_identificar
leer_imagen_gris = _mod_paso6.leer_imagen_gris
simular_captura = _mod_paso6.simular_captura
extraer_matriz_8x8_paso_4 = _mod_paso6.extraer_matriz_8x8_paso_4
_mod_paso3 = _mod_paso6._mod_paso3
normalizar_huella_paso_3 = _mod_paso6.normalizar_huella_paso_3


def extraer_vector_huella(ruta_imagen, simular=False):
    """
    Ejecuta el pipeline completo para una imagen:
      Lectura -> (Simulación de escáner opcional) -> Normalización Paso 3 -> Esqueletización y Matriz 8x8 Paso 4.
    Retorna el vector 64D y la matriz 8x8.
    """
    img_gris = leer_imagen_gris(ruta_imagen)
    if simular:
        img_gris, _ = simular_captura(img_gris)

    if _mod_paso3 is not None:
        img_norm = _mod_paso3.procesar(
            img_gris,
            size=200,
            margen=0,
            suavizado=1.2,
            sharp_cantidad=1
        )
    else:
        img_norm = normalizar_huella_paso_3(img_gris, size=200)

    matriz_8x8, vector_64 = extraer_matriz_8x8_paso_4(img_norm)
    return vector_64, matriz_8x8, img_norm


def buscar_imagenes_en_ruta(ruta_dir):
    """
    Busca de manera recursiva todas las imágenes (.png, .jpg, .jpeg, .bmp, .tif)
    en una carpeta o lista de carpetas.
    """
    extensiones = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif',
                   '*.PNG', '*.JPG', '*.JPEG', '*.BMP', '*.TIF')
    imagenes = []
    if not os.path.exists(ruta_dir):
        return imagenes

    for ext in extensiones:
        imagenes.extend(glob.glob(os.path.join(ruta_dir, "**", ext), recursive=True))

    return sorted(list(set(imagenes)))


def evaluar_conjunto(rutas_imagenes, modelo, distancia_max=3.8, simular=False, etiqueta_esperada=None):
    """
    Evalúa una lista de imágenes contra los centroides del modelo K-Means.
    """
    centroides = modelo['centroides']
    k = len(centroides)
    cluster_a_clase = modelo['cluster_a_clase']

    resultados = []

    for idx, ruta in enumerate(rutas_imagenes, 1):
        nombre_archivo = os.path.basename(ruta)
        nombre_carpeta_padre = os.path.basename(os.path.dirname(ruta))
        etiqueta_real = etiqueta_esperada if etiqueta_esperada is not None else nombre_carpeta_padre

        try:
            vector_64, matriz_8x8, _ = extraer_vector_huella(ruta, simular=simular)
            distancias = np.linalg.norm(centroides - vector_64, axis=1)
            cluster_cercano = int(np.argmin(distancias))
            distancia_min = float(distancias[cluster_cercano])
            clase_cercana = cluster_a_clase[cluster_cercano]

            veredicto = bool(distancia_min <= distancia_max)

            resultados.append({
                'archivo': nombre_archivo,
                'ruta_completa': ruta,
                'carpeta': nombre_carpeta_padre,
                'etiqueta_real': etiqueta_real,
                'vector_64': vector_64,
                'distancias': distancias,
                'cluster_cercano': cluster_cercano,
                'clase_cercana': clase_cercana,
                'distancia_min': distancia_min,
                'veredicto': veredicto,  # True = Aceptada como permitida, False = Rechazada
                'error': None
            })
        except Exception as e:
            resultados.append({
                'archivo': nombre_archivo,
                'ruta_completa': ruta,
                'carpeta': nombre_carpeta_padre,
                'etiqueta_real': etiqueta_real,
                'vector_64': None,
                'distancias': None,
                'cluster_cercano': -1,
                'clase_cercana': 'ERROR',
                'distancia_min': float('inf'),
                'veredicto': False,
                'error': str(e)
            })

    return resultados


def graficar_resultados_completos(modelo, res_permitidas, res_no_permitidas, distancia_max, guardar_figura=None):
    """
    Genera una figura integrada con 3 paneles:
      1. Proyección PCA 2D: Muestra los datos de entrenamiento, centroides y las muestras evaluadas
         (distinguiendo aciertos de falsas aceptaciones).
      2. Matriz de Concordancia / Asignación a Clústeres (Heatmap).
      3. Distribución de distancias euclídeas frente al umbral de corte.
    """
    fig = plt.figure(figsize=(18, 5.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 0.9, 1.1])

    # -------------------------------------------------------------
    # PANEL 1: Proyección PCA 2D
    # -------------------------------------------------------------
    ax_pca = fig.add_subplot(gs[0, 0])
    X_train = modelo['X']
    etiquetas_train = modelo['etiquetas']
    centroides = modelo['centroides']
    k = len(centroides)

    pca = PCA(n_components=2, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    centroides_pca = pca.transform(centroides)

    colores_clusters = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']

    # Puntos de entrenamiento
    for i in range(k):
        pts = X_train_pca[etiquetas_train == i]
        nombre_c = modelo['cluster_a_clase'][i]
        ax_pca.scatter(pts[:, 0], pts[:, 1], c=colores_clusters[i % len(colores_clusters)],
                       alpha=0.30, s=25, label=f'Train: {nombre_c}')

    # Centroides
    ax_pca.scatter(centroides_pca[:, 0], centroides_pca[:, 1], c='black', marker='X', s=160,
                   edgecolors='white', linewidth=1.5, zorder=5, label='Centroides K-Means')

    # Eval Permitidas
    pts_perm = [r['vector_64'] for r in res_permitidas if r['vector_64'] is not None]
    if pts_perm:
        pts_perm_pca = pca.transform(np.array(pts_perm))
        ax_pca.scatter(pts_perm_pca[:, 0], pts_perm_pca[:, 1], c='#2a9d8f', marker='o', s=45,
                       edgecolors='black', linewidth=0.7, alpha=0.85, label='Permitidas Evaluadas')

    # Eval No Permitidas
    # Separar en Rechazadas (Correctas) vs Falsas Aceptaciones
    vectores_rechazados = []
    vectores_falsas_ac = []
    for r in res_no_permitidas:
        if r['vector_64'] is not None:
            if r['veredicto']:  # Falsa aceptación
                vectores_falsas_ac.append(r['vector_64'])
            else:  # Rechazo correcto
                vectores_rechazados.append(r['vector_64'])

    if vectores_rechazados:
        pts_rech_pca = pca.transform(np.array(vectores_rechazados))
        ax_pca.scatter(pts_rech_pca[:, 0], pts_rech_pca[:, 1], c='#4a5568', marker='^', s=55,
                       edgecolors='black', linewidth=0.7, alpha=0.85, label='No Permitidas (Rechazadas ✓)')

    if vectores_falsas_ac:
        pts_fa_pca = pca.transform(np.array(vectores_falsas_ac))
        ax_pca.scatter(pts_fa_pca[:, 0], pts_fa_pca[:, 1], c='#e63946', marker='*', s=160,
                       edgecolors='black', linewidth=1.2, zorder=6, label='Falsas Aceptaciones (Alerta ⚠)')

    ax_pca.set_title("Proyección PCA 2D", fontsize=11, fontweight='bold')
    ax_pca.set_xlabel('Componente Principal 1')
    ax_pca.set_ylabel('Componente Principal 2')
    ax_pca.grid(True, linestyle='--', alpha=0.4)
    ax_pca.legend(loc='best', fontsize=8)

    # -------------------------------------------------------------
    # PANEL 2: Matriz de Concordancia / Asignación
    # -------------------------------------------------------------
    ax_conf = fig.add_subplot(gs[0, 1])
    nombres_filas = []
    matriz_filas = []

    # Agrupar asignaciones de permitidas por carpeta
    carpetas_permitidas = sorted(list(set(r['carpeta'] for r in res_permitidas if r['vector_64'] is not None)))
    for c in carpetas_permitidas:
        nombres_filas.append(f"Perm: {c}")
        conteos = [0] * (k + 1)  # clúster 0, 1, 2, ... y Rechazadas (> umbral)
        for r in res_permitidas:
            if r['carpeta'] == c and r['vector_64'] is not None:
                if r['veredicto']:
                    conteos[r['cluster_cercano']] += 1
                else:
                    conteos[-1] += 1
        matriz_filas.append(conteos)

    # Agrupar no permitidas
    carpetas_no_perm = sorted(list(set(r['carpeta'] for r in res_no_permitidas if r['vector_64'] is not None)))
    for c in carpetas_no_perm:
        nombres_filas.append(f"NoPerm: {c}")
        conteos = [0] * (k + 1)
        for r in res_no_permitidas:
            if r['carpeta'] == c and r['vector_64'] is not None:
                if r['veredicto']:
                    conteos[r['cluster_cercano']] += 1
                else:
                    conteos[-1] += 1
        matriz_filas.append(conteos)

    matriz_data = np.array(matriz_filas) if matriz_filas else np.zeros((1, k + 1))
    encabezados_col = [f"Clúster {i}\n({modelo['cluster_a_clase'][i]})" for i in range(k)] + ["Rechazada\n(> umbral)"]

    im = ax_conf.imshow(matriz_data, cmap='Blues', aspect='auto')
    ax_conf.set_title("Matriz de Asignación y Concordancia", fontsize=11, fontweight='bold')
    ax_conf.set_xticks(range(len(encabezados_col)))
    ax_conf.set_xticklabels(encabezados_col, fontsize=8, rotation=20)
    ax_conf.set_yticks(range(len(nombres_filas)))
    ax_conf.set_yticklabels(nombres_filas, fontsize=9)

    for r_idx in range(matriz_data.shape[0]):
        for c_idx in range(matriz_data.shape[1]):
            val = matriz_data[r_idx, c_idx]
            color_txt = "white" if val > np.max(matriz_data) * 0.6 else "black"
            ax_conf.text(c_idx, r_idx, f"{int(val)}", ha="center", va="center",
                         color=color_txt, fontsize=10, fontweight='bold')

    plt.colorbar(im, ax=ax_conf, fraction=0.046, pad=0.04)

    # -------------------------------------------------------------
    # PANEL 3: Distribución de Distancias vs Umbral
    # -------------------------------------------------------------
    ax_dist = fig.add_subplot(gs[0, 2])
    dists_perm = [r['distancia_min'] for r in res_permitidas if r['distancia_min'] < float('inf')]
    dists_no_perm = [r['distancia_min'] for r in res_no_permitidas if r['distancia_min'] < float('inf')]

    bins = np.linspace(0, max(max(dists_perm or [6]), max(dists_no_perm or [6])) + 1, 20)

    if dists_perm:
        ax_dist.hist(dists_perm, bins=bins, alpha=0.65, color='#2a9d8f', label=f'Permitidas (N={len(dists_perm)})')
    if dists_no_perm:
        ax_dist.hist(dists_no_perm, bins=bins, alpha=0.65, color='#e76f51', label=f'No Permitidas (N={len(dists_no_perm)})')

    ax_dist.axvline(distancia_max, color='#e63946', linestyle='--', linewidth=2.0,
                    label=f'Umbral = {distancia_max:.2f}')

    ax_dist.set_title("Distribución de Distancias Mínimas al Centroide", fontsize=11, fontweight='bold')
    ax_dist.set_xlabel("Distancia Euclídea")
    ax_dist.set_ylabel("Frecuencia")
    ax_dist.grid(True, linestyle='--', alpha=0.4)
    ax_dist.legend(loc='best', fontsize=9)

    if guardar_figura:
        plt.savefig(guardar_figura, dpi=300, bbox_inches='tight')
        print(f"[Gráfica guardada] Figura exportada en: {guardar_figura}")

    if plt.isinteractive() or not plt.get_backend().lower().startswith('agg'):
        plt.show()
    else:
        plt.close(fig)


def generar_reportes(res_permitidas, res_no_permitidas, distancia_max, ruta_txt, ruta_csv):
    """
    Calcula métricas estadísticas y genera los archivos de reporte.
    """
    total_perm = len(res_permitidas)
    total_no_perm = len(res_no_permitidas)

    # Métricas Permitidas
    perm_aceptadas = sum(1 for r in res_permitidas if r['veredicto'])
    perm_rechazadas = total_perm - perm_aceptadas
    tar = (perm_aceptadas / total_perm * 100) if total_perm > 0 else 0.0  # True Acceptance Rate
    frr = (perm_rechazadas / total_perm * 100) if total_perm > 0 else 0.0  # False Rejection Rate

    # Métricas No Permitidas
    no_perm_rechazadas = sum(1 for r in res_no_permitidas if not r['veredicto'])
    no_perm_aceptadas = total_no_perm - no_perm_rechazadas  # FALSAS ACEPTACIONES
    trr = (no_perm_rechazadas / total_no_perm * 100) if total_no_perm > 0 else 0.0  # True Rejection Rate
    far = (no_perm_aceptadas / total_no_perm * 100) if total_no_perm > 0 else 0.0  # False Acceptance Rate

    falsas_aceptaciones = [r for r in res_no_permitidas if r['veredicto']]

    # Exportar CSV de Falsas Aceptaciones
    with open(ruta_csv, mode='w', newline='', encoding='utf-8') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(['archivo', 'carpeta_origen', 'ruta_completa', 'distancia_minima',
                         'umbral', 'cluster_confundido', 'identidad_confundida', 'diferencia_con_umbral'])
        for fa in falsas_aceptaciones:
            writer.writerow([
                fa['archivo'],
                fa['carpeta'],
                fa['ruta_completa'],
                f"{fa['distancia_min']:.4f}",
                distancia_max,
                fa['cluster_cercano'],
                fa['clase_cercana'],
                f"{(distancia_max - fa['distancia_min']):.4f}"
            ])

    # Generar Texto de Reporte
    lineas_txt = []
    lineas_txt.append("=" * 70)
    lineas_txt.append("REPORTE DE EVALUACIÓN DE CONCORDANCIA Y SEGURIDAD BIOMÉTRICA")
    lineas_txt.append(f"Fecha de ejecución : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lineas_txt.append(f"Umbral de distancia: {distancia_max}")
    lineas_txt.append("=" * 70)

    lineas_txt.append("\n1. RESUMEN DE HUELLAS PERMITIDAS (AUTORIZADAS):")
    lineas_txt.append(f"  - Total evaluadas          : {total_perm}")
    lineas_txt.append(f"  - Aceptadas correctamente (TAR): {perm_aceptadas} ({tar:.2f}%)")
    lineas_txt.append(f"  - Rechazadas falsamente  (FRR): {perm_rechazadas} ({frr:.2f}%)")

    lineas_txt.append("\n2. RESUMEN DE HUELLAS NO PERMITIDAS (INTRUSOS / PRUEBA):")
    lineas_txt.append(f"  - Total evaluadas          : {total_no_perm}")
    lineas_txt.append(f"  - Rechazadas correctamente(TRR): {no_perm_rechazadas} ({trr:.2f}%)")
    lineas_txt.append(f"  - Falsas Aceptaciones    (FAR): {no_perm_aceptadas} ({far:.2f}%)")

    lineas_txt.append("\n3. DETALLE DE HUELLAS NO PERMITIDAS CON FALSA ACEPTACIÓN (FAR):")
    if len(falsas_aceptaciones) == 0:
        lineas_txt.append("  [EXCELENTE] Ninguna huella no permitida fue aceptada como autorizada.")
    else:
        lineas_txt.append(f"  Total de anomalías detectadas: {len(falsas_aceptaciones)}")
        lineas_txt.append("  " + "-" * 66)
        lineas_txt.append(f"  {'Archivo':<18} | {'Carpeta':<10} | {'Distancia':<9} | {'Confundida Con':<15}")
        lineas_txt.append("  " + "-" * 66)
        for fa in falsas_aceptaciones:
            lineas_txt.append(
                f"  {fa['archivo']:<18} | {fa['carpeta']:<10} | {fa['distancia_min']:<9.3f} | {fa['clase_cercana']} (Clúster {fa['cluster_cercano']})"
            )
        lineas_txt.append("  " + "-" * 66)

    # Detalle de distancias
    d_no_perm = [r['distancia_min'] for r in res_no_permitidas if r['distancia_min'] < float('inf')]
    if d_no_perm:
        lineas_txt.append("\n4. ESTADÍSTICAS DE DISTANCIAS EN NO PERMITIDAS:")
        lineas_txt.append(f"  - Distancia mínima: {min(d_no_perm):.3f}")
        lineas_txt.append(f"  - Distancia media : {np.mean(d_no_perm):.3f}")
        lineas_txt.append(f"  - Distancia máxima: {max(d_no_perm):.3f}")
        lineas_txt.append(f"  - Desv. estándar  : {np.std(d_no_perm):.3f}")

    contenido_txt = "\n".join(lineas_txt)

    with open(ruta_txt, mode='w', encoding='utf-8') as f_txt:
        f_txt.write(contenido_txt)

    return contenido_txt, falsas_aceptaciones


def main():
    parser = argparse.ArgumentParser(
        description="Evalúa la concordancia y falsas aceptaciones en huellas con K-Means."
    )
    parser.add_argument("--dir_no_permitidas", type=str, default=None,
                        help="Ruta a huellas no permitidas (default: Huellas/VariantesPrueba)")
    parser.add_argument("--dir_permitidas", type=str, default=None,
                        help="Ruta a huellas permitidas (default: Huellas/VariantesPermitidas_Tomadas)")
    parser.add_argument("--distancia", type=float, default=4,
                        help="Umbral de distancia euclídea (default: 4)")
    parser.add_argument("--simular_no_permitidas", action="store_true", default=True,
                        help="Aplica rotación aleatoria y recorte (Paso 2) a las huellas no permitidas (default: True)")
    parser.add_argument("--no_simular", dest="simular_no_permitidas", action="store_false",
                        help="Desactiva la simulación de escáner en huellas no permitidas")
    parser.add_argument("--simular_permitidas", action="store_true", default=False,
                        help="Aplica simulación también a huellas permitidas (default: False)")
    parser.add_argument("--sin_graficos", action="store_true", default=False,
                        help="No muestra las ventanas de visualización emergentes")

    args = parser.parse_args()

    # Directorios por defecto
    if args.dir_no_permitidas is None:
        # Por requerimiento, priorizar VariantesPrueba
        ruta_vp = os.path.join(DIR_PROYECTO, "Huellas", "VariantesPrueba")
        # Si tiene subcarpeta Fingerprints/fp_1, o Impression_1/fp_1, o directamente la carpeta
        if os.path.exists(os.path.join(ruta_vp, "Fingerprints", "fp_1")):
            args.dir_no_permitidas = os.path.join(ruta_vp, "Fingerprints", "fp_1")
        elif os.path.exists(os.path.join(ruta_vp, "Impression_1", "fp_1")):
            args.dir_no_permitidas = os.path.join(ruta_vp, "Impression_1", "fp_1")
        else:
            args.dir_no_permitidas = ruta_vp

    if args.dir_permitidas is None:
        args.dir_permitidas = os.path.join(DIR_PROYECTO, "Huellas", "VariantesPermitidas_Tomadas")

    print("=================================================================")
    print("      EVALUACIÓN DE CONCORDANCIA Y FALSA ACEPTACIÓN (HUELLAS)    ")
    print("=================================================================")
    print(f"Ruta Huellas No Permitidas : {args.dir_no_permitidas}")
    print(f"  -> Simulación en escáner : {args.simular_no_permitidas}")
    print(f"Ruta Huellas Permitidas    : {args.dir_permitidas}")
    print(f"  -> Simulación en escáner : {args.simular_permitidas}")
    print(f"Umbral de distancia        : {args.distancia}")

    # 1. Cargar modelo K-Means
    print("\n[1/4] Cargando modelo K-Means entrenado...")
    modelo = cargar_modelo()
    print(f"  -> Centroides cargados : {len(modelo['centroides'])}")
    print(f"  -> Clases asociadas    : {modelo['cluster_a_clase']}")

    # 2. Búsqueda de archivos
    print("\n[2/4] Buscando archivos de huellas...")
    imgs_no_permitidas = buscar_imagenes_en_ruta(args.dir_no_permitidas)
    imgs_permitidas = buscar_imagenes_en_ruta(args.dir_permitidas)

    print(f"  -> Encontradas {len(imgs_no_permitidas)} imágenes no permitidas.")
    print(f"  -> Encontradas {len(imgs_permitidas)} imágenes permitidas.")

    if len(imgs_no_permitidas) == 0 and len(imgs_permitidas) == 0:
        print("[ERROR] No se encontraron imágenes para evaluar en ninguna de las rutas especificadas.")
        sys.exit(1)

    # 3. Procesamiento y Clasificación
    print("\n[3/4] Procesando huellas y calculando distancias...")
    res_no_permitidas = []
    if imgs_no_permitidas:
        print(f"  -> Evaluando {len(imgs_no_permitidas)} huellas no permitidas (simular={args.simular_no_permitidas})...")
        res_no_permitidas = evaluar_conjunto(
            imgs_no_permitidas,
            modelo,
            distancia_max=args.distancia,
            simular=args.simular_no_permitidas,
            etiqueta_esperada="NoPermitida"
        )

    res_permitidas = []
    if imgs_permitidas:
        print(f"  -> Evaluando {len(imgs_permitidas)} huellas permitidas (simular={args.simular_permitidas})...")
        res_permitidas = evaluar_conjunto(
            imgs_permitidas,
            modelo,
            distancia_max=args.distancia,
            simular=args.simular_permitidas
        )

    # 4. Generación de Reportes
    print("\n[4/4] Generando métricas y reportes...")
    ruta_txt = os.path.join(DIR_ACTUAL, "reporte_concordancia.txt")
    ruta_csv = os.path.join(DIR_ACTUAL, "reporte_falsas_aceptaciones.csv")
    ruta_png = os.path.join(DIR_ACTUAL, "grafico_concordancia.png")

    reporte_texto, falsas_ac = generar_reportes(
        res_permitidas,
        res_no_permitidas,
        distancia_max=args.distancia,
        ruta_txt=ruta_txt,
        ruta_csv=ruta_csv
    )

    print("\n" + reporte_texto)
    print(f"\n[Guardado] Reporte completo en: {ruta_txt}")
    print(f"[Guardado] Detalle de falsas aceptaciones en: {ruta_csv}")

    # 5. Graficación
    if not args.sin_graficos:
        print("\nDesplegando gráficos (PCA, Matriz de Concordancia y Distancias)...")
        graficar_resultados_completos(
            modelo,
            res_permitidas,
            res_no_permitidas,
            distancia_max=args.distancia,
            guardar_figura=ruta_png
        )
    else:
        # Generar y guardar la imagen sin abrir la ventana interactiva
        graficar_resultados_completos(
            modelo,
            res_permitidas,
            res_no_permitidas,
            distancia_max=args.distancia,
            guardar_figura=ruta_png
        )


if __name__ == "__main__":
    main()