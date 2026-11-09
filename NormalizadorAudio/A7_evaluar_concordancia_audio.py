"""
A7_evaluar_concordancia_audio.py
--------------------------------
Evalúa la concordancia y robustez del clasificador de comandos de voz (K-NN).
Aplica el algoritmo de clasificación y el umbral de descarte de A6_evaluarNuevoAudio.py
sobre los audios de la base de datos (comandos legítimos y audios de descarte/basura).

Genera:
  1. Matriz de concordancia / confusión completa (Clases reales vs Clases predichas + NO_RECONOCIDO).
  2. Métricas de seguridad y rendimiento biométrico (TAR, FRR, TRR, FAR, Exactitud).
  3. Registro detallado de errores y falsas aceptaciones en archivo CSV y reporte TXT.
  4. Visualizaciones integradas (3 paneles):
     - Heatmap de Matriz de Concordancia.
     - Histograma y distribución de distancias frente al umbral de corte.
     - Proyección PCA 2D del conjunto de datos y centroides.
"""

import os
import sys
import argparse
import csv
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Configurar rutas locales
DIR_ACTUAL = os.path.dirname(os.path.abspath(__file__))
DIR_PROYECTO = os.path.dirname(DIR_ACTUAL)
DIR_DATASET = os.path.join(DIR_ACTUAL, "DatasetVectores")
DIR_AUDIOS = os.path.join(DIR_ACTUAL, "AudiosPersonas")

if DIR_ACTUAL not in sys.path:
    sys.path.insert(0, DIR_ACTUAL)

from A5_ClasificadorVoz import ClasificadorVoz
import soundfile as sf
from A3_ProcesadorAudio import extraer_vector_ventanas

UMBRAL_DISTANCIA_DEFECTO = 8.5
RUTA_MODELO_DEFECTO = os.path.join(DIR_DATASET, "modelo_voz.npy")


def cargar_datos_desde_dataset():
    """
    Carga los vectores, etiquetas y metadatos preprocesados en DatasetVectores/.
    """
    ruta_X = os.path.join(DIR_DATASET, "X.npy")
    ruta_y = os.path.join(DIR_DATASET, "y.npy")
    ruta_meta = os.path.join(DIR_DATASET, "metadata.npy")

    if not (os.path.exists(ruta_X) and os.path.exists(ruta_y)):
        raise FileNotFoundError(
            f"No se encontraron los archivos X.npy / y.npy en '{DIR_DATASET}'. "
            "Ejecuta primero 'A4_reprocesar_dataset.py' para generarlos."
        )

    X = np.load(ruta_X)
    y = np.load(ruta_y)
    metadata = np.load(ruta_meta) if os.path.exists(ruta_meta) else np.array([f"Audio_{i+1}" for i in range(len(y))])

    return X, y, metadata


def cargar_datos_desde_archivos_audio(top_db=20, n_ventanas=3, duracion_ventana=0.25):
    """
    Procesa directamente los archivos 3_audioLimpio.wav de cada persona en AudiosPersonas/.
    """
    if not os.path.exists(DIR_AUDIOS):
        raise FileNotFoundError(f"No se encontró el directorio de audios en: {DIR_AUDIOS}")

    X = []
    y = []
    metadata = []

    carpetas_personas = sorted(os.listdir(DIR_AUDIOS))
    for persona in carpetas_personas:
        ruta_persona = os.path.join(DIR_AUDIOS, persona)
        if not os.path.isdir(ruta_persona):
            continue

        ruta_etiquetas = os.path.join(ruta_persona, "queDeberiaSer.txt")
        if not os.path.exists(ruta_etiquetas):
            continue

        with open(ruta_etiquetas, 'r', encoding='utf-8') as f:
            etiquetas = [linea.strip().lower() for linea in f.read().splitlines() if linea.strip()]

        for i, etiqueta in enumerate(etiquetas):
            num_audio = i + 1
            ruta_audio = os.path.join(ruta_persona, "Audios", f"Audio{num_audio}", "3_audioLimpio.wav")
            if not os.path.exists(ruta_audio):
                continue

            try:
                audio, fs = sf.read(ruta_audio)
                vector = extraer_vector_ventanas(
                    audio_limpio=audio,
                    fs=fs,
                    n_ventanas=n_ventanas,
                    duracion_ventana=duracion_ventana,
                    top_db=top_db
                )
                if vector is not None:
                    X.append(vector)
                    y.append(etiqueta)
                    metadata.append(f"{persona}_Audio{num_audio}")
            except Exception as e:
                print(f"Error procesando {ruta_audio}: {e}")

    return np.array(X), np.array(y), np.array(metadata)


def clasificar_con_umbral(clasificador, vector, umbral_max=UMBRAL_DISTANCIA_DEFECTO):
    """
    Aplica la misma lógica que A6_evaluarNuevoAudio.clasificar_vector():
    devuelve clase predicha más cercana, mapa de distancias y veredicto de reconocimiento.
    """
    clase_cercana, distancias = clasificador.predecir_uno(vector)
    distancia_minima = float(distancias[clase_cercana])

    es_valido = bool(umbral_max is None or distancia_minima <= umbral_max)
    clase_final = clase_cercana if es_valido else "NO_RECONOCIDO"

    return clase_final, clase_cercana, distancia_minima, distancias, es_valido


def evaluar_base_datos(X, y, metadata, clasificador=None, modo_eval="modelo_guardado", k_vecinos=22, umbral_max=UMBRAL_DISTANCIA_DEFECTO):
    """
    Evalúa cada muestra del dataset con el clasificador y umbral de rechazo.
    """
    total = len(X)
    resultados = []

    # Determinar si usamos modelo fijo o Leave-One-Out
    if modo_eval == "leave_one_out":
        print(f"  -> Ejecutando validación cruzada Leave-One-Out (K={k_vecinos})...")
        # Para Leave-One-Out, entrenamos sólo con las clases válidas
        mascara_validos = np.isin(y, ["reconocer", "identificar", "salir"])
        X_validos = X[mascara_validos]
        y_validos = y[mascara_validos]

        for i in range(total):
            x_i = X[i]
            y_i = str(y[i])
            meta_i = str(metadata[i])

            if y_i in ["reconocer", "identificar", "salir"]:
                # Excluir la muestra i del entrenamiento
                indices_train = [idx for idx in range(len(X_validos)) if not (np.array_equal(X_validos[idx], x_i) and y_validos[idx] == y_i)]
                clf_i = ClasificadorVoz(metodo="knn", k=k_vecinos)
                clf_i.entrenar(X_validos[indices_train], y_validos[indices_train])
            else:
                # Si es 'basura', se evalúa con todo el dataset de entrenamiento válido
                clf_i = ClasificadorVoz(metodo="knn", k=k_vecinos)
                clf_i.entrenar(X_validos, y_validos)

            clase_final, clase_cercana, dist_min, distancias, es_valido = clasificar_con_umbral(clf_i, x_i, umbral_max)
            resultados.append({
                'id': meta_i,
                'clase_real': y_i,
                'clase_final': clase_final,
                'clase_cercana': clase_cercana,
                'distancia_min': dist_min,
                'distancias': distancias,
                'es_valido': es_valido,
                'vector': x_i
            })
    else:
        # Modo modelo guardado
        if clasificador is None:
            if not os.path.exists(RUTA_MODELO_DEFECTO):
                raise FileNotFoundError(f"No existe el modelo de voz en: {RUTA_MODELO_DEFECTO}")
            clasificador = ClasificadorVoz.cargar_modelo(RUTA_MODELO_DEFECTO)

        for i in range(total):
            x_i = X[i]
            y_i = str(y[i])
            meta_i = str(metadata[i])

            clase_final, clase_cercana, dist_min, distancias, es_valido = clasificar_con_umbral(clasificador, x_i, umbral_max)
            resultados.append({
                'id': meta_i,
                'clase_real': y_i,
                'clase_final': clase_final,
                'clase_cercana': clase_cercana,
                'distancia_min': dist_min,
                'distancias': distancias,
                'es_valido': es_valido,
                'vector': x_i
            })

    return resultados


def graficar_concordancia_audio(resultados, clases_ordenadas, umbral_max, guardar_figura=None):
    """
    Genera figura de 3 paneles:
      1. Matriz de Concordancia / Confusión (Heatmap).
      2. Distribución de Distancias vs Umbral (Comandos Válidos vs Basura).
      3. Proyección PCA 2D del conjunto de datos con aciertos, errores y centroides.
    """
    fig = plt.figure(figsize=(18, 5.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.2])

    columnas_pred = [c for c in clases_ordenadas if c != "basura"] + ["NO_RECONOCIDO"]

    # -------------------------------------------------------------
    # PANEL 1: Matriz de Concordancia / Confusión
    # -------------------------------------------------------------
    ax_mat = fig.add_subplot(gs[0, 0])
    matriz = np.zeros((len(clases_ordenadas), len(columnas_pred)), dtype=int)

    idx_fila = {c: i for i, c in enumerate(clases_ordenadas)}
    idx_col = {c: j for j, c in enumerate(columnas_pred)}

    for r in resultados:
        f = idx_fila.get(r['clase_real'], -1)
        c = idx_col.get(r['clase_final'], -1)
        if f >= 0 and c >= 0:
            matriz[f, c] += 1

    im = ax_mat.imshow(matriz, cmap="Blues", aspect="auto")
    ax_mat.set_title("Matriz de Concordancia / Confusión", fontsize=11, fontweight='bold')
    ax_mat.set_xticks(range(len(columnas_pred)))
    ax_mat.set_xticklabels([c.upper() for c in columnas_pred], fontsize=8, rotation=20)
    ax_mat.set_yticks(range(len(clases_ordenadas)))
    ax_mat.set_yticklabels([c.capitalize() for c in clases_ordenadas], fontsize=9)
    ax_mat.set_ylabel("Clase Real (Audio Grabado)", fontsize=10, fontweight='bold')
    ax_mat.set_xlabel("Clase Predicha (Identificación)", fontsize=10, fontweight='bold')

    for r_i in range(matriz.shape[0]):
        for c_j in range(matriz.shape[1]):
            val = matriz[r_i, c_j]
            color_txt = "white" if val > np.max(matriz) * 0.55 else "black"
            ax_mat.text(c_j, r_i, f"{val}", ha="center", va="center", color=color_txt, fontsize=10, fontweight='bold')

    plt.colorbar(im, ax=ax_mat, fraction=0.046, pad=0.04)

    # -------------------------------------------------------------
    # PANEL 2: Distribución de Distancias vs Umbral
    # -------------------------------------------------------------
    ax_dist = fig.add_subplot(gs[0, 1])
    dists_validos = [r['distancia_min'] for r in resultados if r['clase_real'] != 'basura']
    dists_basura = [r['distancia_min'] for r in resultados if r['clase_real'] == 'basura']

    max_val = max((dists_validos or [10]) + (dists_basura or [10])) + 2
    bins = np.linspace(0, max_val, 22)

    if dists_validos:
        ax_dist.hist(dists_validos, bins=bins, alpha=0.65, color='#2a9d8f', label=f'Comandos Válidos (N={len(dists_validos)})')
    if dists_basura:
        ax_dist.hist(dists_basura, bins=bins, alpha=0.65, color='#e76f51', label=f'Basura / Descarte (N={len(dists_basura)})')

    ax_dist.axvline(umbral_max, color='#e63946', linestyle='--', linewidth=2.2, label=f'Umbral = {umbral_max:.1f}')
    ax_dist.set_title("Distribución de Distancias Mínimas al Modelo", fontsize=11, fontweight='bold')
    ax_dist.set_xlabel("Distancia Euclídea Estandarizada")
    ax_dist.set_ylabel("Frecuencia")
    ax_dist.grid(True, linestyle='--', alpha=0.4)
    ax_dist.legend(loc='upper right', fontsize=8)

    # -------------------------------------------------------------
    # PANEL 3: Proyección PCA 2D
    # -------------------------------------------------------------
    ax_pca = fig.add_subplot(gs[0, 2])
    vectores = np.array([r['vector'] for r in resultados])
    media = np.mean(vectores, axis=0)
    std = np.std(vectores, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    vectores_escalados = (vectores - media) / std

    pca = PCA(n_components=2, random_state=42)
    vectores_pca = pca.fit_transform(vectores_escalados)

    colores_clases = {
        'reconocer': '#1f77b4',
        'identificar': '#2ca02c',
        'salir': '#ff7f0e',
        'basura': '#6c757d'
    }

    # Graficar puntos correctos
    for c in clases_ordenadas:
        indices = [i for i, r in enumerate(resultados) if r['clase_real'] == c and r['clase_final'] == c]
        if indices:
            pts = vectores_pca[indices]
            ax_pca.scatter(pts[:, 0], pts[:, 1], c=colores_clases.get(c, '#333'),
                           label=f'{c.capitalize()} (Correcto)', alpha=0.7, s=45, edgecolors='black', linewidth=0.5)

    # Graficar Rechazos y Errores
    indices_error = [i for i, r in enumerate(resultados) if r['clase_real'] != 'basura' and r['clase_final'] != r['clase_real']]
    indices_fa_basura = [i for i, r in enumerate(resultados) if r['clase_real'] == 'basura' and r['clase_final'] != 'NO_RECONOCIDO']
    indices_rechazo_correcto = [i for i, r in enumerate(resultados) if r['clase_real'] == 'basura' and r['clase_final'] == 'NO_RECONOCIDO']

    if indices_error:
        pts = vectores_pca[indices_error]
        ax_pca.scatter(pts[:, 0], pts[:, 1], c='#d62728', marker='X', s=90, edgecolors='black', linewidth=1.0, label='Error / Falso Rechazo')

    if indices_rechazo_correcto:
        pts = vectores_pca[indices_rechazo_correcto]
        ax_pca.scatter(pts[:, 0], pts[:, 1], c='#495057', marker='s', s=55, edgecolors='black', linewidth=1.0, label='Basura (Rechazada ✓)')

    if indices_fa_basura:
        pts = vectores_pca[indices_fa_basura]
        ax_pca.scatter(pts[:, 0], pts[:, 1], c='#e63946', marker='*', s=160, edgecolors='black', linewidth=1.2, label='Basura (Falsa Aceptación ⚠)')

    ax_pca.set_title("Proyección PCA 2D del Espacio de Audio", fontsize=11, fontweight='bold')
    ax_pca.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax_pca.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax_pca.grid(True, linestyle='--', alpha=0.4)
    ax_pca.legend(loc='best', fontsize=8)

    if guardar_figura:
        plt.savefig(guardar_figura, dpi=300, bbox_inches='tight')
        print(f"[Gráfica guardada] Figura exportada en: {guardar_figura}")

    if plt.isinteractive() or not plt.get_backend().lower().startswith('agg'):
        plt.show()
    else:
        plt.close(fig)


def generar_reportes(resultados, umbral_max, ruta_txt, ruta_csv):
    """
    Calcula métricas estadísticas y genera archivos de reporte TXT y CSV.
    """
    res_validos = [r for r in resultados if r['clase_real'] != 'basura']
    res_basura = [r for r in resultados if r['clase_real'] == 'basura']

    total_validos = len(res_validos)
    total_basura = len(res_basura)

    # Métricas para comandos válidos
    aciertos_validos = sum(1 for r in res_validos if r['clase_final'] == r['clase_real'])
    rechazos_falsos_validos = sum(1 for r in res_validos if r['clase_final'] == 'NO_RECONOCIDO')
    confusiones_validos = sum(1 for r in res_validos if r['clase_final'] != r['clase_real'] and r['clase_final'] != 'NO_RECONOCIDO')

    exactitud_validos = (aciertos_validos / total_validos * 100) if total_validos > 0 else 0.0
    frr = (rechazos_falsos_validos / total_validos * 100) if total_validos > 0 else 0.0

    # Métricas para basura / descarte
    rechazos_correctos_basura = sum(1 for r in res_basura if r['clase_final'] == 'NO_RECONOCIDO')
    falsas_aceptaciones_basura = sum(1 for r in res_basura if r['clase_final'] != 'NO_RECONOCIDO')

    trr = (rechazos_correctos_basura / total_basura * 100) if total_basura > 0 else 0.0
    far = (falsas_aceptaciones_basura / total_basura * 100) if total_basura > 0 else 0.0

    # Detección de casos anómalos o de error
    errores = []
    for r in resultados:
        if r['clase_real'] == 'basura':
            if r['clase_final'] != 'NO_RECONOCIDO':
                errores.append({
                    **r,
                    'tipo_error': 'FALSA_ACEPTACION_BASURA',
                    'detalle': f"Audio de basura clasificado como '{r['clase_final']}'"
                })
        else:
            if r['clase_final'] != r['clase_real']:
                tipo = 'FALSO_RECHAZO' if r['clase_final'] == 'NO_RECONOCIDO' else 'CONFUSION_CLASE'
                errores.append({
                    **r,
                    'tipo_error': tipo,
                    'detalle': f"Real='{r['clase_real']}' predicho='{r['clase_final']}'"
                })

    # Guardar CSV de inconsistencias/errores
    with open(ruta_csv, mode='w', newline='', encoding='utf-8') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(['id_muestra', 'clase_real', 'clase_predicha', 'clase_mas_cercana',
                         'distancia_minima', 'umbral', 'tipo_evento', 'detalle'])
        for err in errores:
            writer.writerow([
                err['id'],
                err['clase_real'],
                err['clase_final'],
                err['clase_cercana'],
                f"{err['distancia_min']:.4f}",
                umbral_max,
                err['tipo_error'],
                err['detalle']
            ])

    # Generar Texto de Reporte
    lineas_txt = []
    lineas_txt.append("=" * 72)
    lineas_txt.append("REPORTE DE CONCORDANCIA Y EVALUACIÓN BIOMÉTRICA DE AUDIO")
    lineas_txt.append(f"Fecha de ejecución : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lineas_txt.append(f"Umbral de distancia: {umbral_max}")
    lineas_txt.append("=" * 72)

    lineas_txt.append("\n1. RESUMEN DE COMANDOS VÁLIDOS (RECONOCER, IDENTIFICAR, SALIR):")
    lineas_txt.append(f"  - Total muestras evaluadas     : {total_validos}")
    lineas_txt.append(f"  - Clasificados correctamente   : {aciertos_validos} ({exactitud_validos:.2f}%)")
    lineas_txt.append(f"  - Confusiones entre comandos   : {confusiones_validos} ({(confusiones_validos/total_validos*100):.2f}%)")
    lineas_txt.append(f"  - Falsos Rechazos (> umbral)   : {rechazos_falsos_validos} ({frr:.2f}%)")

    lineas_txt.append("\n2. RESUMEN DE AUDIOS NO VÁLIDOS (BASURA / FUERA DE VOCABULARIO):")
    lineas_txt.append(f"  - Total muestras evaluadas     : {total_basura}")
    lineas_txt.append(f"  - Rechazadas correctamente(TRR): {rechazos_correctos_basura} ({trr:.2f}%)")
    lineas_txt.append(f"  - Falsas Aceptaciones    (FAR): {falsas_aceptaciones_basura} ({far:.2f}%)")

    lineas_txt.append("\n3. LISTADO DETALLADO DE ERRORES E INCONSISTENCIAS:")
    if not errores:
        lineas_txt.append("  [EXCELENTE] Ningún error registrado en la base de datos.")
    else:
        lineas_txt.append(f"  Total de casos a revisar: {len(errores)}")
        lineas_txt.append("  " + "-" * 70)
        lineas_txt.append(f"  {'ID Muestra':<20} | {'Real':<11} | {'Predicho':<13} | {'Dist':<6} | {'Tipo':<20}")
        lineas_txt.append("  " + "-" * 70)
        for err in errores:
            lineas_txt.append(
                f"  {err['id']:<20} | {err['clase_real']:<11} | {err['clase_final']:<13} | {err['distancia_min']:<6.2f} | {err['tipo_error']:<20}"
            )
        lineas_txt.append("  " + "-" * 70)

    # Estadísticas de distancias
    dists_v = [r['distancia_min'] for r in res_validos]
    dists_b = [r['distancia_min'] for r in res_basura]
    lineas_txt.append("\n4. ESTADÍSTICAS DE DISTANCIAS EUCLÍDEAS:")
    if dists_v:
        lineas_txt.append(f"  - Comandos válidos : Media={np.mean(dists_v):.2f}, Mín={np.min(dists_v):.2f}, Máx={np.max(dists_v):.2f}, Std={np.std(dists_v):.2f}")
    if dists_b:
        lineas_txt.append(f"  - Audios basura    : Media={np.mean(dists_b):.2f}, Mín={np.min(dists_b):.2f}, Máx={np.max(dists_b):.2f}, Std={np.std(dists_b):.2f}")

    contenido_txt = "\n".join(lineas_txt)

    with open(ruta_txt, mode='w', encoding='utf-8') as f_txt:
        f_txt.write(contenido_txt)

    return contenido_txt, errores


def main():
    parser = argparse.ArgumentParser(
        description="Evalúa la concordancia y falsas aceptaciones en audios con ClasificadorVoz."
    )
    parser.add_argument("--origen", type=str, choices=["dataset", "audios"], default="dataset",
                        help="Fuente de datos: 'dataset' (DatasetVectores/X.npy) o 'audios' (AudiosPersonas/)")
    parser.add_argument("--umbral", type=float, default=UMBRAL_DISTANCIA_DEFECTO,
                        help=f"Umbral de distancia euclídea para descarte de comandos (default: {UMBRAL_DISTANCIA_DEFECTO})")
    parser.add_argument("--modo", type=str, choices=["modelo_guardado", "leave_one_out"], default="modelo_guardado",
                        help="Modo de evaluación: 'modelo_guardado' o 'leave_one_out' (default: modelo_guardado)")
    parser.add_argument("--k", type=int, default=22,
                        help="Valor de K vecinos en caso de usar leave_one_out (default: 22)")
    parser.add_argument("--sin_graficos", action="store_true", default=False,
                        help="No muestra las ventanas de visualización interactivas emergentes")

    args = parser.parse_args()

    print("=================================================================")
    print("      EVALUACIÓN DE CONCORDANCIA Y FALSA ACEPTACIÓN (AUDIO)      ")
    print("=================================================================")
    print(f"Origen de datos            : {args.origen.upper()}")
    print(f"Modo de evaluación         : {args.modo}")
    print(f"Umbral de descarte (dist)  : {args.umbral}")

    # 1. Cargar datos
    print("\n[1/4] Cargando datos de audio...")
    if args.origen == "dataset":
        X, y, metadata = cargar_datos_desde_dataset()
    else:
        print("  -> Extrayendo características directamente desde AudiosPersonas/...")
        X, y, metadata = cargar_datos_desde_archivos_audio()

    print(f"  -> Total de muestras cargadas: {len(X)}")
    clases_detectadas = sorted(list(np.unique(y)))
    # Ordenar asegurando que 'basura' quede al final si existe
    if 'basura' in clases_detectadas:
        clases_detectadas.remove('basura')
        clases_detectadas.append('basura')
    print(f"  -> Clases encontradas: {clases_detectadas}")

    # 2. Cargar modelo si es modo modelo_guardado
    clasificador = None
    if args.modo == "modelo_guardado":
        print("\n[2/4] Cargando modelo entrenado 'modelo_voz.npy'...")
        clasificador = ClasificadorVoz.cargar_modelo(RUTA_MODELO_DEFECTO)
        print(f"  -> Método clasificador : {clasificador.metodo.upper()} (K={clasificador.k})")
        print(f"  -> Clases del modelo   : {clasificador.clases}")
    else:
        print(f"\n[2/4] Configurando Leave-One-Out (K={args.k})...")

    # 3. Evaluar
    print("\n[3/4] Evaluando clasificaciones y distancias...")
    resultados = evaluar_base_datos(
        X, y, metadata,
        clasificador=clasificador,
        modo_eval=args.modo,
        k_vecinos=args.k,
        umbral_max=args.umbral
    )

    # 4. Generar Reportes y Gráficos
    print("\n[4/4] Generando métricas, reportes y figuras...")
    ruta_txt = os.path.join(DIR_ACTUAL, "reporte_concordancia_audio.txt")
    ruta_csv = os.path.join(DIR_ACTUAL, "reporte_errores_audio.csv")
    ruta_png = os.path.join(DIR_ACTUAL, "grafico_concordancia_audio.png")

    reporte_texto, errores = generar_reportes(
        resultados,
        umbral_max=args.umbral,
        ruta_txt=ruta_txt,
        ruta_csv=ruta_csv
    )

    print("\n" + reporte_texto)
    print(f"\n[Guardado] Reporte completo en: {ruta_txt}")
    print(f"[Guardado] Detalle de inconsistencias en: {ruta_csv}")

    # Graficación
    if not args.sin_graficos:
        print("\nDesplegando gráficos (Heatmap de Concordancia, Distancias y PCA 2D)...")
        graficar_concordancia_audio(
            resultados,
            clases_ordenadas=clases_detectadas,
            umbral_max=args.umbral,
            guardar_figura=ruta_png
        )
    else:
        graficar_concordancia_audio(
            resultados,
            clases_ordenadas=clases_detectadas,
            umbral_max=args.umbral,
            guardar_figura=ruta_png
        )


if __name__ == "__main__":
    main()
