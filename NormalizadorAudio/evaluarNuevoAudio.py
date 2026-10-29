import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Asegurar path local
sys.path.append(os.path.dirname(__file__))

import CapturaAudio
from ClasificadorVoz import ClasificadorVoz

dir_dataset = os.path.join(os.path.dirname(__file__), "DatasetVectores")
ruta_modelo = os.path.join(dir_dataset, "modelo_voz.npy")
ruta_X = os.path.join(dir_dataset, "X.npy")
ruta_y = os.path.join(dir_dataset, "y.npy")

# Umbral máximo de distancia para considerar válido el comando.
# Si la distancia mínima supera este valor, se descarta como "NO_RECONOCIDO".
UMBRAL_DISTANCIA_MAX = 8.5

def clasificar_vector(vector, umbral_max=UMBRAL_DISTANCIA_MAX):
    """
    Recibe un vector de características (39 valores) y devuelve:
    (clase_predicha, distancias, es_valido)
    Si la menor distancia supera umbral_max, clase_predicha es 'NO_RECONOCIDO' y es_valido es False.
    """
    if not os.path.exists(ruta_modelo):
        print(f"Error: No se encontró el modelo entrenado en '{ruta_modelo}'.")
        print("Ejecuta 'ClasificadorVoz.py' para entrenar y generar el modelo.")
        return None, None, False

    clasificador = ClasificadorVoz.cargar_modelo(ruta_modelo)
    clase_predicha, distancias = clasificador.predecir_uno(vector)
    
    distancia_minima = distancias[clase_predicha]
    es_valido = True

    if umbral_max is not None and distancia_minima > umbral_max:
        es_valido = False
        clase_predicha = "NO_RECONOCIDO"
        
    return clase_predicha, distancias, es_valido

def graficar_pca(vector_nuevo=None, etiqueta_predicha=None):
    """
    Proyecta el conjunto de entrenamiento y los centroides en 3D usando PCA (SVD nativo),
    y ubica el nuevo audio capturado para visualizar a qué clase se acerca más.
    Permite rotación interactiva 3D con el cursor del mouse.
    """
    if not (os.path.exists(ruta_X) and os.path.exists(ruta_y)):
        print("No se encontraron los datos del dataset para graficar el PCA.")
        return

    X = np.load(ruta_X)
    y = np.load(ruta_y)

    # Filtrar sólo las tres clases objetivo
    mascara = np.isin(y, ["reconocer", "identificar", "salir"])
    X = X[mascara]
    y = y[mascara]

    # Ajustar PCA con el dataset (centrado respecto a la media global)
    media_global = np.mean(X, axis=0)
    X_centrado = X - media_global
    _, _, Vt = np.linalg.svd(X_centrado, full_matrices=False)
    W = Vt[:3].T  # Matriz de proyección a 3D (39 x 3)

    X_pca = np.dot(X_centrado, W)

    colores = {
        "reconocer": "#1f77b4",    # Azul
        "identificar": "#2ca02c",  # Verde
        "salir": "#ff7f0e"         # Naranja
    }

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Graficar puntos de cada clase
    for clase, color in colores.items():
        puntos_clase = X_pca[y == clase]
        ax.scatter(
            puntos_clase[:, 0], puntos_clase[:, 1], puntos_clase[:, 2],
            c=color, label=f"Clase: {clase.capitalize()}",
            alpha=0.6, edgecolors='none', s=45
        )

        # Graficar centroide proyectado
        centroide_clase = np.mean(X[y == clase], axis=0)
        centroide_pca = np.dot(centroide_clase - media_global, W)
        ax.scatter(
            centroide_pca[0], centroide_pca[1], centroide_pca[2],
            c=color, marker='X', s=180, edgecolors='black',
            linewidths=1.5, label=f"Centroide {clase.capitalize()}"
        )

    # Si se pasó un vector nuevo, proyectarlo y destacarlo
    if vector_nuevo is not None:
        nuevo_centrado = vector_nuevo - media_global
        nuevo_pca = np.dot(nuevo_centrado, W)

        color_punto = colores.get(etiqueta_predicha, "#d62728")
        marcador = '*' if etiqueta_predicha != "NO_RECONOCIDO" else 'P'
        texto_label = f"Nuevo audio ({etiqueta_predicha.upper()})"

        ax.scatter(
            nuevo_pca[0], nuevo_pca[1], nuevo_pca[2],
            c=color_punto, marker=marcador, s=280, edgecolors='black',
            linewidths=2, label=texto_label, zorder=10
        )
        ax.text(
            nuevo_pca[0], nuevo_pca[1], nuevo_pca[2],
            f" Nuevo ({etiqueta_predicha})",
            fontweight='bold', fontsize=11, color='darkred'
        )

    ax.set_title("Mapa PCA 3D de Comandos de Voz", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("PC 1", fontsize=11, labelpad=10)
    ax.set_ylabel("PC 2", fontsize=11, labelpad=10)
    ax.set_zlabel("PC 3", fontsize=11, labelpad=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    plt.tight_layout()
    plt.show()

def evaluar_en_vivo(umbral_max=UMBRAL_DISTANCIA_MAX, duracion_ruido=1, mostrar_grafico=True):
    """
    Captura audio del micrófono y lo clasifica en tiempo real.
    Si mostrar_grafico es True, abre la proyección PCA 3D del nuevo audio.
    """
    print("Iniciando captura de audio...")
    vector = CapturaAudio.capturar_comando_voz(
        grabar=True, 
        carpeta_grabar="Audios/EnVivo", 
        duracion_ruido=duracion_ruido
    )
    
    if vector is None:
        print("No se pudo obtener un vector válido de audio.")
        return

    clase, distancias, es_valido = clasificar_vector(vector, umbral_max=umbral_max)
    
    print("\n" + "=" * 45)
    print(" RESULTADO DE LA CLASIFICACIÓN")
    print("=" * 45)
    
    print("\nDistancias relativas a cada clase (menor es más cercano):")
    for c, dist in distancias.items():
        print(f"  - {c.capitalize():<12}: {dist:.2f}")
    print(f"\nUmbral de rechazo configurado: {umbral_max}")
    print("=" * 45 + "\n")

    if es_valido:
        print(f"Comando detectado: >>> {clase.upper()} <<<")
    else:
        min_dist = min(distancias.values()) if distancias else float('inf')
        print(f"Comando: >>> NO RECONOCIDO <<< (Distancia mínima {min_dist:.2f} > umbral {umbral_max})")
    

    if mostrar_grafico:
        print("Generando visualización PCA 3D...")
        graficar_pca(vector_nuevo=vector, etiqueta_predicha=clase)

if __name__ == "__main__":
    while(input()!="salir"):
        evaluar_en_vivo(mostrar_grafico=True)
