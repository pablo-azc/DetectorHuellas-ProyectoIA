import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

def cargar_matrices_8x8(archivos):
    """
    Lee archivos de texto que contienen matrices de 8x8 delimitadas por { y }.
    Devuelve:
      - X: arreglo NumPy de dimensiones (N, 64) con las matrices aplanadas.
      - y_real: arreglo NumPy con el índice de clase correspondiente (0, 1, 2, ...).
      - nombres_clases: nombres extraídos de las rutas o carpetas.
    """
    X = []
    y_real = []
    nombres_clases = []

    for idx_clase, ruta in enumerate(archivos):
        nombre_carpeta = os.path.basename(os.path.dirname(ruta))
        if not nombre_carpeta:
            nombre_carpeta = f"Clase_{idx_clase + 1}"
        nombres_clases.append(nombre_carpeta)

        if not os.path.exists(ruta):
            print(f"[ADVERTENCIA] No se encontró el archivo: {ruta}")
            continue

        with open(ruta, 'r', encoding='utf-8') as f:
            contenido = f.read()

        # Extraer bloques delimitados por '{' y '}'
        bloques = contenido.split('{')
        muestras_archivo = 0

        for bloque in bloques:
            if '}' in bloque:
                cuerpo = bloque.split('}')[0].strip()
                lineas = [linea.strip() for linea in cuerpo.split('\n') if linea.strip()]
                
                # Cada matriz de huella debe contener 8 filas
                if len(lineas) == 8:
                    matriz = []
                    valido = True
                    for linea in lineas:
                        valores = [float(val) for val in linea.split()]
                        if len(valores) == 8:
                            matriz.append(valores)
                        else:
                            valido = False
                            break
                    
                    if valido and len(matriz) == 8:
                        # Aplanamos la matriz de 8x8 a un vector característico de 64 dimensiones
                        X.append(np.array(matriz, dtype=np.float64).flatten())
                        y_real.append(idx_clase)
                        muestras_archivo += 1

        print(f"  -> {nombre_carpeta}: {muestras_archivo} huellas cargadas exitosamente.")

    return np.array(X, dtype=np.float64), np.array(y_real, dtype=np.int32), nombres_clases

def kmeans_desde_cero(X, k=3, max_iter=300, tol=1e-4, seed=42):
    """
    Implementación alternativa de K-Means desde cero para vectores D-dimensionales (D=64).
    """
    np.random.seed(seed)
    indices_iniciales = np.random.choice(X.shape[0], k, replace=False)
    centroides = X[indices_iniciales].copy()

    for iteracion in range(max_iter):
        # Distancia euclídea de cada punto a cada centroide en dimensión 64
        distancias = np.linalg.norm(X[:, np.newaxis] - centroides, axis=2)
        etiquetas = np.argmin(distancias, axis=1)

        nuevos_centroides = np.zeros_like(centroides)
        for i in range(k):
            puntos_cluster = X[etiquetas == i]
            if len(puntos_cluster) > 0:
                nuevos_centroides[i] = puntos_cluster.mean(axis=0)
            else:
                nuevos_centroides[i] = X[np.random.choice(X.shape[0])]

        desplazamiento = np.linalg.norm(nuevos_centroides - centroides)
        if desplazamiento < tol:
            print(f"[K-Means Propio] Convergencia alcanzada en la iteración {iteracion + 1}")
            break
        centroides = nuevos_centroides

    return etiquetas, centroides

def graficar_resultados(X, etiquetas, centroides, y_real, nombres_clases, k=3):
    """
    Genera dos figuras visuales:
    1. Proyección 2D mediante PCA mostrando puntos agrupados y centroides.
    2. Mapas de calor (Heatmaps) de 8x8 para los centroides de cada clúster.
    """
    # -------------------------------------------------------------
    # Gráfica 1: Proyección 2D con PCA
    # -------------------------------------------------------------
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    centroides_pca = pca.transform(centroides)
    varianza_exp = pca.explained_variance_ratio_.sum() * 100

    plt.figure(figsize=(9, 6))
    colores = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
    
    for i in range(k):
        puntos = X_pca[etiquetas == i]
        plt.scatter(
            puntos[:, 0], puntos[:, 1],
            label=f'Clúster {i} ({len(puntos)} muestras)',
            alpha=0.7, edgecolors='k', s=50,
            c=colores[i % len(colores)]
        )

    # Dibujar centroides en el espacio PCA
    plt.scatter(
        centroides_pca[:, 0], centroides_pca[:, 1],
        c='red', marker='X', s=220, edgecolors='black', linewidth=1.5,
        label='Centroides'
    )

    plt.title(f'K-Means sobre Matrices de Huella 8x8 (Proyección PCA - {varianza_exp:.1f}% Varianza)', fontsize=12)
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best')
    plt.tight_layout()

    # -------------------------------------------------------------
    # Gráfica 2: Representación Matricial 8x8 de los Centroides
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, k, figsize=(4 * k, 4.5), constrained_layout=True)
    if k == 1:
        axes = [axes]

    for i in range(k):
        # Restaurar el vector del centroide (64,) a matriz (8, 8)
        matriz_centroide = centroides[i].reshape(8, 8)
        ax = axes[i]
        cax = ax.imshow(matriz_centroide, cmap='viridis', interpolation='nearest', vmin=0, vmax=3)
        ax.set_title(f'Centroide Clúster {i}', fontsize=11, fontweight='bold')
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        
        # Superponer los valores numéricos promediados en cada celda
        for r in range(8):
            for c in range(8):
                val = matriz_centroide[r, c]
                color_texto = "white" if val < 1.5 else "black"
                ax.text(c, r, f"{val:.1f}", ha="center", va="center", color=color_texto, fontsize=8)

    fig.colorbar(cax, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1, label='Valor Predominante de Minucia')
    fig.suptitle('Patrones Promedio por Clúster (Matriz 8x8)', fontsize=13, fontweight='bold')

    plt.show()

def main():
    # Directorio base del proyecto
    dir_base = os.path.dirname(os.path.abspath(__file__))
    dir_proyecto = os.path.dirname(dir_base)
    
    # Rutas por defecto a los archivos de porcentajes generados por 4 - Limitador.py
    archivos = [
        os.path.join(dir_proyecto, 'Huellas', 'VariantesPermitidas_Preprocesadas', 'huella_1', 'porcentajes.txt'),
        os.path.join(dir_proyecto, 'Huellas', 'VariantesPermitidas_Preprocesadas', 'huella_2', 'porcentajes.txt'),
        os.path.join(dir_proyecto, 'Huellas', 'VariantesPermitidas_Preprocesadas', 'huella_3', 'porcentajes.txt')
    ]

    print("=========================================================")
    print("      PROCESAMIENTO DE HUELLAS 8x8 CON K-MEANS           ")
    print("=========================================================")
    print("Cargando matrices de huellas desde archivos de texto...")
    X, y_real, nombres_clases = cargar_matrices_8x8(archivos)

    if len(X) == 0:
        print("[ERROR] No se pudo cargar ninguna muestra. Verifique las rutas.")
        return

    print(f"\nTotal de matrices procesadas: {X.shape[0]} (Dimensión de cada huella: {X.shape[1]})")

    k = 3
    print(f"\nEjecutando algoritmo K-Means (k={k})...")
    
    # Usamos scikit-learn con múltiples reinicios (n_init=10) para máxima robustez
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    etiquetas = kmeans.fit_predict(X)
    centroides = kmeans.cluster_centers_

    # Matriz de concordancia / confusión vs etiquetas originales
    matriz_conf = confusion_matrix(y_real, etiquetas)
    print("\n--- MATRIZ DE CONCORDANCIA ---")
    print("Filas = Huellas Originales | Columnas = Clúster Asignado:")
    
    # Mapeo de cada clúster al nombre de clase mayoritaria
    mapa_cluster_clase = {}
    for idx, nombre in enumerate(nombres_clases):
        cluster_predicho = np.argmax(matriz_conf[idx])
        mapa_cluster_clase[int(cluster_predicho)] = nombre
        print(f"  {nombre:10s} -> {matriz_conf[idx]} (Asignado al Clúster {cluster_predicho})")

    # Guardar modelo y centroides para que 6-identificarHuella.py los cargue directamente
    ruta_modelo = os.path.join(dir_base, 'modelo_kmeans.npz')
    np.savez(
        ruta_modelo,
        X=X,
        y_real=y_real,
        etiquetas=etiquetas,
        centroides=centroides,
        nombres_clases=np.array(nombres_clases),
        cluster_a_clase=np.array([mapa_cluster_clase.get(i, f"Clúster_{i}") for i in range(k)])
    )
    print(f"\n[Guardado] Modelo y centroides exportados en: {ruta_modelo}")

    print("\nGenerando gráficos visuales...")
    graficar_resultados(X, etiquetas, centroides, y_real, nombres_clases, k=k)

if __name__ == "__main__":
    main()
