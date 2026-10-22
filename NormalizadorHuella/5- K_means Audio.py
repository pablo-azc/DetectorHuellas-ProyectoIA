import numpy as np
import matplotlib.pyplot as plt

def cargar_datos(archivos):
    """Lee los archivos, limpia las llaves y devuelve un arreglo NumPy."""
    datos_completos = []
    for archivo in archivos:
        with open(archivo, 'r') as f:
            for linea in f:
                linea = linea.strip().replace('{', '').replace('}', '')
                if linea:
                    valores = [float(val) for val in linea.split(',')]
                    datos_completos.append(valores)
    return np.array(datos_completos)

def kmeans_desde_cero(X, k=3, max_iter=300, tol=1e-4):
    """
    Implementación del algoritmo K-Means desde cero.
    """
    # 1. Inicialización: Elegir 'k' puntos aleatorios del conjunto de datos como centroides iniciales
    np.random.seed(42) # Fijamos semilla para resultados reproducibles
    indices_iniciales = np.random.choice(X.shape[0], k, replace=False)
    centroides = X[indices_iniciales]
    
    for iteracion in range(max_iter):
        # 2. Paso de asignación: Calcular la distancia euclídea de cada punto a cada centroide
        # X[:, np.newaxis] tiene forma (N, 1, 3) y centroides (k, 3) -> resultado (N, k, 3)
        distancias = np.linalg.norm(X[:, np.newaxis] - centroides, axis=2)
        
        # Asignar cada punto al centroide más cercano
        etiquetas = np.argmin(distancias, axis=1)
        
        # 3. Paso de actualización: Recalcular la posición de cada centroide como el promedio
        nuevos_centroides = np.zeros_like(centroides)
        for i in range(k):
            puntos_del_cluster = X[etiquetas == i]
            if len(puntos_del_cluster) > 0:
                nuevos_centroides[i] = puntos_del_cluster.mean(axis=0)
            else:
                # Si un clúster se queda vacío, reasignamos su centroide a un punto aleatorio
                nuevos_centroides[i] = X[np.random.choice(X.shape[0])]
        
        # 4. Criterio de parada: Verificar si los centroides dejaron de moverse
        desplazamiento = np.linalg.norm(nuevos_centroides - centroides)
        if desplazamiento < tol:
            print(f"Convergencia alcanzada en la iteración {iteracion + 1}")
            break
            
        centroides = nuevos_centroides
        
    return etiquetas, centroides

def main():
    archivos = [r'C:\Users\Pablo\Documents\GitHub\DetectorHuellas-ProyectoIA\Huellas\VariantesPermitidas_Preprocesadas\huella_1\porcentajes.txt', 
                r'C:\Users\Pablo\Documents\GitHub\DetectorHuellas-ProyectoIA\Huellas\VariantesPermitidas_Preprocesadas\huella_2\porcentajes.txt', 
                r'C:\Users\Pablo\Documents\GitHub\DetectorHuellas-ProyectoIA\Huellas\VariantesPermitidas_Preprocesadas\huella_3\porcentajes.txt']
    
    print("Cargando datos...")
    X = cargar_datos(archivos)
    
    # Aplicar K-Means personalizado
    k = 3
    print(f"Ejecutando K-Means propio con {k} clústeres...")
    etiquetas, centroides = kmeans_desde_cero(X, k=k)
    
    # Graficar en 3D
    print("Generando gráfica 3D...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Dispersión de puntos coloreados por su clúster
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], 
                         c=etiquetas, cmap='viridis', marker='o', alpha=0.6)
    
    # Marcación de centroides
    ax.scatter(centroides[:, 0], centroides[:, 1], centroides[:, 2], 
               c='red', s=200, marker='X', label='Centroides')
    
    # Ajuste estricto de los límites de los ejes al rango real de los datos
    ax.set_xlim([X[:, 0].min(), X[:, 0].max()])
    ax.set_ylim([X[:, 1].min(), X[:, 1].max()])
    ax.set_zlim([X[:, 2].min(), X[:, 2].max()])
    
    # Configuración de etiquetas de la gráfica
    ax.set_xlabel('Eje X')
    ax.set_ylabel('Eje Y')
    ax.set_zlabel('Eje Z')
    ax.set_title('K-Means Algoritmo Propio (Espacio 3D)')
    
    plt.colorbar(scatter, label='Clúster asignado')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()