import os
import numpy as np

class ClasificadorVoz:
    """
    Clasificador auto-desarrollado para comandos de voz.
    Implementa dos métodos nativos en NumPy:
    1. Distancia Euclidiana Estandarizada a Centroides (por defecto).
    2. K-Nearest Neighbors (KNN) nativo.
    """
    def __init__(self, metodo="centroides", k=3):
        self.metodo = metodo
        self.k = k
        self.clases = None
        self.centroides = {}
        self.desviaciones = {}
        self.mean_global = None
        self.std_global = None
        self.X_train = None
        self.X_train_scaled = None
        self.y_train = None

    def entrenar(self, X, y):
        self.X_train = np.array(X, dtype=float)
        self.y_train = np.array(y)
        self.clases = np.unique(self.y_train)

        # 1. NORMALIZACIÓN GLOBAL PARA EL CLASIFICADOR
        self.mean_global = np.mean(self.X_train, axis=0)
        self.std_global = np.std(self.X_train, axis=0)
        self.std_global = np.where(self.std_global < 1e-6, 1.0, self.std_global) # Evitar div 0
        
        # Escalar el X_train completo (solo necesario para KNN, pero útil en general)
        self.X_train_scaled = (self.X_train - self.mean_global) / self.std_global

        # 2. Cálculo de centroides sobre datos escalados (opcional, pero mejora)
        for c in self.clases:
            X_c = self.X_train_scaled[self.y_train == c]
            self.centroides[c] = np.mean(X_c, axis=0)
            std_c = np.std(X_c, axis=0)
            self.desviaciones[c] = np.where(std_c < 1e-6, 1.0, std_c)

    def predecir_uno(self, vector):
        vector = np.array(vector, dtype=float)
        # 3. ESCALAR EL VECTOR ENTRANTE
        vector_scaled = (vector - self.mean_global) / self.std_global

        if self.metodo == "centroides":
            distancias = {}
            for c in self.clases:
                diff = (vector_scaled - self.centroides[c]) / self.desviaciones[c]
                dist = np.sqrt(np.sum(diff ** 2))
                distancias[str(c)] = float(dist)
            clase_predicha = min(distancias, key=distancias.get)
            return clase_predicha, distancias

        elif self.metodo == "knn":
            # Usar X_train_scaled y vector_scaled
            dists = np.sqrt(np.sum((self.X_train_scaled - vector_scaled) ** 2, axis=1))
            k_indices = np.argsort(dists)[:self.k]
            # ... el resto sigue igual
            k_labels = self.y_train[k_indices]

            # Votación por mayoría
            valores, conteos = np.unique(k_labels, return_counts=True)
            clase_predicha = str(valores[np.argmax(conteos)])
            
            # Calculamos la distancia media a los k vecinos por cada clase
            distancias = {}
            for c in self.clases:
                mask_c = (self.y_train == c)
                dists_c = dists[mask_c]
                distancias[str(c)] = float(np.min(dists_c)) if len(dists_c) > 0 else float('inf')

            return clase_predicha, distancias
        else:
            raise ValueError(f"Método desconocido: {self.metodo}")

    def predecir(self, X):
        predicciones = []
        for x in X:
            clase, _ = self.predecir_uno(x)
            predicciones.append(clase)
        return np.array(predicciones)

    def guardar_modelo(self, ruta_archivo):
        datos = {
            "metodo": self.metodo,
            "k": self.k,
            "clases": self.clases,
            "centroides": self.centroides,
            "desviaciones": self.desviaciones,
            "mean_global": self.mean_global,
            "std_global": self.std_global,
            "X_train": self.X_train,
            "X_train_scaled": self.X_train_scaled,
            "y_train": self.y_train
        }
        np.save(ruta_archivo, datos, allow_pickle=True)

    @classmethod
    def cargar_modelo(cls, ruta_archivo):
        datos = np.load(ruta_archivo, allow_pickle=True).item()
        clasificador = cls(metodo=datos["metodo"], k=datos.get("k", 3))
        clasificador.clases = datos["clases"]
        clasificador.centroides = datos["centroides"]
        clasificador.desviaciones = datos["desviaciones"]
        clasificador.mean_global = datos.get("mean_global", None)
        clasificador.std_global = datos.get("std_global", None)
        clasificador.X_train = datos.get("X_train", None)
        clasificador.X_train_scaled = datos.get("X_train_scaled", None)
        clasificador.y_train = datos["y_train"]
        return clasificador


def evaluar_clasificador(K_vecinos=3,guardar=True):
    dir_dataset = os.path.join(os.path.dirname(__file__), "DatasetVectores")
    X = np.load(os.path.join(dir_dataset, "X.npy"))
    y = np.load(os.path.join(dir_dataset, "y.npy"))
    metadata = np.load(os.path.join(dir_dataset, "metadata.npy"))

    # Filtrar muestras que no pertenezcan a los tres comandos principales
    mascara_validos = np.isin(y, ["reconocer", "identificar", "salir"])
    X = X[mascara_validos]
    y = y[mascara_validos]
    metadata = metadata[mascara_validos]

    print("=====================================================")
    print(" EVALUACIÓN DE CLASIFICADOR (Leave-One-Out y Cross-Val)")
    print(f" Total de muestras: {len(X)}")
    print("=====================================================\n")

    for metodo in ["knn"]:
        clf = ClasificadorVoz(metodo=metodo, k=K_vecinos)
        aciertos = 0
        total = len(X)
        errores = []

        # Validación cruzada Leave-One-Out (entrenar con todos menos 1)
        for i in range(total):
            X_train = np.delete(X, i, axis=0)
            y_train = np.delete(y, i, axis=0)

            clf.entrenar(X_train, y_train)
            pred, info = clf.predecir_uno(X[i])

            if pred == y[i]:
                aciertos += 1
            else:
                errores.append((metadata[i], y[i], pred, info))

        exactitud = (aciertos / total) * 100
        print(f"Método '{metodo.upper()}': {aciertos}/{total} correctos ({exactitud:.2f}% de precisión)")

        if errores:
            print(f"  Detalle de errores ({len(errores)}):")
            for m, real, pred, info in errores[:5]: # Mostrar los primeros 5
                print(f"    - {m}: Real='{real}' -> Predicho='{pred}'")
            if len(errores) > 5:
                print(f"    ... y {len(errores)-5} más.")
        print()

    # Entrenar modelo final con todos los datos y guardarlo
    modelo_final = ClasificadorVoz(metodo="knn",k=K_vecinos)
    modelo_final.entrenar(X, y)
    if guardar:
        ruta_guardado = os.path.join(dir_dataset, "modelo_voz.npy")
        modelo_final.guardar_modelo(ruta_guardado)
        print(f"Modelo final guardado en: {ruta_guardado}")
    return exactitud

if __name__ == "__main__":
    grafico=open("plotKnn.csv","w")
    for i in range(1,30):
        grafico.write(f"{i};\t{evaluar_clasificador(K_vecinos=i,guardar=False)}\n")
    grafico.close()
    evaluar_clasificador(K_vecinos=1)


