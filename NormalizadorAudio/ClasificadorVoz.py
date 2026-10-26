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
        self.X_train = None
        self.y_train = None

    def entrenar(self, X, y):
        """
        Ajusta el modelo con los datos de entrenamiento X e y.
        """
        self.X_train = np.array(X, dtype=float)
        self.y_train = np.array(y)
        self.clases = np.unique(self.y_train)

        # Cálculo de centroides (media) y desviaciones por clase
        for c in self.clases:
            X_c = self.X_train[self.y_train == c]
            self.centroides[c] = np.mean(X_c, axis=0)
            # Desviación con un epsilon pequeño para evitar divisiones por cero
            std_c = np.std(X_c, axis=0)
            self.desviaciones[c] = np.where(std_c < 1e-6, 1.0, std_c)

    def predecir_uno(self, vector):
        """
        Clasifica un vector individual (1D).
        Devuelve (clase_predicha, dict_de_distancias_o_puntuaciones).
        """
        vector = np.array(vector, dtype=float)

        if self.metodo == "centroides":
            distancias = {}
            for c in self.clases:
                # Distancia euclídea normalizada (Mahalanobis diagonal)
                diff = (vector - self.centroides[c]) / self.desviaciones[c]
                dist = np.sqrt(np.sum(diff ** 2))
                distancias[c] = float(dist)

            # La clase con menor distancia euclídea gana
            clase_predicha = min(distancias, key=distancias.get)
            return clase_predicha, distancias

        elif self.metodo == "knn":
            # Distancia euclídea simple a cada punto del conjunto de entrenamiento
            dists = np.sqrt(np.sum((self.X_train - vector) ** 2, axis=1))
            k_indices = np.argsort(dists)[:self.k]
            k_labels = self.y_train[k_indices]

            # Votación por mayoría
            valores, conteos = np.unique(k_labels, return_counts=True)
            clase_predicha = valores[np.argmax(conteos)]
            
            resumen_votos = {c: int(np.sum(k_labels == c)) for c in self.clases}
            return clase_predicha, resumen_votos
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
            "X_train": self.X_train,
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
        clasificador.X_train = datos["X_train"]
        clasificador.y_train = datos["y_train"]
        return clasificador


def evaluar_clasificador():
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

    for metodo in ["centroides", "knn"]:
        clf = ClasificadorVoz(metodo=metodo, k=3)
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
    modelo_final = ClasificadorVoz(metodo="knn")
    modelo_final.entrenar(X, y)
    ruta_guardado = os.path.join(dir_dataset, "modelo_voz.npy")
    modelo_final.guardar_modelo(ruta_guardado)
    print(f"Modelo final guardado en: {ruta_guardado}")

if __name__ == "__main__":
    evaluar_clasificador()

