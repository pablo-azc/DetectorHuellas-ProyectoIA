import os
import re
import numpy as np

dir_base = r"C:\Users\Pablo\Documents\GitHub\DetectorHuellas-ProyectoIA\NormalizadorAudio\BaseDatosOrdenada"

def parsear_entrada(texto):
    """
    Limpia las llaves, corchetes y saltos de línea de tu formato 
    y convierte los números en un array de NumPy de 1 dimensión.
    """
    texto_limpio = re.sub(r'[\{\}\[\]\n]', ' ', texto)
    valores = [float(x) for x in texto_limpio.split()]
    return np.array(valores)

def calcular_estadisticos_y_mascara():
    # Ajustamos la impresión de NumPy para que sea legible en consola
    np.set_printoptions(precision=3, suppress=True, linewidth=120)

    for etiqueta in os.listdir(dir_base):
        ruta_etiqueta = os.path.join(dir_base, etiqueta)
        
        if not os.path.isdir(ruta_etiqueta):
            continue
            
        ruta_completos = os.path.join(ruta_etiqueta, "datosCompletos.txt")
        if not os.path.exists(ruta_completos):
            continue

        with open(ruta_completos, 'r', encoding='utf-8') as f:
            contenido = f.read()

        bloques = re.findall(r'\[.*?\]', contenido, re.DOTALL)
        matrices = [parsear_entrada(b) for b in bloques]
        
        if not matrices:
            continue
            
        X = np.vstack(matrices)
        
        media = np.mean(X, axis=0)
        varianza = np.var(X, axis=0)
        std = np.std(X, axis=0)
        
        # Máscara basada en 2 desviaciones estándar
        limite_inferior = media - (1.5 * std)
        limite_superior = media + (1.5 * std)
        
        print(f"\n==========================================")
        print(f" ETIQUETA: {etiqueta.upper()} ")
        print(f"==========================================")
        
        print("\n--- VALORES CALCULADOS DE LA MÁSCARA ---")
        print(f"Media:\n{media}")
        print(f"\nVarianza:\n{varianza}")
        print(f"\nLímite Inferior (Media - 2*Std):\n{limite_inferior}")
        print(f"\nLímite Superior (Media + 2*Std):\n{limite_superior}\n")
        
        print("--- PORCENTAJE DE ACEPTACIÓN POR AUDIO ---")
        # Evaluamos muestra por muestra cuánto se ajustan al modelo de su propia clase
        for i, muestra in enumerate(X):
            # Compara el vector de este audio contra los límites superiores e inferiores
            mascara_booleana = (muestra >= limite_inferior) & (muestra <= limite_superior)
            
            # np.mean de un array booleano da la proporción de True's. Al multiplicar por 100 da el porcentaje
            porcentaje_valido = np.mean(mascara_booleana) * 100
            
            print(f"Audio {i+1}:\t{porcentaje_valido:.2f}% de los valores están dentro de la máscara esperada.")

        # Opcional: seguir guardando los arrays por si los necesitas en otro script
        np.save(os.path.join(ruta_etiqueta, "media.npy"), media)
        np.save(os.path.join(ruta_etiqueta, "varianza.npy"), varianza)
        np.save(os.path.join(ruta_etiqueta, "mascara_limite_inf.npy"), limite_inferior)
        np.save(os.path.join(ruta_etiqueta, "mascara_limite_sup.npy"), limite_superior)

if __name__ == "__main__":
    calcular_estadisticos_y_mascara()