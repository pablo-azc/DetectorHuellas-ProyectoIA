import os
import re
import numpy as np

import CapturaAudio



dir_base = r"C:\Users\Pablo\Documents\GitHub\DetectorHuellas-ProyectoIA\NormalizadorAudio\BaseDatosOrdenada"
clases_esperadas = ["reconocer", "identificar", "salir"]

def parsear_entrada(texto):
    """
    Limpia el formato y convierte el texto en un array de NumPy.
    """
    texto_limpio = re.sub(r'[\{\}\[\]\n]', ' ', texto)
    valores = [float(x) for x in texto_limpio.split()]
    return np.array(valores)

def evaluar_nuevo_audio(texto_nuevo):
    nuevo_vector = parsear_entrada(texto_nuevo)
    
    print(f"Tamaño del vector nuevo: {len(nuevo_vector)} valores.")
    print("-" * 40)
    
    resultados = {}

    for etiqueta in clases_esperadas:
        ruta_inf = os.path.join(dir_base, etiqueta, "mascara_limite_inf.npy")
        ruta_sup = os.path.join(dir_base, etiqueta, "mascara_limite_sup.npy")

        # Verificar que las máscaras existan
        if not (os.path.exists(ruta_inf) and os.path.exists(ruta_sup)):
            print(f"[{etiqueta.upper()}] -> No se encontraron las máscaras calculadas.")
            continue

        # Cargar los límites
        limite_inferior = np.load(ruta_inf)
        limite_superior = np.load(ruta_sup)

        # Verificar que el vector nuevo tenga la misma cantidad de datos que las máscaras
        if len(nuevo_vector) != len(limite_inferior):
            print(f"[{etiqueta.upper()}] -> Error de dimensión. Esperado: {len(limite_inferior)}, Recibido: {len(nuevo_vector)}.")
            continue

        # Aplicar la máscara booleana
        mascara = (nuevo_vector >= limite_inferior) & (nuevo_vector <= limite_superior)
        
        # Calcular porcentaje
        porcentaje_coincidencia = np.mean(mascara) * 100
        resultados[etiqueta] = porcentaje_coincidencia
        
        print(f"[{etiqueta.upper()}] -> {porcentaje_coincidencia:.2f}% de los valores están dentro del rango.")

    print("-" * 40)
    
    # Determinar la clase ganadora
    #if resultados:
    #    clase_ganadora = max(resultados, key=resultados.get)
    #    porcentaje_max = resultados[clase_ganadora]
        
    #    print(f"RESULTADO: El audio se clasifica como '{clase_ganadora.upper()}' ({porcentaje_max:.2f}% de similitud estadística).")
    #else:
    #    print("No se pudo evaluar el audio contra ninguna clase.")

if __name__ == "__main__":
    # Aquí pegas el contenido del archivo datosCompletos de tu audio nuevo.
    # Usé el ejemplo que me pasaste antes para probar.
    dato_prueba = f"{CapturaAudio.capturar_comando_voz()}"
    
    print("Evaluando muestra de prueba...\n")
    evaluar_nuevo_audio(dato_prueba)