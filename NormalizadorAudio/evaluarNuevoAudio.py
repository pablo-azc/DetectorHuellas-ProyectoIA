import os
import sys
import numpy as np

# Asegurar path local
sys.path.append(os.path.dirname(__file__))

import CapturaAudio
from ClasificadorVoz import ClasificadorVoz

dir_dataset = os.path.join(os.path.dirname(__file__), "DatasetVectores")
ruta_modelo = os.path.join(dir_dataset, "modelo_voz.npy")

def clasificar_vector(vector):
    """
    Recibe un vector de características (39 valores) y devuelve la predicción.
    """
    if not os.path.exists(ruta_modelo):
        print(f"Error: No se encontró el modelo entrenado en '{ruta_modelo}'.")
        print("Ejecuta 'ClasificadorVoz.py' para entrenar y generar el modelo.")
        return None, None

    clasificador = ClasificadorVoz.cargar_modelo(ruta_modelo)
    clase_predicha, distancias = clasificador.predecir_uno(vector)
    
    return clase_predicha, distancias

def evaluar_en_vivo():
    """
    Captura audio del micrófono y lo clasifica en tiempo real.
    """
    print("Iniciando captura de audio...")
    vector = CapturaAudio.capturar_comando_voz(grabar=True, carpeta_grabar="Audios/EnVivo")
    
    if vector is None:
        print("No se pudo obtener un vector válido de audio.")
        return

    clase, distancias = clasificar_vector(vector)
    
    print("\n" + "=" * 45)
    print(" RESULTADO DE LA CLASIFICACIÓN")
    print("=" * 45)
    print(f"Comando detectado: >>> {clase.upper()} <<<")
    print("\nDistancias relativas a cada clase (menor es más cercano):")
    for c, dist in distancias.items():
        print(f"  - {c.capitalize():<12}: {dist:.2f}")
    print("=" * 45 + "\n")

if __name__ == "__main__":
    evaluar_en_vivo()
