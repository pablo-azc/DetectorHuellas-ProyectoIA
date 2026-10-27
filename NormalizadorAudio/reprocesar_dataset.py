import os
import soundfile as sf
import numpy as np
from ProcesadorAudio import extraer_vector_ventanas

dir_origen = os.path.join(os.path.dirname(__file__), "AudiosPersonas")
dir_salida = os.path.join(os.path.dirname(__file__), "DatasetVectores")

def reprocesar_dataset(top_db=20, n_ventanas=3, duracion_ventana=0.25):
    """
    Recorre AudiosPersonas, lee queDeberiaSer.txt de cada persona,
    toma 3_audioLimpio.wav de cada AudioX, extrae el nuevo vector de características
    (3 ventanas de 0.25s tras trim) y guarda el dataset consolidado.
    """
    os.makedirs(dir_salida, exist_ok=True)
    
    X = []
    y = []
    metadata = []
    
    print("Iniciando reprocesamiento de AudiosPersonas...")
    
    carpetas_personas = sorted(os.listdir(dir_origen))
    for persona in carpetas_personas:
        ruta_persona = os.path.join(dir_origen, persona)
        if not os.path.isdir(ruta_persona):
            continue
            
        ruta_etiquetas = os.path.join(ruta_persona, "queDeberiaSer.txt")
        if not os.path.exists(ruta_etiquetas):
            print(f"[-] Saltando {persona}: no tiene queDeberiaSer.txt")
            continue
            
        with open(ruta_etiquetas, 'r', encoding='utf-8') as f:
            etiquetas = [linea.strip().lower() for linea in f.read().splitlines() if linea.strip()]
            
        conteo_persona = 0
        for i, etiqueta in enumerate(etiquetas):
            num_audio = i + 1
            ruta_audio_limpio = os.path.join(ruta_persona, "Audios", f"Audio{num_audio}", "3_audioLimpio.wav")
            
            if not os.path.exists(ruta_audio_limpio):
                continue
                
            try:
                audio, fs = sf.read(ruta_audio_limpio)
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
                    conteo_persona += 1
                else:
                    print(f"    [!] Advertencia: {persona}_Audio{num_audio} no produjo vector suficiente.")
            except Exception as e:
                print(f"    [!] Error al procesar {ruta_audio_limpio}: {e}")
                
        print(f"[+] {persona}: {conteo_persona} audios procesados correctamente.")
        
    X = np.array(X)
    y = np.array(y)
    metadata = np.array(metadata)
    
    print("\n--- Resumen del Dataset ---")
    print(f"Total de muestras procesadas: {len(X)}")
    print(f"Dimensión de cada vector: {X.shape[1] if len(X) > 0 else 0}")
    
    clases, counts = np.unique(y, return_counts=True)
    for c, cnt in zip(clases, counts):
        print(f"  - Clase '{c}': {cnt} muestras")
        
    # Guardar en DatasetVectores
    np.save(os.path.join(dir_salida, "X.npy"), X)
    np.save(os.path.join(dir_salida, "y.npy"), y)
    np.save(os.path.join(dir_salida, "metadata.npy"), metadata)
    
    # También guardar un archivo CSV / de texto legible para inspección
    ruta_txt = os.path.join(dir_salida, "dataset_resumen.csv")
    with open(ruta_txt, "w", encoding="utf-8") as f:
        f.write("id,clase," + ",".join([f"f{i+1}" for i in range(X.shape[1])]) + "\n")
        for meta, etiqueta, vec in zip(metadata, y, X):
            vec_str = ",".join([f"{v:.4f}" for v in vec])
            f.write(f"{meta},{etiqueta},{vec_str}\n")
            
    print(f"\nArchivos guardados en: {dir_salida}")

if __name__ == "__main__":
    reprocesar_dataset(top_db=20)

