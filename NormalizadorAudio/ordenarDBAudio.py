import os
import shutil
import re

dir_origen = "AudiosPersonas"
dir_destino = "BaseDatosOrdenada"

def leer_datos(ruta):
    """
    Lee el archivo y extrae las entradas.
    Busca específicamente bloques encerrados entre { y } 
    para no romper las matrices/vectores que tienen saltos de línea internos.
    """
    if not os.path.exists(ruta):
        return []
    
    with open(ruta, 'r', encoding='utf-8') as f:
        contenido = f.read()
    
    # re.DOTALL permite que el '.' coincida también con los saltos de línea (\n)
    bloques = re.findall(r'\[.*?\]', contenido, re.DOTALL)
    
    if bloques:
        return bloques
    else:
        # Fallback por si datosResumen.txt no tiene llaves y es solo texto por línea
        return [linea.strip() for linea in contenido.splitlines() if linea.strip()]

def organizar_base_de_datos():
    if os.path.exists(dir_destino):
        shutil.rmtree(dir_destino)
    
    os.makedirs(dir_destino)

    etapas = {
        "ruido": "1_ruido.wav",
        "audioRaw": "2_audioRaw.wav",
        "audioLimpio": "3_audioLimpio.wav",
        "audioCortado": "4_audioCortado.wav"
    }

    for persona in os.listdir(dir_origen):
        ruta_persona = os.path.join(dir_origen, persona)
        
        if not os.path.isdir(ruta_persona):
            continue

        ruta_etiquetas = os.path.join(ruta_persona, "queDeberiaSer.txt")
        ruta_resumen = os.path.join(ruta_persona, "datosResumen.txt")
        ruta_completos = os.path.join(ruta_persona, "datosCompletos.txt")

        if not os.path.exists(ruta_etiquetas):
            continue

        with open(ruta_etiquetas, 'r', encoding='utf-8') as f:
            etiquetas = [linea.strip() for linea in f.read().splitlines() if linea.strip()]

        # Usamos la nueva función para procesar los .txt de datos
        datos_resumen = leer_datos(ruta_resumen)
        datos_completos = leer_datos(ruta_completos)

        for i, etiqueta in enumerate(etiquetas):
            num_audio = i + 1
            ruta_audio_origen = os.path.join(ruta_persona, "Audios", f"Audio{num_audio}")

            if not os.path.exists(ruta_audio_origen):
                continue

            ruta_base_etiqueta = os.path.join(dir_destino, etiqueta)

            for nombre_carpeta in etapas.keys():
                os.makedirs(os.path.join(ruta_base_etiqueta, nombre_carpeta), exist_ok=True)

            nuevo_nombre_audio = f"{persona}_Audio{num_audio}.wav"

            for nombre_carpeta, archivo_origen in etapas.items():
                src_wav = os.path.join(ruta_audio_origen, archivo_origen)
                dst_wav = os.path.join(ruta_base_etiqueta, nombre_carpeta, nuevo_nombre_audio)
                
                if os.path.exists(src_wav):
                    shutil.copy2(src_wav, dst_wav)

            # Escribir la entrada completa de resumen (tal cual estaba originalmente)
            if i < len(datos_resumen):
                with open(os.path.join(ruta_base_etiqueta, "datosResumen.txt"), "a", encoding='utf-8') as f:
                    f.write(datos_resumen[i] + "\n")

            # Escribir el bloque matriz/vector completo 
            if i < len(datos_completos):
                with open(os.path.join(ruta_base_etiqueta, "datosCompletos.txt"), "a", encoding='utf-8') as f:
                    f.write(datos_completos[i] + "\n")

    print(f"¡Listo! Base de datos estructurada en '{dir_destino}'")

if __name__ == "__main__":
    organizar_base_de_datos()