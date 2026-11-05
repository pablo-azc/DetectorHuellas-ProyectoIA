import numpy as np
import librosa

def extraer_vector_ventanas(audio_limpio, fs=16000, n_ventanas=3, duracion_ventana=0.25, top_db=20, n_mfcc=13):
    """
    Recorta el audio limpio con top_db=20 para detectar el inicio de la voz,
    toma las primeras n_ventanas (por defecto 3 de 0.25s c/u = 0.75s en total),
    y extrae los coeficientes MFCC promedio para cada una de las ventanas.
    
    Devuelve un vector NumPy 1D de tamaño (n_ventanas * n_mfcc) = 39 dimensiones.
    """
    # 1. Recorte con top_db=20 (detecta inicio de voz)
    audio_recortado, _ = librosa.effects.trim(audio_limpio, top_db=top_db)
    
    # Verificación mínima de presencia de audio
    if len(audio_recortado) < int(fs * 0.1):
        return None
        
    muestras_por_ventana = int(duracion_ventana * fs)
    total_muestras_esperadas = muestras_por_ventana * n_ventanas
    
    # 2. Tomar las primeras n_ventanas (con padding con ceros al final si fue más corto)
    if len(audio_recortado) < total_muestras_esperadas:
        audio_segmentos = np.pad(audio_recortado, (0, total_muestras_esperadas - len(audio_recortado)), mode='constant')
    else:
        audio_segmentos = audio_recortado[:total_muestras_esperadas]
        
    # 3. Extracción de MFCCs por cada ventana
    vectores_sub = []
    for i in range(n_ventanas):
        inicio = i * muestras_por_ventana
        fin = inicio + muestras_por_ventana
        sub_audio = audio_segmentos[inicio:fin]
        
        # Extracción de MFCCs para la sub-ventana
        mfcc = librosa.feature.mfcc(
            y=sub_audio, 
            sr=fs, 
            n_mfcc=n_mfcc, 
            n_fft=512, 
            hop_length=160
        )
        # Media de cada coeficiente en el tiempo de la ventana
        mfcc_media = np.mean(mfcc, axis=1)
        vectores_sub.append(mfcc_media)

    #Generacion del espectro total
    # 1. Calcular la energía RMS por ventana
    rms = librosa.feature.rms(y=audio_recortado, frame_length=512, hop_length=160)

    # 2. Obtener el índice del frame con máxima energía
    frame_max_energia = np.argmax(rms[0])

    # 3. Convertir el frame a tiempo (segundos) o muestras
    porcentajeMax= (frame_max_energia)*160/len(audio_recortado)
    vectores_sub.append(np.array([porcentajeMax]))
        
    # Vector concatenado resultante (ej. 39 dimensiones)
    vector_final = np.concatenate(vectores_sub)
    return vector_final
