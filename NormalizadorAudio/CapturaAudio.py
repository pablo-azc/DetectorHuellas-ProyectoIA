import numpy as np
import sounddevice as sd
import librosa
import noisereduce as nr
import time
import os
import soundfile as sf

def capturar_comando_voz(duracion_total=3.5, duracion_ruido=2, countdown=3, fs=16000, grabar=True, carpeta_grabar="Audios"):
    """
    Captura audio del micrófono, reduce el ruido ambiental, recorta silencios 
    y devuelve un vector de características unidimensional basado en MFCC.
    """

    if grabar:
        os.makedirs(carpeta_grabar, exist_ok=True)

    print("\n--- Preparando micrófono ---")
    print("Preparese para algun comando ('Reconocer', 'Identificar' o 'Salir')...")
    for i in range(countdown,countdown-duracion_ruido,-1):
        time.sleep(1) 
        print(f"{i}...")

    print(f"{countdown-duracion_ruido}...")
    #print("[DEBUG] Capturando Ruido")
    ruido = sd.rec(int(duracion_ruido * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    # 1. Captura de Audio
    print("🔴 GRABANDO: ('Reconocer', 'Identificar' o 'Salir')...")
    grabacion = sd.rec(int(duracion_total * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    #conjunto=ruido+grabacion
    #audio = conjunto.flatten()
    print("✅ Grabación finalizada.")

    

    # 2. Sustracción Espectral de Ruido
    # Tomamos la primera fracción del audio como perfil de ruido ambiental
    #muestras_ruido = int(duracion_ruido * fs)
    #perfil_ruido = audio[:muestras_ruido]
    
    # noisereduce aplica FFT, calcula la densidad del ruido y la resta del espectrograma
    audio_limpio = nr.reduce_noise(y=grabacion.flatten(), sr=fs, y_noise=ruido)

    if grabar:
        sf.write(os.path.join(carpeta_grabar, "1_ruido.wav"), ruido, fs)
        sf.write(os.path.join(carpeta_grabar, "2_audioRaw.wav"), grabacion, fs)
        sf.write(os.path.join(carpeta_grabar, "3_audioLimpio.wav"), audio_limpio, fs)

    # 3. Recorte de Silencios (Voice Activity Detection)
    # top_db define el umbral (en dB) por debajo del cual se considera silencio
    audio_recortado, _ = librosa.effects.trim(audio_limpio, top_db=20)
    
    # Verificación de seguridad: si el recorte borró todo el audio (ej. no se habló)
    if len(audio_recortado) < (fs * 0.2):
        print("⚠️ Advertencia: No se detectó voz o el comando es demasiado corto.")
        return None

    if grabar:
            sf.write(os.path.join(carpeta_grabar, "4_audioCortado.wav"), audio_recortado, fs)

    # 4. Extracción de MFCCs
    # Se obtienen 13 coeficientes Mel, lo estándar para voz humana
    mfccs = librosa.feature.mfcc(y=audio_recortado, sr=fs, n_mfcc=13, n_fft=1024, hop_length=256)

    # 5. Aplanado para K-NN (Vector de longitud fija)
    # Calculamos la media y la desviación estándar temporal de cada coeficiente
    mfccs_media = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    
    # Vector final de 26 dimensiones (13 medias + 13 desviaciones)
    vector_caracteristicas = np.concatenate((mfccs_media, mfccs_std))

    return vector_caracteristicas

# --- Bloque de prueba ---
if __name__ == "__main__":
    temp=input()
    c=0
    datosCompleto=open("datosCompletos.txt","w")
    datosResumen=open("datosResumen.txt","w")
    while (temp != "salir"):
        c+=1
        vector = capturar_comando_voz(carpeta_grabar=f"Audios/Audio{c}")
        if vector is not None:
            print("\nExtracción exitosa.")
            print(f"Dimensión del vector para K-NN: {vector.shape}")
            #print("vector entero", vector)
            datosCompleto.write(f"{np.round(vector, 3)}\n")
            datosResumen.write(f"{np.round(vector[:5], 3)}\n")
            #print("Primeros 5 valores del vector:", np.round(vector[:5], 3))
        else:
            print("\nExtracción Fallida. probá hablar mas fuerte")
        temp=input("Presione Enter cuando esté listo para otra sesión:")