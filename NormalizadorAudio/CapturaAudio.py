import numpy as np
import sounddevice as sd
import librosa
import noisereduce as nr
import time
import os
import soundfile as sf
from ProcesadorAudio import extraer_vector_ventanas

def capturar_comando_voz(duracion_total=3, duracion_ruido=1, countdown=3, fs=16000, 
                          grabar=True, carpeta_grabar="Audios", ruidoPostrior=True,
                          modo_debug=False, callback_estado=None):
    """
    Captura audio del micrófono, reduce el ruido ambiental, recorta silencios 
    y devuelve un vector de características basado en 3 ventanas de 0.25s (MFCCs).
    
    :param modo_debug: Si True, imprime mensajes concisos tipo '[DEBUG] ...'.
    :param callback_estado: Función opcional callback(estado_str) para notificar a una UI gráfica.
    """
    def notificar(msg_consola, estado_ui=None):
        if callback_estado and estado_ui:
            callback_estado(estado_ui)
        if msg_consola:
            print(msg_consola)

    if grabar:
        os.makedirs(carpeta_grabar, exist_ok=True)

    if modo_debug:
        notificar("[DEBUG] Preparando captura...", "¡Prepárate!")
    else:
        notificar("\n--- Preparando micrófono ---\nPreparese para algun comando ('Reconocer', 'Identificar' o 'Salir')...", "¡Prepárate!")

    for i in range(countdown, duracion_ruido, -1):
        if modo_debug:
            notificar(f"[DEBUG] Cuenta regresiva: {i}", f"¡Prepárate! ({i}...)")
        else:
            notificar(f"{i}...", f"¡Prepárate! ({i}...)")
        time.sleep(1)

    if modo_debug:
        notificar("[DEBUG] Grabando Silencio (Perfil de Ruido)...", f"¡Prepárate! ({duracion_ruido}...)")
    else:
        notificar(f"{duracion_ruido}...", f"¡Prepárate! ({duracion_ruido}...)")

    # Capturando perfil de ruido ambiental
    ruido = sd.rec(int(duracion_ruido * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    
    # 1. Captura de Audio
    if modo_debug:
        notificar("[DEBUG] Grabando Audio (Comando de voz)...", "🔴 GRABANDO COMANDO...")
    else:
        notificar("🔴 GRABANDO: ('Reconocer', 'Identificar' o 'Salir')...", "🔴 GRABANDO COMANDO...")

    grabacion = sd.rec(int(duracion_total * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()

    if ruidoPostrior:
        if modo_debug:
            notificar("[DEBUG] Grabando Silencio posterior...", "Analizando audio...")
        ruido2 = sd.rec(int(duracion_ruido * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        ruido = ruido + ruido2

    if modo_debug:
        notificar("[DEBUG] Grabación finalizada. Procesando...", "Procesando audio...")
    else:
        notificar("✅ Grabación finalizada.", "Procesando audio...")

    # 2. Sustracción Espectral de Ruido
    audio_limpio = nr.reduce_noise(y=grabacion.flatten(), sr=fs, y_noise=ruido.flatten())

    if grabar:
        sf.write(os.path.join(carpeta_grabar, "1_ruido.wav"), ruido, fs)
        sf.write(os.path.join(carpeta_grabar, "2_audioRaw.wav"), grabacion, fs)
        sf.write(os.path.join(carpeta_grabar, "3_audioLimpio.wav"), audio_limpio, fs)

    # 3. Extracción de características por ventanas (3 ventanas de 0.25s tras trim)
    vector_caracteristicas = extraer_vector_ventanas(
        audio_limpio=audio_limpio, 
        fs=fs, 
        n_ventanas=3, 
        duracion_ventana=0.25, 
        top_db=20, 
        n_mfcc=13
    )

    if vector_caracteristicas is None:
        print("⚠️ Advertencia: No se detectó voz o el comando es demasiado corto.")
        return None

    # Guardar audio recortado por referencia si se desea
    if grabar:
        audio_recortado, _ = librosa.effects.trim(audio_limpio, top_db=20)
        sf.write(os.path.join(carpeta_grabar, "4_audioCortado.wav"), audio_recortado, fs)

    return vector_caracteristicas

# --- Bloque de prueba ---
if __name__ == "__main__":
    temp=input()
    c=0
    #datosCompleto=open("datosCompletos.txt","w")
    #datosResumen=open("datosResumen.txt","w")
    while (temp != "salir"):
        c+=1
        vector = capturar_comando_voz(carpeta_grabar=f"Audios/Audio{c}")
        if vector is not None:
            print("\nExtracción exitosa.")
            print(f"Dimensión del vector para K-NN: {vector.shape}")
            #print("vector entero", vector)
            #datosCompleto.write(f"{np.round(vector, 3)}\n")
            #datosResumen.write(f"{np.round(vector[:5], 3)}\n")
            #print("Primeros 5 valores del vector:", np.round(vector[:5], 3))
        else:
            print("\nExtracción Fallida. probá hablar mas fuerte")
        temp=input("Presione Enter cuando esté listo para otra sesión:")