import cv2
import numpy as np
import random
from pathlib import Path

def simular_captura_real(
    ruta_imagen: str, 
    ruta_salida: str, 
    angulo_max: float = 45.0, 
    pad_porcentaje: float = 0.35
):
    """
    Toma una imagen de huella dactilar, añade espacio en blanco alrededor
    y aplica una rotación aleatoria entre -angulo_max y +angulo_max.
    """
    ruta_img = Path(ruta_imagen)
    if not ruta_img.exists():
        print(f"Error: No se encontró el archivo {ruta_imagen}")
        return

    # 1. Cargar la imagen
    img = cv2.imread(str(ruta_img))
    h, w = img.shape[:2]

    # 2. Calcular y añadir el borde blanco (Padding)
    pad_h = int(h * pad_porcentaje)
    pad_w = int(w * pad_porcentaje)

    img_padded = cv2.copyMakeBorder(
        img,
        top=pad_h,
        bottom=pad_h,
        left=pad_w,
        right=pad_w,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255] # Blanco
    )

    # 3. Generar un ángulo aleatorio entre -angulo_max y +angulo_max
    angulo = random.uniform(-angulo_max, angulo_max)

    # 4. Obtener centro y matriz de rotación
    hp, wp = img_padded.shape[:2]
    centro = (wp // 2, hp // 2)
    matriz_rotacion = cv2.getRotationMatrix2D(centro, angulo, scale=1.0)

    # 5. Rotar la imagen manteniendo el fondo blanco
    img_rotada = cv2.warpAffine(
        img_padded,
        matriz_rotacion,
        (wp, hp),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )

    # 6. Guardar la imagen transformada
    Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ruta_salida), img_rotada)
    print(f"Procesada: {ruta_salida} | Ángulo aplicado: {angulo:.2f}°")


def procesar_dataset_completo(directorio_origen: str, directorio_destino: str):
    """
    Aplica la transformación a todas las imágenes JPG/PNG dentro de la carpeta del dataset.
    """
    origen = Path(directorio_origen)
    destino = Path(directorio_destino)

    extensiones = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    archivos = []
    for ext in extensiones:
        archivos.extend(origen.rglob(ext))

    print(f"Encontrados {len(archivos)} archivos. Generando transformaciones...")

    for archivo in archivos:
        # Mantener la misma estructura de carpetas
        relativo = archivo.relative_to(origen)
        salida = destino / relativo

        simular_captura_real(
            ruta_imagen=str(archivo),
            ruta_salida=str(salida),
            angulo_max=90.0,
            pad_porcentaje=0.35
        )

if __name__ == "__main__":
    # --- EJEMPLO 1: Para probar con una sola imagen ---
    #simular_captura_real(
    #    ruta_imagen="3_1.jpg", 
    #    ruta_salida="3_1_simulada.jpg", 
    #    angulo_max=90.0
    #)

    # --- EJEMPLO 2: Para procesar todo tu dataset reorganizado ---
    RUTA_ORIGEN = r"VariantesPermitidas_Ordenada\Complejas"
    RUTA_DESTINO = r"VariantesPermitidas_Ordenada\Simuladas"
    procesar_dataset_completo(RUTA_ORIGEN, RUTA_DESTINO)