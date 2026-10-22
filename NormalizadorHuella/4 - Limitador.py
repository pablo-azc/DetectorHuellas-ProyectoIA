import cv2
import numpy as np
import os
import glob

def maximoInternoEnHuella(lista):
    maximo=0
    ganador=0
    for i in range(3):
        if lista[i]>maximo:
            maximo=lista[i]
            ganador=i+1
    return ganador


def extraer_minucias_directorio(dir_entrada, dir_salida,archivoProm):
    """
    Procesa un directorio de huellas, identifica terminaciones y bifurcaciones,
    y guarda el resultado coloreado en un directorio de salida.
    """
    if not os.path.exists(dir_salida):
        os.makedirs(dir_salida)

    # Buscar imágenes PNG y JPG en el directorio de origen
    extensiones = ('*.png', '*.jpg', '*.jpeg')
    archivos = []
    for ext in extensiones:
        archivos.extend(glob.glob(os.path.join(dir_entrada, ext)))
        archivos.extend(glob.glob(os.path.join(dir_entrada, ext.upper())))

    if not archivos:
        print(f"No se encontraron imágenes en el directorio: {dir_entrada}")
        return

    # Definir el kernel de 3x3 para contar pixeles vecinos en el esqueleto
    kernel_izquierdo = np.array([[1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]], dtype=np.uint8)
    kernel_derecho = np.array([[0, 0, 1],
                                [0, 1, 0],
                                [1, 0, 0]], dtype=np.uint8)
    kernel_Horizontal   = np.array([[0, 0, 0],
                                    [1, 1, 1],
                                    [0, 0, 0]], dtype=np.uint8)

    for ruta in archivos:
        nombre_archivo = os.path.basename(ruta)
        print(f"Procesando: {nombre_archivo}...")

        # 1. Lectura en escala de grises
        img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # 2. Binarización
        # Asumiendo que la huella (como 3_1.png) tiene crestas oscuras y fondo claro,
        # usamos THRESH_BINARY_INV para que las crestas sean blancas (255) y el fondo negro (0).
        _, binarizada = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # 3. Esqueletización (Thinning)
        # Reduce el grosor de las líneas a 1 solo pixel para poder contar vecinos.
        esqueleto = cv2.ximgproc.thinning(binarizada)

        # 4. Cálculo de intersecciones (Crossing Number)
        # Normalizamos a 1 y 0 para la convolución matemática
        esqueleto_norm = esqueleto // 255 
        
        # Aplicamos el filtro 2D. Esto suma los valores de los 8 vecinos inmediatos de cada pixel.
        conteo_derecho = cv2.filter2D(esqueleto_norm, -1, kernel_derecho)
        conteo_izquierdo = cv2.filter2D(esqueleto_norm, -1, kernel_izquierdo)
        conteo_horiz = cv2.filter2D(esqueleto_norm, -1, kernel_Horizontal)

        # Lógica de identificación (Crossing Number sobre el esqueleto):
        # - si concuerda exactamente con el planteo, se considera correcto
        derechos = (esqueleto_norm == 1) & (conteo_derecho == 2)
        izquierdo = (esqueleto_norm == 1) & (conteo_izquierdo == 2)
        horizonales = (esqueleto_norm == 1) & (conteo_horiz == 3)
        totales = (esqueleto_norm == 1)

        # 5. Dibujo visual de resultados
        # Convertimos la imagen original a BGR para poder usar colores
        resultado_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Obtener coordenadas de las matrices booleanas
        y_d, x_d = np.where(derechos)
        y_i, x_i = np.where(izquierdo)
        y_h, x_h = np.where(horizonales)
        y_t, x_t = np.where(totales)

        ##Divide la huella en una matriz de 8 x 8
        limite_y_min=y_t.min()
        limite_y_max=y_t.max()
        limite_x_min=x_t.min()
        limite_x_max=x_t.max()
        deltay=int((limite_y_max-limite_y_min)/8)
        deltax=int((limite_x_max-limite_x_min)/8)

        ##Matriz: huella[0][0]=[cuentaDer,cuentaIzq,cuentaHor]
        huella=[]
        for indice_x_huella in range(8):
            huella.append([])
            for indice_y_huella in range(8):
                huella[indice_x_huella].append([])
                huella[indice_x_huella][indice_y_huella]=[0,0,0]

        # Dibujar Derecha (Fines de línea) en ROJO
        for x, y in zip(x_d, y_d):
            cv2.circle(resultado_color, (x, y), 3, (0, 0, 255), -1)
            for indice_x_huella in range(8):
                if (x>limite_x_min+(indice_x_huella)*deltax) and (x<limite_x_min+(indice_x_huella+1)*deltax):
                    for indice_y_huella in range(8):
                        if (y>limite_y_min+(indice_y_huella)*deltay) and (y<limite_y_min+(indice_y_huella+1)*deltay):
                            huella[indice_x_huella][indice_y_huella][0] += 1

        # Dibujar Bifurcaciones (Uniones) en AZUL
        for x, y in zip(x_i, y_i):
            cv2.circle(resultado_color, (x, y), 3, (255, 0, 0), -1)
            for indice_x_huella in range(8):
                if (x>limite_x_min+(indice_x_huella)*deltax) and (x<limite_x_min+(indice_x_huella+1)*deltax):
                    for indice_y_huella in range(8):
                        if (y>limite_y_min+(indice_y_huella)*deltay) and (y<limite_y_min+(indice_y_huella+1)*deltay):
                            huella[indice_x_huella][indice_y_huella][1] += 1

        # Dibujar Bifurcaciones (Uniones) en AZUL
        for x, y in zip(x_h, y_h):
            cv2.circle(resultado_color, (x, y), 3, (0, 255, 0), -1)
            for indice_x_huella in range(8):
                if (x>limite_x_min+(indice_x_huella)*deltax) and (x<limite_x_min+(indice_x_huella+1)*deltax):
                    for indice_y_huella in range(8):
                        if (y>limite_y_min+(indice_y_huella)*deltay) and (y<limite_y_min+(indice_y_huella+1)*deltay):
                            huella[indice_x_huella][indice_y_huella][2] += 1



        # Guardar imagen resultante
        ruta_guardado = os.path.join(dir_salida, f"minucias_{nombre_archivo}")
        cv2.imwrite(ruta_guardado, resultado_color)
        print(f"  -> Guardado en: {ruta_guardado}")
        archivoProm.write("{\n")
        for indice_y_huella in range(8):
            for indice_x_huella in range(8):
                archivoProm.write(f"{maximoInternoEnHuella(huella[indice_x_huella][indice_y_huella])}\t")
            archivoProm.write("\n")
        archivoProm.write("\n}\n")
    archivoProm.close()

if __name__ == '__main__':
    # Configura aquí tus directorios relativos o absolutos
    directorio_origen = r"C:\Users\Pablo\Documents\GitHub\DetectorHuellas-ProyectoIA\Huellas\VariantesPermitidas_Tomadas\huella_"
    directorio_destino = r"C:\Users\Pablo\Documents\GitHub\DetectorHuellas-ProyectoIA\Huellas\VariantesPermitidas_Preprocesadas\huella_"
    
    extraer_minucias_directorio(directorio_origen+"1", directorio_destino+"1",open(directorio_destino+"1/porcentajes.txt","w"))
    extraer_minucias_directorio(directorio_origen+"2", directorio_destino+"2",open(directorio_destino+"2/porcentajes.txt","w"))
    extraer_minucias_directorio(directorio_origen+"3", directorio_destino+"3",open(directorio_destino+"3/porcentajes.txt","w"))