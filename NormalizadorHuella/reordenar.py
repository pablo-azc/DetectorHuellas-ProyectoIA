import os
import shutil
from pathlib import Path

def reorganizar_anguli_por_nombre_archivo(
    origen_path: str, 
    destino_path: str = None, 
    mover: bool = False, 
    prefijo_huella: str = "fp_"  # Cambia a "huella_" si prefieres ese prefijo
):
    """
    Reorganiza el dataset de Anguli extrayendo el ID del sujeto desde el nombre del archivo.
    
    Origen:  Impression_1 / fp_1 / 1.jpg, 2.jpg
    Destino: fp_1 / Impression_1 / 1.jpg
             fp_2 / Impression_1 / 2.jpg
    """
    origen = Path(origen_path).resolve()
    
    if not origen.exists():
        print(f"Error: La carpeta de origen '{origen}' no existe.")
        return

    # Si no se define destino, crea una carpeta '_reorganizado'
    if destino_path is None:
        destino = origen.parent / f"{origen.name}_reorganizado"
    else:
        destino = Path(destino_path).resolve()

    print(f"Carpeta Origen:  {origen}")
    print(f"Carpeta Destino: {destino}")
    print(f"Modo: {'MOVER' if mover else 'COPIAR (Recomendado)'}\n")

    archivos_procesados = 0
    numero_iteracion=0
    # 1. Recorrer las carpetas de impresión (ej. Impression_1, Impression_2)
    for imp_dir in origen.iterdir():
        if not imp_dir.is_dir():
            continue

        nombre_impresion = imp_dir.name
        
        # 2. Recorrer la carpeta interna (ej. fp_1)
        for sub_dir in imp_dir.iterdir():
            if sub_dir.is_dir():
                
                # 3. Recorrer cada archivo (ej. 1.jpg, 1.min, 2.jpg)
                for archivo in sub_dir.iterdir():
                    if archivo.is_file():
                        # Obtener el número/ID del nombre del archivo sin extensión (ej: "1" de "1.jpg" o "1.min")
                        id_huella = archivo.stem

                        # Validar si es un archivo numérico de huella
                        if id_huella.isdigit():
                            nombre_huella = f"{prefijo_huella}{id_huella}"
                        else:
                            nombre_huella = "otros_archivos"

                        # Ruta de destino: Destino / fp_I / Impression_N
                        nueva_carpeta = destino / nombre_huella #/ nombre_impresion
                        nueva_carpeta.mkdir(parents=True, exist_ok=True)
 
                        

                        if mover:
                            shutil.move(str(archivo), str(nueva_carpeta) + "/" + str(numero_iteracion) +"_" + str(archivo.name))
                        else:
                            shutil.copy2(str(archivo), str(nueva_carpeta) + "/" + str(numero_iteracion) +"_" + str(archivo.name))

                        print(str(nueva_carpeta) + "/" + str(numero_iteracion) +"_" + str(archivo.name))

                        archivos_procesados += 1
                        numero_iteracion +=1

    print(f"¡Estructura reorganizada con éxito!")
    print(f"Total de archivos procesados: {archivos_procesados}")
    print(f"Nuevo dataset disponible en: {destino}")

if __name__ == "__main__":
    # -------------------------------------------------------------
    # CONFIGURACIÓN
    # -------------------------------------------------------------
    RUTA_ORIGEN = r"VariantesPermitidas/Complejas"
    
    # Deja None para crear una carpeta paralela con el dataset limpio
    RUTA_DESTINO = r"VariantesPermitidas_Ordenada/Complejas" 
    
    # 'fp_' resultará en carpetas "fp_1", "fp_2"... 
    # Puedes cambiarlo por "huella_" si lo prefieres
    PREFIJO = "huella_"
    
    # Cambia a True si deseas MOVER los archivos en lugar de COPIAR
    MOVER = False
    # -------------------------------------------------------------

    reorganizar_anguli_por_nombre_archivo(
        RUTA_ORIGEN, 
        RUTA_DESTINO, 
        mover=MOVER, 
        prefijo_huella=PREFIJO
    )