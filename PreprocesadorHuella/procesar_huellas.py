
"""
Procesa huellas dactilares simuladas (crestas oscuras sobre fondo claro):
  1. Segmenta la huella
  2. La rota para dejar su eje mayor vertical (sin cortar nada)
  3. La recorta con un margen
  4. Aplica un suavizado leve
  5. Aplica un "sharpening" (unsharp mask)

Uso:
    python procesar_huellas.py entrada.png salida/
    python procesar_huellas.py carpeta_entrada/ carpeta_salida/ --size 400
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# --------------------------------------------------------------------------
# Lectura
# --------------------------------------------------------------------------
def leer_gris(path):
    """Lee una imagen como escala de grises uint8; si tiene alpha, compone sobre blanco."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"No se pudo leer {path}")
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb = img[:, :, :3].astype(np.float32)
        img = (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


# --------------------------------------------------------------------------
# Segmentación
# --------------------------------------------------------------------------
def mascara_huella(gray, close_size=None):
    """Devuelve una máscara binaria (0/255) con la región ocupada por la huella."""
    h, w = gray.shape
    if close_size is None:
        close_size = max(9, int(0.035 * min(h, w)))
    close_size |= 1  # impar

    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Si "lo oscuro" ocupa más de la mitad, la imagen estaba invertida
    if np.count_nonzero(bw) > 0.5 * bw.size:
        bw = cv2.bitwise_not(bw)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    mask = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == idx).astype(np.uint8) * 255


# --------------------------------------------------------------------------
# Rotación
# --------------------------------------------------------------------------
def angulo_para_vertical(mask):
    """Ángulo (grados, antihorario en pantalla) que hay que rotar para que el eje mayor quede vertical."""
    ys, xs = np.nonzero(mask)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    pts -= pts.mean(axis=0)
    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    vx, vy = vecs[:, np.argmax(vals)]
    theta = -np.degrees(np.arctan2(vy, vx))      # ángulo del eje mayor (y hacia abajo)
    delta = 90.0 - theta
    delta = (delta + 90.0) % 180.0 - 90.0        # rotación mínima, en [-90, 90)
    return delta


def matriz_rotacion_expandida(shape, angulo):
    """Matriz de rotación + tamaño de lienzo que contiene la imagen rotada completa."""
    h, w = shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angulo, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw = int(np.ceil(h * sin + w * cos))
    nh = int(np.ceil(h * cos + w * sin))
    M[0, 2] += nw / 2.0 - cx
    M[1, 2] += nh / 2.0 - cy
    return M, (nw, nh)


# --------------------------------------------------------------------------
# Recorte
# --------------------------------------------------------------------------
def recortar(gray, mask, margen=0.05):
    x, y, w, h = cv2.boundingRect(mask)
    m = int(round(margen * max(w, h)))
    # Padding blanco para que el margen nunca se salga de la imagen
    gray_p = cv2.copyMakeBorder(gray, m, m, m, m, cv2.BORDER_CONSTANT, value=255)
    x0, y0 = x, y  # tras el padding, (x, y) original pasa a (x+m, y+m); restamos m => igual
    return gray_p[y0:y0 + h + 2 * m, x0:x0 + w + 2 * m]


def a_cuadrado(gray, size):
    """Centra en un lienzo cuadrado blanco (sin deformar) y escala a size x size."""
    h, w = gray.shape
    lado = max(h, w)
    top = (lado - h) // 2
    left = (lado - w) // 2
    canvas = np.full((lado, lado), 255, np.uint8)
    canvas[top:top + h, left:left + w] = gray
    interp = cv2.INTER_AREA if size < lado else cv2.INTER_CUBIC
    return cv2.resize(canvas, (size, size), interpolation=interp)


# --------------------------------------------------------------------------
# Filtros
# --------------------------------------------------------------------------
def suavizar(gray, sigma):
    if sigma <= 0:
        return gray
    return cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)


def unsharp(gray, sigma, amount):
    """out = (1+a)*img - a*blur(img)"""
    if amount <= 0:
        return gray
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
    return cv2.addWeighted(gray, 1.0 + amount, blur, -amount, 0)  # saturate_cast a uint8


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
def procesar(gray, args):
    mask = mascara_huella(gray)
    if mask is None:
        raise ValueError("no se detectó ninguna huella")

    if not args.no_rotar:
        ang = angulo_para_vertical(mask)
        M, dsize = matriz_rotacion_expandida(gray.shape, ang)
        gray = cv2.warpAffine(gray, M, dsize, flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        mask = cv2.warpAffine(mask, M, dsize, flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    out = recortar(gray, mask, args.margen)
    out = suavizar(out, args.suavizado)
    out = unsharp(out, args.sharp_sigma, args.sharp_cantidad)

    if args.size:
        out = a_cuadrado(out, args.size)
    return out


def main():
    ap = argparse.ArgumentParser(description="Rota, recorta, suaviza y afila huellas.")
    ap.add_argument("entrada", type=Path, help="Imagen o carpeta de imágenes")
    ap.add_argument("salida", type=Path, help="Carpeta de salida")
    ap.add_argument("--margen", type=float, default=0.05,
                    help="Margen del recorte como fracción del tamaño de la huella (def. 0.05)")
    ap.add_argument("--suavizado", type=float, default=0.8,
                    help="Sigma del Gaussiano de suavizado; 0 = sin suavizado (def. 0.8)")
    ap.add_argument("--sharp-sigma", type=float, default=1.5,
                    help="Sigma del desenfoque base del unsharp mask (def. 1.5)")
    ap.add_argument("--sharp-cantidad", type=float, default=1.5,
                    help="Intensidad del sharpening; 0 = sin sharpening (def. 1.5)")
    ap.add_argument("--size", type=int, default=0,
                    help="Si se indica, salida cuadrada size x size con fondo blanco")
    ap.add_argument("--no-rotar", action="store_true", help="No rotar la huella")
    args = ap.parse_args()

    if args.entrada.is_dir():
        archivos = sorted(p for p in args.entrada.iterdir() if p.suffix.lower() in EXTS)
    else:
        archivos = [args.entrada]

    args.salida.mkdir(parents=True, exist_ok=True)
    ok = 0
    for p in archivos:
        try:
            res = procesar(leer_gris(p), args)
            cv2.imwrite(str(args.salida / (p.stem + ".png")), res)
            ok += 1
        except Exception as e:
            print(f"[!] {p.name}: {e}")
    print(f"Listo: {ok}/{len(archivos)} imágenes procesadas -> {args.salida}")


if __name__ == "__main__":
    main()
