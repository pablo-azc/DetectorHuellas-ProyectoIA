import os
import sys
import threading
import subprocess
import glob
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk

# Configurar rutas para módulos hermanos
directorio_raiz = os.path.dirname(os.path.abspath(__file__))
sys.path.append(directorio_raiz)
sys.path.append(os.path.join(directorio_raiz, "NormalizadorAudio"))
sys.path.append(os.path.join(directorio_raiz, "NormalizadorHuella"))

import CapturaAudio
from evaluarNuevoAudio import clasificar_vector
import importlib
import importlib.util
identificar_huella_modulo = importlib.import_module("NormalizadorHuella.6-identificarHuella")

# El archivo "2 - simular_captura.py" tiene espacios en el nombre → cargarlo por ruta
_sc_path = os.path.join(directorio_raiz, "NormalizadorHuella", "2 - simular_captura.py")
_sc_spec = importlib.util.spec_from_file_location("simular_captura", _sc_path)
simular_captura_modulo = importlib.util.module_from_spec(_sc_spec)
_sc_spec.loader.exec_module(simular_captura_modulo)

# Rutas de Anguli
ANGULI_DIR         = r"C:\Users\Pablo\Documents\Proyecto IA\Anguli-MinGW"
ANGULI_EXE         = os.path.join(ANGULI_DIR, "Anguli.exe")
ANGULI_OUTDIR_PERM = "VariantesTemporales"      # 3 huellas fijas (permitidas)
ANGULI_OUTDIR_RAND = "VariantesTemporalesNoP"   # 1 huella aleatoria (no permitida)
ANGULI_SEED_PERM   = "763982"                   # semilla fija para las 3 identidades

class SistemaBiometricoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema Biométrico: Reconocimiento Huella + Voz")
        self.root.geometry("980x680")
        self.root.minsize(880, 600)
        self.root.configure(bg="#1e1e24")

        # Variables de estado
        self.ruta_huella = None
        self.resultado_huella = None
        self.grabando_audio = False

        # Configurar estilos ttk
        self.configurar_estilos()

        # Construir interfaz gráfica
        self.construir_interfaz()

        # Atajo de teclado: ENTER inicia la grabación de comando
        self.root.bind("<Return>", lambda event: self.iniciar_grabacion_hilo())

    def configurar_estilos(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background="#1e1e24")
        style.configure("Card.TFrame", background="#2b2d42", relief="ridge")
        style.configure("Header.TLabel", background="#1e1e24", foreground="#edf2f4", font=("Segoe UI", 16, "bold"))
        style.configure("SubHeader.TLabel", background="#2b2d42", foreground="#8d99ae", font=("Segoe UI", 10))
        style.configure("Info.TLabel", background="#2b2d42", foreground="#edf2f4", font=("Segoe UI", 11))
        style.configure("Status.TLabel", background="#2b2d42", foreground="#e29578", font=("Segoe UI", 13, "bold"))

    def construir_interfaz(self):
        # 1. Encabezado principal
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill="x", padx=20, pady=(15, 10))

        titulo = ttk.Label(header_frame, text="🛡️ Control Biométrico Dactilar y Comandos por Voz", style="Header.TLabel")
        titulo.pack(side="left")

        subtitulo = ttk.Label(
            header_frame, 
            text="Comandos disponibles: 'Reconocer' | 'Identificar' | 'Salir'", 
            font=("Segoe UI", 10, "italic"), 
            background="#1e1e24", 
            foreground="#8d99ae"
        )
        subtitulo.pack(side="right", pady=5)

        # 2. Contenedor central (Panel Huella + Panel Voz)
        panel_central = ttk.Frame(self.root)
        panel_central.pack(fill="both", expand=True, padx=20, pady=10)

        # --- PANEL IZQUIERDO: HUELLA DACTILAR ---
        card_huella = ttk.Frame(panel_central, style="Card.TFrame")
        card_huella.pack(side="left", fill="both", expand=True, padx=(0, 10))

        lbl_huella_titulo = ttk.Label(card_huella, text="1. Captura de Huella Dactilar", style="Info.TLabel", font=("Segoe UI", 12, "bold"))
        lbl_huella_titulo.pack(anchor="w", padx=15, pady=(15, 5))

        btn_cargar = tk.Button(
            card_huella, text="📁 Cargar Imagen de Huella", font=("Segoe UI", 10, "bold"),
            bg="#3a86ff", fg="white", activebackground="#2a75e6", activeforeground="white",
            relief="flat", padx=15, pady=6, cursor="hand2", command=self.cargar_huella
        )
        btn_cargar.pack(fill="x", padx=15, pady=(8, 2))

        # --- Sección Anguli: título + 4 botones ---
        lbl_anguli_sep = tk.Label(
            card_huella, text="─── Generar con Anguli ───",
            bg="#2b2d42", fg="#c77dff", font=("Segoe UI", 8, "bold")
        )
        lbl_anguli_sep.pack(fill="x", padx=15, pady=(2, 0))

        frame_anguli_btns = tk.Frame(card_huella, bg="#2b2d42")
        frame_anguli_btns.pack(fill="x", padx=15, pady=(4, 10))

        # Los primeros 3 son huellas permitidas (fp_1, fp_2, fp_3); el 4to es aleatoria (None)
        BTN_DEFS = [
            ("🧬 Huella 1", "#1b4332", "#145a32", 1),
            ("🧬 Huella 2", "#1b4332", "#145a32", 2),
            ("🧬 Huella 3", "#1b4332", "#145a32", 3),
            ("🎲 Aleatoria", "#7d3c00", "#9c4900", None),
        ]

        self._btns_anguli = []
        for texto, bg, abg, fp_idx in BTN_DEFS:
            btn = tk.Button(
                frame_anguli_btns,
                text=texto, font=("Segoe UI", 9, "bold"),
                bg=bg, fg="white", activebackground=abg, activeforeground="white",
                relief="flat", pady=5, cursor="hand2",
                command=lambda idx=fp_idx: self.generar_huella_anguli(idx)
            )
            btn.pack(side="left", fill="x", expand=True, padx=2)
            self._btns_anguli.append(btn)

        # Visualizadores de imagen (Original y Normalizada)
        frame_imagenes = tk.Frame(card_huella, bg="#2b2d42")
        frame_imagenes.pack(fill="both", expand=True, padx=15, pady=5)

        # Vista Huella Original
        frame_img_orig = tk.Frame(frame_imagenes, bg="#1e1e24", bd=1, relief="solid")
        frame_img_orig.pack(side="left", fill="both", expand=True, padx=(0, 5), pady=5)
        lbl_t1 = tk.Label(frame_img_orig, text="Original", bg="#1e1e24", fg="#8d99ae", font=("Segoe UI", 9))
        lbl_t1.pack(anchor="n", pady=2)
        self.lbl_canvas_orig = tk.Label(frame_img_orig, text="Sin imagen", bg="#1e1e24", fg="#5c677d")
        self.lbl_canvas_orig.pack(fill="both", expand=True)

        # Vista Huella Normalizada
        frame_img_norm = tk.Frame(frame_imagenes, bg="#1e1e24", bd=1, relief="solid")
        frame_img_norm.pack(side="right", fill="both", expand=True, padx=(5, 0), pady=5)
        lbl_t2 = tk.Label(frame_img_norm, text="Normalizada (200x200)", bg="#1e1e24", fg="#8d99ae", font=("Segoe UI", 9))
        lbl_t2.pack(anchor="n", pady=2)
        self.lbl_canvas_norm = tk.Label(frame_img_norm, text="Sin procesar", bg="#1e1e24", fg="#5c677d")
        self.lbl_canvas_norm.pack(fill="both", expand=True)

        self.lbl_info_huella = tk.Label(
            card_huella, text="Ninguna huella cargada actualmente.",
            bg="#2b2d42", fg="#8d99ae", font=("Segoe UI", 9), anchor="w"
        )
        self.lbl_info_huella.pack(fill="x", padx=15, pady=(5, 12))

        # --- PANEL DERECHO: CONTROL POR VOZ ---
        card_voz = ttk.Frame(panel_central, style="Card.TFrame")
        card_voz.pack(side="right", fill="both", expand=True, padx=(10, 0))

        lbl_voz_titulo = ttk.Label(card_voz, text="2. Control por Comando de Voz", style="Info.TLabel", font=("Segoe UI", 12, "bold"))
        lbl_voz_titulo.pack(anchor="w", padx=15, pady=(15, 5))

        lbl_ayuda = tk.Label(
            card_voz,
            text="Hacé clic en el botón o presioná la tecla [ENTER] para hablar.\n"
                 "El sistema grabará el perfil de silencio y luego tu comando.",
            bg="#2b2d42", fg="#8d99ae", font=("Segoe UI", 9), justify="left"
        )
        lbl_ayuda.pack(anchor="w", padx=15, pady=5)

        # Estado dinámico de captura
        self.frame_estado = tk.Frame(card_voz, bg="#1e1e24", bd=1, relief="groove")
        self.frame_estado.pack(fill="x", padx=15, pady=10)

        self.lbl_estado_icon = tk.Label(self.frame_estado, text="🎙️", font=("Segoe UI", 24), bg="#1e1e24", fg="#edf2f4")
        self.lbl_estado_icon.pack(side="left", padx=15, pady=10)

        self.lbl_estado_texto = tk.Label(
            self.frame_estado, text="Micrófono en espera...",
            font=("Segoe UI", 12, "bold"), bg="#1e1e24", fg="#edf2f4"
        )
        self.lbl_estado_texto.pack(side="left", fill="x", expand=True, padx=5, pady=10)

        # Botón grande de grabación
        self.btn_grabar = tk.Button(
            card_voz, text="🎙️ Iniciar Grabación (o presione ENTER)", font=("Segoe UI", 12, "bold"),
            bg="#e63946", fg="white", activebackground="#c92a37", activeforeground="white",
            relief="flat", padx=20, pady=12, cursor="hand2", command=self.iniciar_grabacion_hilo
        )
        self.btn_grabar.pack(fill="x", padx=15, pady=15)

        # Desglose de distancias de voz
        frame_distancias = tk.LabelFrame(card_voz, text="Distancias de Audio Calculadas", bg="#2b2d42", fg="#8d99ae", font=("Segoe UI", 9))
        frame_distancias.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        self.lbl_dist_rec = tk.Label(frame_distancias, text="Reconocer   : --", bg="#2b2d42", fg="#edf2f4", font=("Consolas", 10), anchor="w")
        self.lbl_dist_rec.pack(fill="x", padx=10, pady=2)
        self.lbl_dist_ide = tk.Label(frame_distancias, text="Identificar : --", bg="#2b2d42", fg="#edf2f4", font=("Consolas", 10), anchor="w")
        self.lbl_dist_ide.pack(fill="x", padx=10, pady=2)
        self.lbl_dist_sal = tk.Label(frame_distancias, text="Salir       : --", bg="#2b2d42", fg="#edf2f4", font=("Consolas", 10), anchor="w")
        self.lbl_dist_sal.pack(fill="x", padx=10, pady=2)

        # 3. PANEL INFERIOR: RESULTADO BIOMÉTRICO FINAL
        self.frame_resultado = tk.Frame(self.root, bg="#2b2d42", bd=2, relief="groove")
        self.frame_resultado.pack(fill="x", padx=20, pady=(5, 15))

        self.lbl_res_titulo = tk.Label(
            self.frame_resultado, text="RESULTADO DEL SISTEMA",
            font=("Segoe UI", 10, "bold"), bg="#2b2d42", fg="#8d99ae"
        )
        self.lbl_res_titulo.pack(anchor="w", padx=20, pady=(8, 2))

        self.lbl_res_mensaje = tk.Label(
            self.frame_resultado, text="Cargue una huella y presione Enter para hablar.",
            font=("Segoe UI", 16, "bold"), bg="#2b2d42", fg="#edf2f4"
        )
        self.lbl_res_mensaje.pack(anchor="w", padx=20, pady=(2, 10))

    def generar_huella_anguli(self, fp_idx):
        """
        Genera una huella con Anguli y la carga automáticamente.
        fp_idx = 1, 2 o 3  →  huella permitida (fp_X/1.jpg, seed fija)
        fp_idx = None       →  huella aleatoria (seed random, carpeta NoP)
        """
        self._set_btns_anguli(False)
        modo = f"Huella {fp_idx}" if fp_idx else "Aleatoria"
        self.lbl_info_huella.config(text=f"⏳ Ejecutando Anguli ({modo}), espere...")
        self.root.update_idletasks()
        hilo = threading.Thread(
            target=self._hilo_generar_anguli, args=(fp_idx,), daemon=True
        )
        hilo.start()

    def _set_btns_anguli(self, habilitado: bool):
        """Habilita o deshabilita todos los botones de Anguli."""
        state = "normal" if habilitado else "disabled"
        for btn in self._btns_anguli:
            btn.config(state=state)

    def _hilo_generar_anguli(self, fp_idx):
        import random as _random
        try:
            if fp_idx is not None:
                # ── Huella permitida: 3 identidades, seed fija, tomar fp_{fp_idx}/1.jpg ──
                outdir_rel = ANGULI_OUTDIR_PERM
                outdir_abs = os.path.join(ANGULI_DIR, outdir_rel)
                cmd = [
                    ANGULI_EXE,
                    "-num", "3",
                    "-ni",  "1",
                    "-outdir", outdir_rel,
                    "-trans", "20",
                    "-rot",   "15",
                    "-seed",  ANGULI_SEED_PERM,
                ]
                fp_carpeta = f"fp_{fp_idx}"
                patron = os.path.join(outdir_abs, "Impression_1", "fp_1", f"{fp_idx}.jpg")
            else:
                # ── Huella aleatoria: 1 identidad, seed random, tomar fp_1/1.jpg ──
                outdir_rel = ANGULI_OUTDIR_RAND
                outdir_abs = os.path.join(ANGULI_DIR, outdir_rel)
                semilla = str(_random.randint(100000, 999999))
                cmd = [
                    ANGULI_EXE,
                    "-num", "1",
                    "-ni",  "1",
                    "-outdir", outdir_rel,
                    "-trans", "20",
                    "-rot",   "15",
                    "-seed",  semilla,
                ]
                fp_carpeta = "fp_1"
                patron = os.path.join(outdir_abs, "Impression_1", fp_carpeta, "*.jpg")

            resultado = subprocess.run(
                cmd, cwd=ANGULI_DIR,
                capture_output=True, text=True, timeout=60,
            )

            if resultado.returncode != 0:
                stderr = resultado.stderr.strip() or resultado.stdout.strip()
                self.root.after(0, lambda: messagebox.showerror(
                    "Error de Anguli", f"Anguli terminó con error:\n{stderr}"
                ))
                self.root.after(0, lambda: self._set_btns_anguli(True))
                return

            archivos = sorted(glob.glob(patron))
            if not archivos:
                carpeta_mostrar = os.path.join(outdir_abs, "Impression_1", fp_carpeta)
                self.root.after(0, lambda: messagebox.showerror(
                    "Error de Anguli",
                    f"No se encontró ningún .jpg en:\n{carpeta_mostrar}"
                ))
                self.root.after(0, lambda: self._set_btns_anguli(True))
                return

            ruta_original = archivos[0]   # siempre 1.jpg (único archivo por carpeta)

            # Pasar por simular_captura
            ruta_simulada = os.path.join(outdir_abs, "huella_simulada.jpg")
            simular_captura_modulo.simular_captura_real(
                ruta_imagen=ruta_original,
                ruta_salida=ruta_simulada,
            )

            self.root.after(0, lambda: self._cargar_desde_ruta(ruta_simulada))

        except FileNotFoundError:
            self.root.after(0, lambda: messagebox.showerror(
                "Anguli no encontrado",
                f"No se encontró Anguli.exe en:\n{ANGULI_EXE}\n\n"
                "Verificá que la ruta ANGULI_EXE en el código sea correcta."
            ))
            self.root.after(0, lambda: self._set_btns_anguli(True))
        except subprocess.TimeoutExpired:
            self.root.after(0, lambda: messagebox.showerror(
                "Tiempo agotado", "Anguli tardó demasiado en responder (>60s)."
            ))
            self.root.after(0, lambda: self._set_btns_anguli(True))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror(
                "Error inesperado", f"Ocurrió un error al generar la huella:\n{e}"
            ))
            self.root.after(0, lambda: self._set_btns_anguli(True))

    def _cargar_desde_ruta(self, ruta: str):
        """Carga y procesa una huella a partir de una ruta ya conocida (sin diálogo)."""
        self._set_btns_anguli(True)
        self.ruta_huella = ruta
        nombre_archivo = os.path.basename(ruta)
        self.lbl_info_huella.config(text=f"Archivo: {nombre_archivo} | Procesando...")
        self.root.update_idletasks()

        try:
            res = identificar_huella_modulo.procesar_e_identificar(
                ruta, distancia_max=4.0, mostrar_grafico=False, simular=True
            )
            self.resultado_huella = res

            img_cv = cv2.imread(ruta)
            if img_cv is not None:
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_cv)
                img_pil.thumbnail((160, 160))
                self.tk_img_orig = ImageTk.PhotoImage(img_pil)
                self.lbl_canvas_orig.config(image=self.tk_img_orig, text="")

            if 'img_norm' in res:
                norm_pil = Image.fromarray(res['img_norm'])
                norm_pil.thumbnail((160, 160))
                self.tk_img_norm = ImageTk.PhotoImage(norm_pil)
                self.lbl_canvas_norm.config(image=self.tk_img_norm, text="")

            self.lbl_info_huella.config(
                text=f"Huella generada con Anguli: {nombre_archivo} lista."
            )
            self.lbl_res_mensaje.config(
                text="Huella cargada. Presione [ENTER] o 'Iniciar Grabación' para dar la orden.",
                fg="#edf2f4"
            )

        except Exception as e:
            messagebox.showerror("Error de Huella", f"No se pudo procesar la huella generada:\n{e}")
            self.lbl_info_huella.config(text="Error al procesar la huella generada.")

    def cargar_huella(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar imagen de huella dactilar",
            filetypes=[("Archivos de imagen", "*.png;*.jpg;*.jpeg;*.bmp;*.tif")]
        )
        if not ruta:
            return

        self.ruta_huella = ruta
        nombre_archivo = os.path.basename(ruta)
        self.lbl_info_huella.config(text=f"Archivo: {nombre_archivo} | Procesando...")
        self.root.update_idletasks()

        try:
            # Procesar la huella usando el módulo de identificación (sin mostrar popup)
            res = identificar_huella_modulo.procesar_e_identificar(ruta, distancia_max=4.0, mostrar_grafico=False,simular=True)
            self.resultado_huella = res

            # Mostrar huella original en thumbnail
            img_cv = cv2.imread(ruta)
            if img_cv is not None:
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_cv)
                img_pil.thumbnail((160, 160))
                self.tk_img_orig = ImageTk.PhotoImage(img_pil)
                self.lbl_canvas_orig.config(image=self.tk_img_orig, text="")

            # Mostrar huella normalizada en thumbnail
            if 'img_norm' in res:
                norm_pil = Image.fromarray(res['img_norm'])
                norm_pil.thumbnail((160, 160))
                self.tk_img_norm = ImageTk.PhotoImage(norm_pil)
                self.lbl_canvas_norm.config(image=self.tk_img_norm, text="")

            self.lbl_info_huella.config(
                text=f"Huella: {nombre_archivo} cargada y preprocesada con éxito."
            )
            self.lbl_res_mensaje.config(
                text="Huella cargada. Presione [ENTER] o 'Iniciar Grabación' para dar la orden.",
                fg="#edf2f4"
            )

        except Exception as e:
            messagebox.showerror("Error de Huella", f"No se pudo procesar la huella:\n{e}")
            self.lbl_info_huella.config(text="Error al procesar la huella.")

    def actualizar_estado_ui(self, texto):
        """Callback llamado desde CapturaAudio para actualizar la GUI en vivo."""
        def _update():
            self.lbl_estado_texto.config(text=texto)
            if "Prepárate" in texto:
                self.frame_estado.config(bg="#f4a261")
                self.lbl_estado_icon.config(bg="#f4a261", text="⏳")
                self.lbl_estado_texto.config(bg="#f4a261", fg="#1e1e24")
            elif "silencio" in texto.lower():
                self.frame_estado.config(bg="#457b9d")
                self.lbl_estado_icon.config(bg="#457b9d", text="🤫")
                self.lbl_estado_texto.config(bg="#457b9d", fg="white")
            elif "GRABANDO" in texto:
                self.frame_estado.config(bg="#e63946")
                self.lbl_estado_icon.config(bg="#e63946", text="🔴")
                self.lbl_estado_texto.config(bg="#e63946", fg="white")
            else:
                self.frame_estado.config(bg="#1e1e24")
                self.lbl_estado_icon.config(bg="#1e1e24", text="🎙️")
                self.lbl_estado_texto.config(bg="#1e1e24", fg="#edf2f4")
        self.root.after(0, _update)

    def iniciar_grabacion_hilo(self):
        if self.grabando_audio:
            return

        if not self.ruta_huella or not self.resultado_huella:
            messagebox.showwarning("Atención", "Primero debe cargar una huella dactilar.")
            return

        self.grabando_audio = True
        self.btn_grabar.config(state="disabled", bg="#6c757d", text="⏳ Grabando...")

        # Lanzar en segundo plano para no congelar la UI
        hilo = threading.Thread(target=self._hilo_proceso_audio, daemon=True)
        hilo.start()

    def _hilo_proceso_audio(self):
        try:
            vector_audio = CapturaAudio.capturar_comando_voz(
                duracion_total=3,
                duracion_ruido=1,
                countdown=1,
                grabar=True,
                carpeta_grabar="Audios/EnVivoApp",
                ruidoPostrior=True,
                modo_debug=True,
                callback_estado=self.actualizar_estado_ui
            )

            # Clasificar el vector de audio
            if vector_audio is None:
                self.root.after(0, lambda: self.finalizar_proceso(None, None, False, "No se detectó audio claro."))
                return

            clase, distancias, es_valido = clasificar_vector(vector_audio)
            self.root.after(0, lambda: self.finalizar_proceso(clase, distancias, es_valido))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error de Audio", f"Ocurrió un error al capturar voz:\n{e}"))
            self.root.after(0, self._restaurar_boton)

    def finalizar_proceso(self, comando, distancias, es_valido, error_msg=None):
        self._restaurar_boton()
        self.actualizar_estado_ui("Micrófono en espera...")

        if distancias:
            self.lbl_dist_rec.config(text=f"Reconocer   : {distancias.get('reconocer', 0):.2f}")
            self.lbl_dist_ide.config(text=f"Identificar : {distancias.get('identificar', 0):.2f}")
            self.lbl_dist_sal.config(text=f"Salir       : {distancias.get('salir', 0):.2f}")

        if error_msg:
            self.lbl_res_mensaje.config(text=f"⚠️ {error_msg}", fg="#f4a261")
            return

        if not es_valido:
            self.lbl_res_mensaje.config(
                text="⚠️ COMANDO NO RECONOCIDO (Hable más claro o repita 'Reconocer', 'Identificar' o 'Salir')",
                fg="#f4a261"
            )
            return

        # Procesar según el comando detectado
        if comando == "reconocer":
            veredicto = self.resultado_huella['veredicto']
            dist = self.resultado_huella['distancia_min']
            if veredicto:
                self.lbl_res_mensaje.config(
                    text=f"✅ ACCESO AUTORIZADO (Distancia huella: {dist:.2f} <= 4.0)",
                    fg="#2a9d8f"
                )
            else:
                self.lbl_res_mensaje.config(
                    text=f"❌ ACCESO DENEGADO / NO AUTORIZADO (Distancia huella: {dist:.2f} > 4.0)",
                    fg="#e63946"
                )

        elif comando == "identificar":
            veredicto = self.resultado_huella['veredicto']
            identidad = self.resultado_huella['clase_identificada']
            dist = self.resultado_huella['distancia_min']
            if veredicto and identidad:
                self.lbl_res_mensaje.config(
                    text=f"👤 IDENTIDAD CONFIRMADA: {identidad.upper()} (Distancia: {dist:.2f})",
                    fg="#2a9d8f"
                )
            else:
                c_cercano = self.resultado_huella['cluster_cercano']
                self.lbl_res_mensaje.config(
                    text=f"❓ IDENTIDAD DESCONOCIDA (Huella no registrada, clúster más próximo: {c_cercano})",
                    fg="#e63946"
                )

        elif comando == "salir":
            respuesta = messagebox.askyesno("Confirmar Salida", "¿Está seguro que desea salir del sistema?")
            if respuesta:
                self.root.destroy()
            else:
                self.lbl_res_mensaje.config(
                    text="Operación cancelada. El sistema sigue en ejecución.",
                    fg="#edf2f4"
                )

    def _restaurar_boton(self):
        self.grabando_audio = False
        self.btn_grabar.config(state="normal", bg="#e63946", text="🎙️ Iniciar Grabación (o presione ENTER)")


def main():
    root = tk.Tk()
    app = SistemaBiometricoApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

