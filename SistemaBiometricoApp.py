import os
import sys
import threading
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
identificar_huella_modulo = importlib.import_module("NormalizadorHuella.6-identificarHuella")

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
        btn_cargar.pack(fill="x", padx=15, pady=8)

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
            res = identificar_huella_modulo.procesar_e_identificar(ruta, distancia_max=4.0, mostrar_grafico=False)
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

