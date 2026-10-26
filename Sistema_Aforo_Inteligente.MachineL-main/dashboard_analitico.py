import sys
import cv2
import csv
import os
from datetime import datetime

# IMPORTANTE: ultralytics debe importarse ANTES de PyQt5 en Windows
# para evitar el conflicto DLL "WinError 1114" con PyTorch.
from ultralytics import YOLO

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel,
    QVBoxLayout, QWidget, QHBoxLayout, QPushButton, QFrame
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer

# Librerías para integrar gráficos de Matplotlib en PyQt5
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


# =========================================================
# HOJA DE ESTILOS GLOBAL (MODO OSCURO - DARK MODE)
# =========================================================
DARK_STYLESHEET = """
    QMainWindow {
        background-color: #0d1117;
    }
    QWidget {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: 'Segoe UI', Arial, sans-serif;
    }
    QLabel {
        background-color: transparent;
        color: #e6edf3;
    }
    QPushButton {
        background-color: #1f6feb;
        color: #ffffff;
        font-size: 14px;
        font-weight: bold;
        padding: 12px 18px;
        border-radius: 8px;
        border: 1px solid #388bfd;
    }
    QPushButton:hover {
        background-color: #388bfd;
        border: 1px solid #58a6ff;
    }
    QPushButton:pressed {
        background-color: #1158c7;
    }
    QFrame {
        background-color: #161b22;
        border-radius: 12px;
        border: 1px solid #30363d;
    }
"""


# =========================================================
# 1. HILO DE VIDEO
# =========================================================
# Esta clase se ejecuta en segundo plano para evitar que
# la interfaz gráfica se congele mientras se procesa el video.
#
# Funciones principales:
# - Capturar video desde la cámara
# - Ejecutar YOLOv8 Pose (personas y posturas)
# - Ejecutar YOLOv8 Nano (detección de celulares)
# - Identificar manos levantadas
# - Detectar posibles distracciones por cabeza baja
# - Detectar uso de celular
# - Enviar métricas a la interfaz
# =========================================================
class VideoThread(QThread):

    # Señal para enviar imágenes procesadas a la interfaz
    change_pixmap_signal = pyqtSignal(QImage)

    # Señal para enviar métricas:
    # (aforo_total, manos_levantadas, distraidos, celulares)
    metricas_signal = pyqtSignal(int, int, int, int)

    def __init__(self):
        super().__init__()
        # Se inicializa aquí para evitar que el gráfico falle
        # si lo consulta antes de que YOLO termine de cargar
        self.registro_acumulado = {}

    def run(self):

        # =====================================================
        # CARGA DE MODELOS
        # =====================================================
        # Modelo 1: YOLOv8 Pose -> detecta personas y esqueleto
        modelo_pose = YOLO('yolov8n-pose.pt')

        # Modelo 2: YOLOv8 Nano -> detecta objetos generales
        # (incluyendo 'cell phone', clase ID = 67 en COCO)
        modelo_objetos = YOLO('yolov8n.pt')

        # Captura de video desde la webcam
        cap = cv2.VideoCapture(0)

        # Cantidad mínima de frames consecutivos necesarios
        # para considerar válida una acción.
        # Esto evita contar falsos positivos o movimientos rápidos.
        UMBRAL_FRAMES = 25

        # Bucle principal del procesamiento de video
        while True:

            exito, frame = cap.read()

            # Si no se pudo leer el frame, continuar
            if not exito:
                continue

            # =====================================================
            # DETECCIÓN DE CELULARES (Modelo 2)
            # =====================================================
            # Ejecutamos el modelo de objetos sobre el mismo frame.
            # Filtramos solo la clase 67 (cell phone).
            # Guardamos los recuadros (bounding boxes) de cada
            # celular detectado para compararlos luego con las
            # personas.
            # =====================================================
            resultados_objetos = modelo_objetos(
                frame,
                conf=0.5,
                device='cpu',
                classes=[67],   # Solo detecta teléfonos celulares
                verbose=False
            )

            cajas_celulares = []
            for r in resultados_objetos:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cajas_celulares.append((x1, y1, x2, y2))

                    # Dibujamos el recuadro del celular en rojo brillante
                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 0, 255), 3
                    )
                    cv2.putText(
                        frame,
                        "CELULAR DETECTADO",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2
                    )

            # =====================================================
            # TRACKING DE PERSONAS (Modelo 1)
            # =====================================================
            # persist=True permite mantener IDs estables
            # para cada persona entre frames.
            # ByteTrack se usa para el seguimiento.
            # =====================================================
            resultados = modelo_pose.track(
                frame,
                persist=True,
                conf=0.7,
                device='cpu',
                tracker="bytetrack.yaml"
            )

            # Variables temporales para métricas actuales
            aforo_total = 0
            manos_levantadas_actual = 0
            alumnos_distraidos_actual = 0
            alumnos_celular_actual = 0

            # =====================================================
            # RECORRIDO DE RESULTADOS DETECTADOS
            # =====================================================
            for resultado in resultados:

                # Genera un frame anotado con las detecciones
                frame_anotado = resultado.plot()

                # Cantidad total de personas detectadas
                aforo_total = len(resultado.boxes)

                # Validamos que existan puntos clave e IDs
                if (
                    resultado.keypoints is not None and
                    len(resultado.keypoints.xy) > 0 and
                    resultado.boxes.id is not None
                ):

                    # Puntos clave del cuerpo
                    keypoints_personas = resultado.keypoints.xy

                    # IDs únicos asignados por el tracker
                    ids = resultado.boxes.id.int().cpu().tolist()

                    # Recorremos cada persona detectada
                    for i, kp in enumerate(keypoints_personas):

                        # Validación de índice
                        if i < len(ids):

                            id_persona = ids[i]

                            # =================================================
                            # INICIALIZACIÓN DEL REGISTRO DE CADA PERSONA
                            # =================================================
                            # Se crea una estructura para guardar:
                            # - participaciones
                            # - distracciones
                            # - uso de celular
                            # - control antirrebote para cada métrica
                            # =================================================
                            if id_persona not in self.registro_acumulado:

                                self.registro_acumulado[id_persona] = {

                                    "participaciones": 0,
                                    "distracciones": 0,
                                    "uso_celular": 0,

                                    "frames_consecutivos_mano": 0,
                                    "frames_consecutivos_distraido": 0,
                                    "frames_consecutivos_celular": 0,

                                    "mano_contada": False,
                                    "distraccion_contada": False,
                                    "celular_contado": False
                                }

                            # =================================================
                            # VALIDACIÓN DE PUNTOS CLAVE
                            # =================================================
                            # YOLO Pose entrega varios puntos:
                            #
                            # 0  -> nariz
                            # 5  -> hombro izquierdo
                            # 6  -> hombro derecho
                            # 9  -> muñeca izquierda
                            # 10 -> muñeca derecha
                            # =================================================
                            if len(kp) >= 11:

                                # Coordenada Y de la nariz
                                nariz_y = float(kp[0][1])

                                # Promedio vertical de ambos hombros
                                h_promedio_y = (
                                    float(kp[5][1]) +
                                    float(kp[6][1])
                                ) / 2

                                # =================================================
                                # DETECCIÓN DE MANO LEVANTADA
                                # =================================================
                                # En coordenadas de imagen:
                                # un valor Y menor significa "más arriba"
                                #
                                # Si alguna muñeca está más arriba
                                # que su respectivo hombro -> mano levantada
                                # =================================================
                                mano_arriba = (
                                    float(kp[9][1]) < float(kp[5][1]) or
                                    float(kp[10][1]) < float(kp[6][1])
                                )

                                # =================================================
                                # DETECCIÓN DE POSIBLE DISTRACCIÓN (cabeza baja)
                                # =================================================
                                # Se compara la distancia vertical entre
                                # la nariz y los hombros.
                                # Si la cabeza está muy inclinada -> distracción.
                                # =================================================
                                distraido = (
                                    (h_promedio_y - nariz_y) < 30
                                )

                                # =================================================
                                # DETECCIÓN DE USO DE CELULAR
                                # =================================================
                                # Obtenemos el recuadro de la persona actual
                                # y verificamos si algún celular detectado
                                # está dentro o cerca de ese recuadro.
                                # =================================================
                                usando_celular = False
                                if len(cajas_celulares) > 0:
                                    px1, py1, px2, py2 = resultado.boxes.xyxy[i].tolist()
                                    for (cx1, cy1, cx2, cy2) in cajas_celulares:
                                        # Verificamos solapamiento entre recuadros
                                        solapamiento = not (
                                            cx2 < px1 or cx1 > px2 or
                                            cy2 < py1 or cy1 > py2
                                        )
                                        if solapamiento:
                                            usando_celular = True
                                            break

                                # Conteo visual instantáneo
                                if mano_arriba:
                                    manos_levantadas_actual += 1

                                if distraido:
                                    alumnos_distraidos_actual += 1

                                if usando_celular:
                                    alumnos_celular_actual += 1

                                # =================================================
                                # LÓGICA ANTIRREBOTE - PARTICIPACIÓN
                                # =================================================
                                if mano_arriba:
                                    self.registro_acumulado[id_persona]["frames_consecutivos_mano"] += 1
                                    if (
                                        self.registro_acumulado[id_persona]["frames_consecutivos_mano"] >= UMBRAL_FRAMES
                                        and not self.registro_acumulado[id_persona]["mano_contada"]
                                    ):
                                        self.registro_acumulado[id_persona]["participaciones"] += 1
                                        self.registro_acumulado[id_persona]["mano_contada"] = True
                                else:
                                    self.registro_acumulado[id_persona]["frames_consecutivos_mano"] = 0
                                    self.registro_acumulado[id_persona]["mano_contada"] = False

                                # =================================================
                                # LÓGICA ANTIRREBOTE - DISTRACCIÓN
                                # =================================================
                                if distraido:
                                    self.registro_acumulado[id_persona]["frames_consecutivos_distraido"] += 1
                                    if (
                                        self.registro_acumulado[id_persona]["frames_consecutivos_distraido"] >= UMBRAL_FRAMES
                                        and not self.registro_acumulado[id_persona]["distraccion_contada"]
                                    ):
                                        self.registro_acumulado[id_persona]["distracciones"] += 1
                                        self.registro_acumulado[id_persona]["distraccion_contada"] = True
                                else:
                                    self.registro_acumulado[id_persona]["frames_consecutivos_distraido"] = 0
                                    self.registro_acumulado[id_persona]["distraccion_contada"] = False

                                # =================================================
                                # LÓGICA ANTIRREBOTE - USO DE CELULAR
                                # =================================================
                                if usando_celular:
                                    self.registro_acumulado[id_persona]["frames_consecutivos_celular"] += 1
                                    if (
                                        self.registro_acumulado[id_persona]["frames_consecutivos_celular"] >= UMBRAL_FRAMES
                                        and not self.registro_acumulado[id_persona]["celular_contado"]
                                    ):
                                        self.registro_acumulado[id_persona]["uso_celular"] += 1
                                        self.registro_acumulado[id_persona]["celular_contado"] = True
                                else:
                                    self.registro_acumulado[id_persona]["frames_consecutivos_celular"] = 0
                                    self.registro_acumulado[id_persona]["celular_contado"] = False

            # =====================================================
            # CONVERSIÓN DE FRAME A FORMATO QT
            # =====================================================
            rgb_image = cv2.cvtColor(frame_anotado, cv2.COLOR_BGR2RGB)

            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w

            convert_to_Qt_format = QImage(
                rgb_image.data,
                w,
                h,
                bytes_per_line,
                QImage.Format_RGB888
            )

            # Escalado de imagen para mostrar en la interfaz
            p = convert_to_Qt_format.scaled(
                640,
                480,
                Qt.KeepAspectRatio
            )

            # Envío de imagen y métricas a la interfaz
            self.change_pixmap_signal.emit(p)

            self.metricas_signal.emit(
                aforo_total,
                manos_levantadas_actual,
                alumnos_distraidos_actual,
                alumnos_celular_actual
            )


# =========================================================
# 2. CLASE DEL GRÁFICO (MODO OSCURO)
# =========================================================
# Esta clase crea un gráfico dinámico usando Matplotlib.
#
# El gráfico muestra (barras con tema oscuro):
# - Participaciones acumuladas
# - Distracciones acumuladas
# - Usos de celular acumulados
# =========================================================
class CanvasGrafico(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):

        # Fondo del gráfico en modo oscuro
        fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#161b22')

        self.axes = fig.add_subplot(111)
        self.axes.set_facecolor('#0d1117')

        super().__init__(fig)
        self.setStyleSheet("background-color: #161b22; border-radius: 8px;")

        # Dibujo inicial vacío
        self._dibujar(0, 0, 0)

    def _dibujar(self, total_p, total_d, total_c):
        """Método interno que renderiza el gráfico con estilo oscuro."""
        self.axes.clear()
        self.axes.set_facecolor('#0d1117')

        categorias = ['Participaciones', 'Distracciones', 'Celular']
        valores    = [total_p, total_d, total_c]
        colores    = ['#3fb950', '#f85149', '#e3b341']

        bars = self.axes.bar(categorias, valores, color=colores, width=0.5, zorder=3)

        # Etiquetas encima de cada barra
        for bar, val in zip(bars, valores):
            self.axes.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                str(val),
                ha='center', va='bottom',
                color='#e6edf3', fontsize=11, fontweight='bold'
            )

        # Estilo del título y ejes
        self.axes.set_title(
            'Historial Acumulado de la Clase',
            color='#e6edf3', fontsize=12, fontweight='bold', pad=10
        )
        self.axes.tick_params(colors='#8b949e')
        for spine in self.axes.spines.values():
            spine.set_edgecolor('#30363d')

        # Ajuste automático del límite vertical
        max_val = max(valores) if max(valores) > 0 else 5
        self.axes.set_ylim(0, max_val + max(2, int(max_val * 0.2)))

        # Líneas de cuadrícula sutiles
        self.axes.yaxis.grid(True, color='#21262d', linewidth=0.8, zorder=0)
        self.axes.set_axisbelow(True)

        self.figure.tight_layout()
        self.draw()

    def actualizar_grafico(self, total_participaciones, total_distracciones, total_celulares):
        self._dibujar(total_participaciones, total_distracciones, total_celulares)


# =========================================================
# 3. TARJETA DE MÉTRICA (Componente reutilizable)
# =========================================================
def crear_tarjeta_metrica(parent, titulo, valor_inicial, color_valor):
    """Crea un contenedor visual con título y valor para cada métrica."""
    frame = QFrame(parent)
    frame.setFixedHeight(80)

    layout = QVBoxLayout(frame)
    layout.setContentsMargins(14, 8, 14, 8)
    layout.setSpacing(2)

    lbl_titulo = QLabel(titulo, frame)
    lbl_titulo.setStyleSheet(
        "font-size: 11px; color: #8b949e; font-weight: bold; "
        "text-transform: uppercase; letter-spacing: 1px; background: transparent; border: none;"
    )

    lbl_valor = QLabel(valor_inicial, frame)
    lbl_valor.setStyleSheet(
        f"font-size: 26px; font-weight: bold; color: {color_valor}; "
        "background: transparent; border: none;"
    )

    layout.addWidget(lbl_titulo)
    layout.addWidget(lbl_valor)

    return frame, lbl_valor


# =========================================================
# 4. INTERFAZ PRINCIPAL
# =========================================================
class DashboardAnalitico(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("Sistema Inteligente UPN - Dashboard Analítico")
        self.setGeometry(80, 60, 1180, 680)
        self.setStyleSheet(DARK_STYLESHEET)

        # =====================================================
        # CONTENEDOR PRINCIPAL
        # =====================================================
        widget_central = QWidget()
        layout_principal = QHBoxLayout()
        layout_principal.setContentsMargins(16, 16, 16, 16)
        layout_principal.setSpacing(16)

        # =====================================================
        # PANEL IZQUIERDO -> VIDEO
        # =====================================================
        frame_video = QFrame()
        frame_video.setFixedSize(648, 488)
        layout_video = QVBoxLayout(frame_video)
        layout_video.setContentsMargins(4, 4, 4, 4)

        self.label_video = QLabel()
        self.label_video.setFixedSize(640, 480)
        self.label_video.setAlignment(Qt.AlignCenter)
        self.label_video.setStyleSheet(
            "background-color: #010409;"
            "border-radius: 8px;"
        )
        layout_video.addWidget(self.label_video)

        layout_principal.addWidget(frame_video)

        # =====================================================
        # PANEL DERECHO -> MÉTRICAS Y GRÁFICO
        # =====================================================
        layout_derecho = QVBoxLayout()
        layout_derecho.setSpacing(12)

        # Título del panel
        label_titulo = QLabel("⬤  Análisis Pedagógico en Tiempo Real")
        label_titulo.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #58a6ff; "
            "padding: 8px 0px; border: none; background: transparent;"
        )
        layout_derecho.addWidget(label_titulo)

        # =====================================================
        # TARJETAS DE MÉTRICAS
        # =====================================================
        tarjeta_aforo, self.label_aforo = crear_tarjeta_metrica(
            widget_central, "Aforo Actual", "0 personas", "#e6edf3"
        )
        tarjeta_manos, self.label_manos = crear_tarjeta_metrica(
            widget_central, "Manos Levantadas", "0", "#3fb950"
        )
        tarjeta_distraidos, self.label_distraidos = crear_tarjeta_metrica(
            widget_central, "Posibles Distraídos", "0", "#f85149"
        )
        tarjeta_celular, self.label_celular = crear_tarjeta_metrica(
            widget_central, "Usando Celular 📱", "0", "#e3b341"
        )

        layout_derecho.addWidget(tarjeta_aforo)
        layout_derecho.addWidget(tarjeta_manos)
        layout_derecho.addWidget(tarjeta_distraidos)
        layout_derecho.addWidget(tarjeta_celular)

        # =====================================================
        # GRÁFICO DINÁMICO (MODO OSCURO)
        # =====================================================
        self.canvas = CanvasGrafico(widget_central, width=4, height=2.8)
        layout_derecho.addWidget(self.canvas)

        # =====================================================
        # BOTÓN DE EXPORTAR
        # =====================================================
        self.btn_exportar = QPushButton("⬇  Exportar Reporte Final (CSV)")
        self.btn_exportar.setFixedHeight(44)
        self.btn_exportar.clicked.connect(self.guardar_datos_csv)

        layout_derecho.addWidget(self.btn_exportar)

        # Ensamblamos el layout
        layout_principal.addLayout(layout_derecho)

        widget_central.setLayout(layout_principal)
        self.setCentralWidget(widget_central)

        # =====================================================
        # CONFIGURACIÓN DEL HILO DE VIDEO
        # =====================================================
        self.hilo_video = VideoThread()

        self.hilo_video.change_pixmap_signal.connect(self.actualizar_imagen)
        self.hilo_video.metricas_signal.connect(self.actualizar_metricas)

        # Inicia el procesamiento en segundo plano
        self.hilo_video.start()

        # =====================================================
        # TEMPORIZADOR DEL GRÁFICO
        # =====================================================
        # Actualiza el gráfico de barras cada 2 segundos
        self.timer_grafico = QTimer()
        self.timer_grafico.timeout.connect(self.refrescar_estadisticas)
        self.timer_grafico.start(2000)

    # =========================================================
    # ACTUALIZAR VIDEO EN LA INTERFAZ
    # =========================================================
    def actualizar_imagen(self, cv_img):
        self.label_video.setPixmap(QPixmap.fromImage(cv_img))

    # =========================================================
    # ACTUALIZAR MÉTRICAS EN TIEMPO REAL
    # =========================================================
    def actualizar_metricas(self, aforo, manos, distraidos, celulares):

        self.label_aforo.setText(f"{aforo} personas")
        self.label_manos.setText(str(manos))
        self.label_distraidos.setText(str(distraidos))
        self.label_celular.setText(str(celulares))

    # =========================================================
    # REFRESCAR DATOS DEL GRÁFICO
    # =========================================================
    def refrescar_estadisticas(self):

        registros = self.hilo_video.registro_acumulado.values()

        total_p = sum(d["participaciones"] for d in registros)
        total_d = sum(d["distracciones"]   for d in self.hilo_video.registro_acumulado.values())
        total_c = sum(d["uso_celular"]     for d in self.hilo_video.registro_acumulado.values())

        self.canvas.actualizar_grafico(total_p, total_d, total_c)

    # =========================================================
    # EXPORTAR REPORTE CSV (con columna de uso de celular)
    # =========================================================
    def guardar_datos_csv(self):

        nombre = "reporte_analitico.csv"
        archivo_existe = os.path.isfile(nombre) and os.stat(nombre).st_size > 0

        with open(nombre, mode='a', newline='', encoding='utf-8') as f:

            writer = csv.writer(f)

            if not archivo_existe:
                writer.writerow([
                    "Fecha",
                    "ID Estudiante",
                    "Participaciones",
                    "Distracciones",
                    "Uso de Celular"
                ])

            ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            for id_est, datos in self.hilo_video.registro_acumulado.items():
                writer.writerow([
                    ahora,
                    f"Estudiante {id_est}",
                    datos["participaciones"],
                    datos["distracciones"],
                    datos["uso_celular"]
                ])

        # Retroalimentación visual: el botón cambia de color brevemente
        self.btn_exportar.setText("✔  ¡Reporte Guardado!")
        self.btn_exportar.setStyleSheet(
            "background-color: #238636; color: white; font-size: 14px; "
            "font-weight: bold; padding: 12px; border-radius: 8px; border: 1px solid #2ea043;"
        )

        # Restaurar botón a su estado original luego de 1.5 segundos
        QTimer.singleShot(
            1500,
            lambda: (
                self.btn_exportar.setText("⬇  Exportar Reporte Final (CSV)"),
                self.btn_exportar.setStyleSheet(
                    "background-color: #1f6feb; color: white; font-size: 14px; "
                    "font-weight: bold; padding: 12px; border-radius: 8px; border: 1px solid #388bfd;"
                )
            )
        )


# =========================================================
# PUNTO DE INICIO DEL PROGRAMA
# =========================================================
if __name__ == '__main__':

    app = QApplication(sys.argv)

    ventana = DashboardAnalitico()

    ventana.show()

    sys.exit(app.exec_())