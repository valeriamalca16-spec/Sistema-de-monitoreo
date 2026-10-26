import sys
import cv2
import csv
import os
from datetime import datetime

# IMPORTANTE: ultralytics debe importarse ANTES de PyQt5 en Windows
from ultralytics import YOLO

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QHBoxLayout, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, Qt

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    metricas_signal = pyqtSignal(int, int, int) 

    def __init__(self):
        super().__init__()
        self.registro_acumulado = {}

    def run(self):
        modelo = YOLO('yolov8n-pose.pt')
        cap = cv2.VideoCapture(0)
        

        UMBRAL_FRAMES = 25 

        while True:
            exito, frame = cap.read()
            if not exito:
                continue

            resultados = modelo.track(frame, persist=True, conf=0.7, device='cpu', tracker="bytetrack.yaml")
            
            aforo_total = 0
            manos_levantadas_actual = 0
            alumnos_distraidos_actual = 0 

            for resultado in resultados:
                frame_anotado = resultado.plot()
                aforo_total = len(resultado.boxes)
                
                if resultado.keypoints is not None and len(resultado.keypoints.xy) > 0 and resultado.boxes.id is not None:
                    keypoints_personas = resultado.keypoints.xy
                    ids = resultado.boxes.id.int().cpu().tolist()
                    
                    for i, kp in enumerate(keypoints_personas):
                        if i < len(ids):
                            id_persona = ids[i]
                            
                            # Nuevo diccionario con contadores de fotogramas
                            if id_persona not in self.registro_acumulado:
                                self.registro_acumulado[id_persona] = {
                                    "participaciones": 0,
                                    "distracciones": 0,
                                    "frames_consecutivos_mano": 0,
                                    "frames_consecutivos_distraido": 0,
                                    "mano_contada": False,
                                    "distraccion_contada": False
                                }

                            if len(kp) >= 11:
                                nariz_y = float(kp[0][1]) 
                                hombro_izq_y = float(kp[5][1])
                                hombro_der_y = float(kp[6][1])
                                muneca_izq_y = float(kp[9][1])
                                muneca_der_y = float(kp[10][1])
                                
                                mano_izq_arriba = (muneca_izq_y > 0 and hombro_izq_y > 0) and (muneca_izq_y < hombro_izq_y)
                                mano_der_arriba = (muneca_der_y > 0 and hombro_der_y > 0) and (muneca_der_y < hombro_der_y)
                                mano_arriba = mano_izq_arriba or mano_der_arriba
                                
                                distraido = False
                                if hombro_izq_y > 0 and hombro_der_y > 0 and nariz_y > 0:
                                    hombros_y_promedio = (hombro_izq_y + hombro_der_y) / 2
                                    distancia_cabeza = hombros_y_promedio - nariz_y
                                    if distancia_cabeza < 30: 
                                        distraido = True
                                        
                                if mano_arriba: manos_levantadas_actual += 1
                                if distraido: alumnos_distraidos_actual += 1

                                if mano_arriba:
                                    self.registro_acumulado[id_persona]["frames_consecutivos_mano"] += 1
                                    if self.registro_acumulado[id_persona]["frames_consecutivos_mano"] >= UMBRAL_FRAMES and not self.registro_acumulado[id_persona]["mano_contada"]:
                                        self.registro_acumulado[id_persona]["participaciones"] += 1
                                        self.registro_acumulado[id_persona]["mano_contada"] = True
                                else:
                                    self.registro_acumulado[id_persona]["frames_consecutivos_mano"] = 0
                                    self.registro_acumulado[id_persona]["mano_contada"] = False
                                    
                                if distraido:
                                    self.registro_acumulado[id_persona]["frames_consecutivos_distraido"] += 1
                                    if self.registro_acumulado[id_persona]["frames_consecutivos_distraido"] >= UMBRAL_FRAMES and not self.registro_acumulado[id_persona]["distraccion_contada"]:
                                        self.registro_acumulado[id_persona]["distracciones"] += 1
                                        self.registro_acumulado[id_persona]["distraccion_contada"] = True
                                else:
                                    self.registro_acumulado[id_persona]["frames_consecutivos_distraido"] = 0
                                    self.registro_acumulado[id_persona]["distraccion_contada"] = False

            rgb_image = cv2.cvtColor(frame_anotado, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
            
            self.change_pixmap_signal.emit(p)
            self.metricas_signal.emit(aforo_total, manos_levantadas_actual, alumnos_distraidos_actual)

class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema Inteligente de Aforo y Monitoreo Pedagógico")
        self.setGeometry(100, 100, 900, 500)

        self.current_aforo = 0
        self.current_manos = 0
        self.current_distraidos = 0

        widget_central = QWidget()
        layout_principal = QHBoxLayout()

        self.label_video = QLabel(self)
        self.label_video.setFixedSize(640, 480)
        self.label_video.setStyleSheet("background-color: black; border: 2px solid #bdc3c7;")
        layout_principal.addWidget(self.label_video)

        layout_metricas = QVBoxLayout()
        self.label_titulo = QLabel("Estado del Aula", self)
        self.label_aforo = QLabel("Aforo Total: 0", self)
        self.label_manos = QLabel("Participando: 0", self)
        self.label_distraidos = QLabel("Distraídos: 0", self) 
        
        self.label_titulo.setStyleSheet("font-size: 26px; font-weight: bold; margin-bottom: 20px; color: #2c3e50;")
        self.label_aforo.setStyleSheet("font-size: 20px; color: #34495e;")
        self.label_manos.setStyleSheet("font-size: 20px; color: #27ae60; font-weight: bold;") 
        self.label_distraidos.setStyleSheet("font-size: 20px; color: #c0392b; font-weight: bold; margin-bottom: 30px;") 
        
        self.btn_exportar = QPushButton("Registrar Muestra (CSV)", self)
        self.btn_exportar.setStyleSheet("""
            QPushButton {
                background-color: #2980b9; 
                color: white; 
                font-size: 16px; 
                font-weight: bold; 
                padding: 10px; 
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #3498db; }
            QPushButton:pressed { background-color: #1c5981; }
        """)
        self.btn_exportar.clicked.connect(self.guardar_datos_csv)

        layout_metricas.addWidget(self.label_titulo)
        layout_metricas.addWidget(self.label_aforo)
        layout_metricas.addWidget(self.label_manos)
        layout_metricas.addWidget(self.label_distraidos)
        layout_metricas.addWidget(self.btn_exportar) # Añadimos el botón al panel
        layout_metricas.addStretch() 

        layout_principal.addLayout(layout_metricas)
        widget_central.setLayout(layout_principal)
        self.setCentralWidget(widget_central)

        self.hilo_video = VideoThread()
        self.hilo_video.change_pixmap_signal.connect(self.actualizar_imagen)
        self.hilo_video.metricas_signal.connect(self.actualizar_metricas)
        self.hilo_video.start()

    def actualizar_imagen(self, cv_img):
        self.label_video.setPixmap(QPixmap.fromImage(cv_img))

    def actualizar_metricas(self, aforo, manos, distraidos):
        self.label_aforo.setText(f"Aforo Total: {aforo}")
        self.label_manos.setText(f"Participando: {manos}")
        self.label_distraidos.setText(f"Distraídos: {distraidos}")
        
        self.current_aforo = aforo
        self.current_manos = manos
        self.current_distraidos = distraidos

    def guardar_datos_csv(self):
        nombre_archivo = "reporte_acumulado_clase.csv"
        archivo_existe = os.path.isfile(nombre_archivo)
        
        with open(nombre_archivo, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            
            if not archivo_existe:
                writer.writerow(["Fecha y Hora", "ID Estudiante", "Total Participaciones", "Total Distracciones"])
            
            ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            historial = self.hilo_video.registro_acumulado
            
            for id_estudiante, datos in historial.items():
                writer.writerow([ahora, f"Alumno {id_estudiante}", datos["participaciones"], datos["distracciones"]])
            
            writer.writerow([])
            
        self.btn_exportar.setText("¡Reporte Guardado!")
        self.btn_exportar.setStyleSheet("background-color: #27ae60; color: white; font-size: 16px; font-weight: bold; padding: 10px; border-radius: 5px;")
        
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(1000, self.restaurar_boton)

    def restaurar_boton(self):
        self.btn_exportar.setText("Registrar Muestra (CSV)")
        self.btn_exportar.setStyleSheet("""
            QPushButton { background-color: #2980b9; color: white; font-size: 16px; font-weight: bold; padding: 10px; border-radius: 5px; }
            QPushButton:hover { background-color: #3498db; }
        """)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ventana = Dashboard()
    ventana.show()
    sys.exit(app.exec_())