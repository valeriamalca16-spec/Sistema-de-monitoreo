import cv2
from ultralytics import YOLO

print("Cargando modelo...")
modelo = YOLO('yolov8n-pose.pt')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo acceder a la cámara web.")
    exit()

print("Cámara iniciada. Presiona la tecla 'q' en la ventana de video para salir.")

while True:
    exito, frame = cap.read()
    
    if not exito:
        print("Error al leer el frame de la cámara.")
        break

    resultados = modelo(frame, stream=True)
    
    for resultado in resultados:
        frame_anotado = resultado.plot()

    cv2.imshow("Monitoreo YOLOv8 Pose - Prueba", frame_anotado)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Proceso finalizado.")