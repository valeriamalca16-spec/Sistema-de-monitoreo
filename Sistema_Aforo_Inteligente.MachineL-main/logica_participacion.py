import cv2
from ultralytics import YOLO

print("Cargando modelo...")
modelo = YOLO('yolov8n-pose.pt')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo acceder a la cámara web.")
    exit()

while True:
    exito, frame = cap.read()
    if not exito:
        break

    resultados = modelo(frame, stream=True, conf=0.7)
    
    aforo_total = 0
    manos_levantadas = 0

    for resultado in resultados:
        frame_anotado = resultado.plot()
        
        aforo_total = len(resultado.boxes)
        
        if resultado.keypoints is not None and len(resultado.keypoints.xy) > 0:
            keypoints_personas = resultado.keypoints.xy
            
            for kp in keypoints_personas:
                if len(kp) >= 11: 
                    hombro_izq_y = float(kp[5][1])
                    hombro_der_y = float(kp[6][1])
                    muneca_izq_y = float(kp[9][1])
                    muneca_der_y = float(kp[10][1])

                    mano_izq_arriba = (muneca_izq_y > 0 and hombro_izq_y > 0) and (muneca_izq_y < hombro_izq_y)
                    mano_der_arriba = (muneca_der_y > 0 and hombro_der_y > 0) and (muneca_der_y < hombro_der_y)
                    
                    if mano_izq_arriba or mano_der_arriba:
                        manos_levantadas += 1

    cv2.putText(frame_anotado, f"Aforo: {aforo_total}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame_anotado, f"Manos Levantadas: {manos_levantadas}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

    cv2.imshow("Monitoreo Inteligente - Aforo y Participacion", frame_anotado)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Proceso finalizado.")