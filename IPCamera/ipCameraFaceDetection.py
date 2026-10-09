import cv2
import mediapipe as mp

def deteccion_facial_mediapipe():
    # Inicializar MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    # Configurar la detección facial
    # min_detection_confidence: umbral de confianza para que una detección se considere válida
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    url_rtsp = 'rtsp://admin:321_monzad@192.168.1.110/Streaming/channels/1001' # Reemplaza con tu URL RTSP  rtsp://admin:Admin8350@192.168.31.168:65003/8000

    # Inicializar la captura de video desde la cámara web
    # El 0 indica la cámara por defecto del sistema
    cap = cv2.VideoCapture(url_rtsp)

    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    print("Presiona 'q' para salir.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame de la cámara.")
            break

        # Voltear la imagen horizontalmente para una vista tipo espejo (opcional, pero común)
        #frame = cv2.flip(frame, 1)

        # Convertir la imagen de BGR a RGB, ya que MediaPipe espera imágenes RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar la imagen con MediaPipe Face Detection
        results = face_detection.process(image_rgb)

        # Dibujar las detecciones faciales en la imagen
        if results.detections:
            for detection in results.detections:
                # Dibujar el recuadro delimitador y los 6 puntos clave del rostro
                mp_drawing.draw_detection(frame, detection)

        # Mostrar el frame procesado
        cv2.imshow('Deteccion Facial con MediaPipe', frame)

        # Esperar la tecla 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar los recursos
    face_detection.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    deteccion_facial_mediapipe()