import cv2
import mediapipe as mp

def deteccion_corporal_mediapipe():
    # Inicializar MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles # Para estilos de dibujo más bonitos

    # Configurar la detección de pose
    # static_image_mode=False: Para procesamiento de video (más rápido para video)
    # model_complexity: 0, 1 o 2 (siendo 2 el más preciso pero más lento)
    # enable_segmentation=False: No necesitamos la segmentación del fondo para solo detección de pose
    # min_detection_confidence: umbral para la detección inicial del cuerpo
    # min_tracking_confidence: umbral para el seguimiento de los puntos clave
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1, # Puedes probar 0, 1 o 2
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    url_rtsp = 'rtsp://admin:321_monzad@192.168.1.110/Streaming/channels/1602' # Reemplaza con tu URL RTSP  rtsp://admin:Admin8350@192.168.31.168:65003/8000

    # Inicializar la captura de video desde la cámara web
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

        # Voltear la imagen horizontalmente para una vista tipo espejo (opcional)
        frame = cv2.flip(frame, 1)

        # Convertir la imagen de BGR a RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Para mejorar el rendimiento, marca la imagen como no escribible para pasarla por referencia.
        image_rgb.flags.writeable = False
        
        # Procesar la imagen con MediaPipe Pose
        results = pose.process(image_rgb)

        # Volver a marcar la imagen como escribible antes de dibujar
        image_rgb.flags.writeable = True
        
        # Convertir de nuevo a BGR para mostrar con OpenCV
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Dibujar los puntos clave y las conexiones del esqueleto
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image_bgr,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
        # Mostrar el frame procesado
        cv2.imshow('Deteccion Corporal con MediaPipe', image_bgr)

        # Esperar la tecla 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar los recursos
    pose.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    deteccion_corporal_mediapipe()