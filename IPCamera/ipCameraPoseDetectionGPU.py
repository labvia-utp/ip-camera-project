import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2 # Para dibujar los landmarks

def deteccion_corporal_mediapipe_gpu():
    # --- Configuración para usar GPU ---
    # Es crucial que tu entorno tenga CUDA (para NVIDIA GPUs) o la implementación
    # de GPU correspondiente para tu tarjeta gráfica y que MediaPipe pueda detectarla.
    # Para PoseLandmarker, hay un problema conocido donde la GPU no se usa actualmente.
    # Sin embargo, esta es la forma correcta de intentar habilitarla.
    base_options = python.BaseOptions(
        model_asset_path='pose_landmarker.task',
        delegate=python.BaseOptions.Delegate.GPU # ¡Aquí se especifica la GPU!
    )

    # Si quieres forzar la CPU (para depuración o si la GPU no funciona)
    # base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')

    # Configurar las opciones del detector de postura
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False, # Pon True si quieres la máscara de segmentación
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Crear el detector de postura
    detector = vision.PoseLandmarker.create_from_options(options)

    url_rtsp = 'rtsp://admin:321_monzad@192.168.1.110/Streaming/channels/1001' # Reemplaza con tu URL RTSP  rtsp://admin:Admin8350@192.168.31.168:65003/8000

    # Inicializar la captura de video desde la cámara web
    cap = cv2.VideoCapture(url_rtsp)

    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    print("Presiona 'q' para salir.")
    print("Intentando usar GPU. Ten en cuenta que PoseLandmarker tiene un problema conocido con la GPU en la API de Python.")

    # Utilidades de dibujo de MediaPipe
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose # Necesario para POSE_CONNECTIONS

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame de la cámara.")
            break

        # Voltear la imagen horizontalmente (opcional)
        frame = cv2.flip(frame, 1)

        # Convertir la imagen de BGR a RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Crear un objeto Image de MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.RGB, data=image_rgb)

        # Realizar la detección de postura
        detection_result = detector.detect(mp_image)

        # Dibujar los landmarks si se detectaron
        if detection_result.pose_landmarks:
            for pose_landmarks in detection_result.pose_landmarks:
                # Convertir los landmarks a un formato que mp_drawing pueda entender
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                pose_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in pose_landmarks
                ])
                mp_drawing.draw_landmarks(
                    frame,
                    pose_landmarks_proto,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )

        # Mostrar el frame procesado
        cv2.imshow('Deteccion Corporal con MediaPipe Pose (GPU intentado)', frame)

        # Esperar la tecla 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar los recursos
    cap.release()
    cv2.destroyAllWindows()
    detector.close() # Es importante cerrar el detector para liberar recursos

if __name__ == '__main__':
    deteccion_corporal_mediapipe_gpu()