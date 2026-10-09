import cv2
import mediapipe as mp
import threading
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2 # Para dibujar los landmarks

# --- Clase para manejar el flujo de una cámara en un hilo separado ---
class CameraThread:
    def __init__(self, camera_id, model_path, window_name):
        self.camera_id = camera_id
        self.window_name = window_name
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            print(f"Error: No se pudo abrir la cámara {camera_id}")
            self.cap = None
            return

        # Configuración del detector de pose para este hilo
        base_options = python.BaseOptions(
            model_asset_path=model_path,
            delegate=python.BaseOptions.Delegate.GPU # Intenta usar GPU
        )
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

        self.frame = None
        self.ret = False
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True # El hilo se cerrará cuando el programa principal termine

        # Utilidades de dibujo
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose # Necesario para POSE_CONNECTIONS

        print(f"Cámara {camera_id} inicializada para ventana '{window_name}'.")
        print(f"Intentando usar GPU para cámara {camera_id}.")


    def _run(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print(f"Error: No se pudo leer el frame de la cámara {self.camera_id}.")
                self.running = False
                break

            # Voltear la imagen horizontalmente (opcional)
            frame = cv2.flip(frame, 1)

            # Convertir a RGB y crear mp.Image
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.RGB, data=image_rgb)

            # Realizar la detección
            detection_result = self.detector.detect(mp_image)

            # Dibujar los landmarks
            if detection_result.pose_landmarks:
                for pose_landmarks in detection_result.pose_landmarks:
                    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                    pose_landmarks_proto.landmark.extend([
                        landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in pose_landmarks
                    ])
                    self.mp_drawing.draw_landmarks(
                        frame,
                        pose_landmarks_proto,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                    )

            self.frame = frame # Almacenar el frame procesado
            self.ret = ret
            # Pequeña pausa para no saturar la CPU con lectura de frames si el procesamiento es muy rápido
            time.sleep(0.001)

    def start(self):
        if self.cap is not None:
            self.thread.start()

    def get_frame(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
        self.detector.close()
        self.thread.join() # Esperar a que el hilo termine

# --- Función principal para manejar ambas cámaras ---
def deteccion_doble_camara():
    model_path = 'pose_landmarker.task' # El modelo se descargará si no existe


    camara1 = 'rtsp://admin:321_monzad@192.168.1.110/Streaming/channels/101'
    camara2 = 'rtsp://admin:321_monzad@192.168.1.110/Streaming/channels/201'
    # Crear instancias de CameraThread para cada cámara
    # Asegúrate de que estos IDs de cámara sean correctos para tu sistema
    cam1_thread = CameraThread(camera_id=camara1, model_path=model_path, window_name="Camara 1: Deteccion de Pose")
    cam2_thread = CameraThread(camera_id=camara2, model_path=model_path, window_name="Camara 2: Deteccion de Pose")

    if cam1_thread.cap is None and cam2_thread.cap is None:
        print("No se pudo iniciar ninguna de las cámaras. Saliendo.")
        return

    # Iniciar los hilos de las cámaras
    if cam1_thread.cap is not None:
        cam1_thread.start()
    if cam2_thread.cap is not None:
        cam2_thread.start()

    print("Presiona 'q' para salir de cualquier ventana.")

    while True:
        ret1, frame1 = cam1_thread.get_frame() if cam1_thread.cap is not None else (False, None)
        ret2, frame2 = cam2_thread.get_frame() if cam2_thread.cap is not None else (False, None)

        if ret1:
            cv2.imshow(cam1_thread.window_name, frame1)
        # Asegurarse de que la ventana se muestre incluso si solo una cámara está funcionando
        elif cam1_thread.cap is not None and not cam1_thread.running:
            print(f"Cámara {cam1_thread.camera_id} ha dejado de funcionar.")
            cam1_thread.stop() # Asegurar que el hilo se detenga
            break # Si una cámara falla, puedes decidir si quieres que el programa continúe o salga

        if ret2:
            cv2.imshow(cam2_thread.window_name, frame2)
        elif cam2_thread.cap is not None and not cam2_thread.running:
            print(f"Cámara {cam2_thread.camera_id} ha dejado de funcionar.")
            cam2_thread.stop()
            break


        # Salir si una de las cámaras no está corriendo O si se presiona 'q'
        if (cam1_thread.cap is not None and not cam1_thread.running) or \
           (cam2_thread.cap is not None and not cam2_thread.running) or \
           (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    # Detener ambos hilos y liberar recursos al salir del bucle principal
    if cam1_thread.cap is not None:
        cam1_thread.stop()
    if cam2_thread.cap is not None:
        cam2_thread.stop()

    cv2.destroyAllWindows()
    print("Programa finalizado.")

if __name__ == '__main__':
    deteccion_doble_camara()