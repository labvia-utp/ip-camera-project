import cv2
import mediapipe as mp
import threading
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# --- Clase para manejar el flujo de una cámara (ahora soporta URL) ---
class CameraThread:
    def __init__(self, camera_source, model_path, window_name):
        self.camera_source = camera_source
        self.window_name = window_name
        
        self.cap = cv2.VideoCapture(camera_source) 
        
        if not self.cap.isOpened():
            print(f"Error: No se pudo abrir la cámara {camera_source}. Asegúrate de que la URL o ID sea correcta y accesible.")
            self.cap = None
            return

        # Configuración del detector de pose para este hilo
        base_options = python.BaseOptions(
            model_asset_path=model_path,
            delegate=python.BaseOptions.Delegate.GPU # Intenta usar GPU
        )
        
        # --- CORRECCIÓN AQUÍ: Nombres de argumentos para PoseLandmarkerOptions ---
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            # Argumentos corregidos para la confianza de detección y seguimiento
            min_pose_detection_confidence=0.5, # Antes era min_detection_confidence
            min_tracking_confidence=0.5       # Este nombre se mantiene pero su significado es más específico para el seguimiento de los landmarks
        )
        # --- FIN DE CORRECCIÓN ---

        self.detector = vision.PoseLandmarker.create_from_options(options)

        self.frame = None
        self.ret = False
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True 

        # Utilidades de dibujo
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

        print(f"Cámara '{camera_source}' inicializada para ventana '{window_name}'.")
        print(f"Intentando usar GPU para cámara '{camera_source}'.")


    def _run(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print(f"Advertencia: No se pudo leer el frame de la cámara {self.camera_source}. Intentando reconectar...")
                self.cap.release()
                time.sleep(3) # Espera y intenta reconectar
                self.cap = cv2.VideoCapture(self.camera_source)
                if not self.cap.isOpened():
                    print(f"Error: No se pudo reconectar con la cámara {self.camera_source}. Deteniendo hilo.")
                    self.running = False
                continue 

            frame = cv2.flip(frame, 1) # Opcional: Voltear la imagen

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.RGB, data=image_rgb)

            detection_result = self.detector.detect(mp_image)

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

            self.frame = frame
            self.ret = ret
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
        self.thread.join()

# --- Función principal para manejar ambas cámaras ---
def deteccion_doble_camara_ip():
    model_path = 'pose_landmarker.task' 

    # --- ¡CAMBIA ESTAS URLs por las de tus cámaras IP o usa IDs de cámara USB! ---
    # Ejemplo de URL RTSP (común para cámaras IP)
    # Algunas cámaras requieren usuario y contraseña: 'rtsp://user:password@ip_address:port/path_to_stream'
    # Consulta la documentación de tu cámara para la URL exacta.
    
    # EJEMPLOS:
    # cam1_source = 0 # Para una webcam USB (típicamente la integrada)
    # cam2_source = 1 # Para una segunda webcam USB (o una externa)
    # cam1_source = 'rtsp://admin:password@192.168.1.100:554/stream1' # Tu cámara IP 1
    # cam2_source = 'http://192.168.1.101:8080/video' # Tu cámara IP 2 (o una segunda webcam USB)

    # Configuración por defecto para probar (cambia a tus cámaras reales)
    cam1_source = 'rtsp://admin:321_monzad@192.168.1.110/Streaming/channels/1602'
    cam2_source = 'rtsp://admin:321_monzad@192.168.1.110/Streaming/channels/1002'

    cam1_thread = CameraThread(camera_source=cam1_source, model_path=model_path, window_name="Camara 1: Deteccion de Pose")
    cam2_thread = CameraThread(camera_source=cam2_source, model_path=model_path, window_name="Camara 2: Deteccion de Pose")

    # Verificar si al menos una cámara se inicializó correctamente
    if cam1_thread.cap is None and cam2_thread.cap is None:
        print("No se pudo iniciar ninguna de las cámaras. Saliendo.")
        return

    if cam1_thread.cap is not None:
        cam1_thread.start()
    if cam2_thread.cap is not None:
        cam2_thread.start()

    print("Presiona 'q' para salir de cualquier ventana.")

    while True:
        # Obtener frames de forma segura, incluso si un hilo no inició o falló
        ret1, frame1 = (cam1_thread.get_frame() if cam1_thread.cap is not None and cam1_thread.running else (False, None))
        ret2, frame2 = (cam2_thread.get_frame() if cam2_thread.cap is not None and cam2_thread.running else (False, None))

        if ret1:
            cv2.imshow(cam1_thread.window_name, frame1)
        elif cam1_thread.cap is not None and not cam1_thread.running:
            # Si el hilo de la cámara 1 falló después de iniciar, podemos detener el bucle
            print(f"Cámara '{cam1_thread.camera_source}' ha dejado de funcionar permanentemente.")
            break 

        if ret2:
            cv2.imshow(cam2_thread.window_name, frame2)
        elif cam2_thread.cap is not None and not cam2_thread.running:
            # Si el hilo de la cámara 2 falló después de iniciar, podemos detener el bucle
            print(f"Cámara '{cam2_thread.camera_source}' ha dejado de funcionar permanentemente.")
            break

        # Condición de salida: 'q' presionado O (ambos hilos que iniciaron han terminado/fallado)
        if (cv2.waitKey(1) & 0xFF == ord('q')) or \
           ( (cam1_thread.cap is None or not cam1_thread.running) and \
             (cam2_thread.cap is None or not cam2_thread.running) ):
            break

    # Asegurarse de detener y liberar los recursos de cualquier hilo que haya iniciado
    if cam1_thread.cap is not None:
        cam1_thread.stop()
    if cam2_thread.cap is not None:
        cam2_thread.stop()

    cv2.destroyAllWindows()
    print("Programa finalizado.")

if __name__ == '__main__':
    deteccion_doble_camara_ip()