import cv2
import mediapipe as mp
import threading
import time

# --- Clase para manejar el flujo de una cámara en un hilo separado (Legacy) ---
class CameraThreadLegacy:
    def __init__(self, camera_source, window_name):
        self.camera_source = camera_source
        self.window_name = window_name
        
        # Inicializar la captura de video
        self.cap = cv2.VideoCapture(camera_source) 
        
        if not self.cap.isOpened():
            print(f"Error: No se pudo abrir la cámara {camera_source}. Asegúrate de que la URL o ID sea correcta y accesible.")
            self.cap = None
            return

        # Inicializar MediaPipe Pose para este hilo (Legacy)
        # model_complexity: 0 (light), 1 (full), 2 (heavy). Mayor = más precisión, menor velocidad.
        # min_detection_confidence: Umbral para la detección inicial de la persona.
        # min_tracking_confidence: Umbral para el seguimiento de los puntos clave.
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False, # Pon True si quieres la máscara de segmentación (puede ser más lento)
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Utilidades de dibujo de MediaPipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.frame = None
        self.ret = False
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True # El hilo se cerrará cuando el programa principal termine

        print(f"Cámara '{camera_source}' inicializada para ventana '{window_name}'.")
        print(f"Utilizando MediaPipe Pose (Legacy) para cámara '{camera_source}'.")

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

            # Voltear la imagen horizontalmente (opcional, común para cámaras web)
            frame = cv2.flip(frame, 1)

            # Convertir la imagen de BGR a RGB, ya que MediaPipe espera imágenes RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Hacer la imagen no editable para mejorar el rendimiento con MediaPipe
            image_rgb.flags.writeable = False

            # Procesar la imagen con MediaPipe Pose
            results = self.pose_detector.process(image_rgb)

            # Hacer la imagen editable de nuevo
            image_rgb.flags.writeable = True

            # Dibujar los puntos clave y las conexiones de la postura
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                    # connection_drawing_spec=self.mp_drawing_styles.get_default_pose_connections_style() # Puedes personalizar los estilos
                )
            
            self.frame = frame # Almacenar el frame procesado
            self.ret = ret
            time.sleep(0.001) # Pequeña pausa para evitar saturar el CPU/GPU

    def start(self):
        if self.cap is not None:
            self.thread.start()

    def get_frame(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
        self.pose_detector.close() # Importante cerrar el detector en la versión legacy
        self.thread.join() # Esperar a que el hilo termine



### Función Principal (Uso de Cámaras IP/USB)


def deteccion_doble_camara_legacy():
    # --- ¡CAMBIA ESTAS FUENTES por las de tus cámaras IP o IDs de cámaras USB! ---
    # Para cámaras USB: usa el ID numérico (ej., 0, 1)
    # Para cámaras IP: usa la URL completa de la transmisión (RTSP, HTTP, etc.)
    
    # EJEMPLOS:
    # cam1_source = 0 # Para una webcam USB (típicamente la integrada)
    # cam2_source = 1 # Para una segunda webcam USB (o una externa)
    # cam1_source = 'rtsp://user:password@192.168.1.100:554/stream1' # Tu cámara IP 1
    # cam2_source = '[http://192.168.1.101:8080/video](http://192.168.1.101:8080/video)' # Tu cámara IP 2

    # Configuración por defecto para probar (cambia a tus cámaras reales)
    cam1_source = 'rtsp://admin:321_monzad@192.168.1.110/Streaming/channels/1602' # Asume una webcam USB en ID 0
    cam2_source = 'rtsp://admin:321_monzad@192.168.1.110/Streaming/channels/1002' # Asume una webcam USB en ID 1

    cam1_thread = CameraThreadLegacy(camera_source=cam1_source, window_name="Camara 1: Deteccion de Pose (Legacy)")
    cam2_thread = CameraThreadLegacy(camera_source=cam2_source, window_name="Camara 2: Deteccion de Pose (Legacy)")

    # Verificar si al menos una cámara se inicializó correctamente
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
        # Obtener frames de forma segura, incluso si un hilo no inició o falló
        ret1, frame1 = (cam1_thread.get_frame() if cam1_thread.cap is not None and cam1_thread.running else (False, None))
        ret2, frame2 = (cam2_thread.get_frame() if cam2_thread.cap is not None and cam2_thread.running else (False, None))

        if ret1:
            cv2.imshow(cam1_thread.window_name, frame1)
        elif cam1_thread.cap is not None and not cam1_thread.running:
            print(f"Cámara '{cam1_thread.camera_source}' ha dejado de funcionar permanentemente.")
            break 

        if ret2:
            cv2.imshow(cam2_thread.window_name, frame2)
        elif cam2_thread.cap is not None and not cam2_thread.running:
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
    deteccion_doble_camara_legacy()