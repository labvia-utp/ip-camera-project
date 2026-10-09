import cv2
import mediapipe as mp
import threading
import time

# --- Clase para manejar el flujo de una cámara en un hilo separado (Legacy) ---
class CameraThreadLegacy:
    def __init__(self, camera_source, window_name, output_size_factor=0.5):
        self.camera_source = camera_source
        self.window_name = window_name
        self.output_size_factor = output_size_factor # Factor para reducir el tamaño de salida
        
        # Inicializar la captura de video
        self.cap = cv2.VideoCapture(camera_source) 
        
        if not self.cap.isOpened():
            print(f"Error: No se pudo abrir la cámara {camera_source}. Asegúrate de que la URL o ID sea correcta y accesible.")
            self.cap = None
            return

        # Inicializar MediaPipe Pose para este hilo (Legacy)
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1, # Puedes probar con 0 o 2 para rendimiento/precisión
            enable_segmentation=False, 
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
                print(f"Advertencia: No se pudo leer el frame de la cámara {self.camera_source}. Intentando reconectar en 3 segundos...")
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
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
                    connection_drawing_spec=self.mp_drawing_styles.get_default_pose_connections_style()
                )
            
            # --- REDIMENSIONAR LA IMAGEN DE SALIDA ---
            if self.output_size_factor != 1.0:
                width = int(frame.shape[1] * self.output_size_factor)
                height = int(frame.shape[0] * self.output_size_factor)
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

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
        self.pose_detector.close()
        self.thread.join()

'''

### Función Principal (Manejo de 4 Cámaras)

```python'''
def deteccion_cuatro_camaras_legacy():
    # --- ¡CONFIGURA AQUÍ LAS FUENTES DE TUS 4 CÁMARAS! ---
    # Puedes mezclar IDs de cámaras USB y URLs de cámaras IP.
    # Si no tienes 4 cámaras, puedes repetir el ID de una cámara USB existente para probar,
    # pero recuerda que el rendimiento puede sufrir si intentas leer de la misma cámara múltiples veces.
    
    # EJEMPLOS:
    # cam_sources = [0, 1, 2, 3] # Para 4 webcams USB
    # cam_sources = [
    #     'rtsp://user:pass@192.168.1.100:554/stream',
    #     'rtsp://user:pass@192.168.1.101:554/stream',
    #     0, # Una webcam USB
    #     '[http://192.168.1.102:8080/video](http://192.168.1.102:8080/video)'
    # ]

    # Configuración por defecto para probar (cambia a tus cámaras reales)
    cam_sources = ['rtsp://admin:321_monzad@192.168.1.110/Streaming/channels/902', 
                   'rtsp://admin:321_monzad@192.168.1.110/Streaming/channels/1002'
                   'rtsp://admin:321_monzad@192.168.1.110/Streaming/channels/1602'
                    ]     # ,                     'rtsp://admin:321_monzad@192.168.1.110/Streaming/channels/1602'

    # Crear una lista de hilos para las cámaras
    camera_threads = []
    for i, source in enumerate(cam_sources):
        thread = CameraThreadLegacy(
            camera_source=source, 
            window_name=f"Camara {i+1}: Deteccion de Pose (Legacy)",
            output_size_factor=0.8 # Reducir la salida a la mitad del tamaño original
        )
        if thread.cap is not None: # Solo añade hilos para cámaras que se pudieron abrir
            camera_threads.append(thread)
        else:
            print(f"La cámara {source} no pudo ser iniciada, no se añadirá al procesamiento.")

    if not camera_threads:
        print("Ninguna cámara se pudo iniciar. Saliendo.")
        return

    # Iniciar todos los hilos de las cámaras
    for thread in camera_threads:
        thread.start()

    print("Presiona 'q' para salir de cualquier ventana.")

    while True:
        all_stopped = True # Bandera para saber si todos los hilos activos han parado

        for thread in camera_threads:
            ret, frame = thread.get_frame()
            if ret:
                cv2.imshow(thread.window_name, frame)
                all_stopped = False # Al menos un hilo sigue produciendo frames
            elif thread.running: # Si el hilo sigue "running" pero no produce frames (falló temporalmente)
                all_stopped = False

        # Condición de salida: 'q' presionado O (todos los hilos activos han terminado/fallado)
        if (cv2.waitKey(1) & 0xFF == ord('q')) or all_stopped:
            break

    # Asegurarse de detener y liberar los recursos de todos los hilos
    for thread in camera_threads:
        thread.stop()

    cv2.destroyAllWindows()
    print("Programa finalizado.")

if __name__ == '__main__':
    deteccion_cuatro_camaras_legacy()