import cv2
import mediapipe as mp
import threading
import time

# --- Clase para manejar el flujo de una cámara en un hilo separado (Detección Facial) ---
class CameraThreadFace:
    def __init__(self, camera_source, window_name, output_size_factor=0.5):
        self.camera_source = camera_source
        self.window_name = window_name
        self.output_size_factor = output_size_factor
        
        # Inicializar la captura de video
        self.cap = cv2.VideoCapture(camera_source) 
        
        if not self.cap.isOpened():
            print(f"Error: No se pudo abrir la cámara {camera_source}. Asegúrate de que la URL o ID sea correcta y accesible.")
            self.cap = None
            return

        # Inicializar MediaPipe Face Mesh para este hilo
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh_detector = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=4,  # Máximo número de caras a detectar
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Utilidades de dibujo de MediaPipe
        self.mp_drawing = mp.solutions.drawing_utils
        # --- Configuración para dibujar las conexiones faciales ---
        self.landmark_drawing_spec = self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=1, circle_radius=1)
        # Usamos las conexiones predefinidas para Face Mesh
        self.connection_drawing_spec = self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=1)

        self.frame = None
        self.ret = False
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True

        print(f"Cámara '{camera_source}' inicializada para ventana '{window_name}'.")
        print(f"Utilizando MediaPipe Face Mesh para cámara '{camera_source}'.")

    def _run(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print(f"Advertencia: No se pudo leer el frame de la cámara {self.camera_source}. Intentando reconectar en 3 segundos...")
                self.cap.release()
                time.sleep(3)
                self.cap = cv2.VideoCapture(self.camera_source)
                if not self.cap.isOpened():
                    print(f"Error: No se pudo reconectar con la cámara {self.camera_source}. Deteniendo hilo.")
                    self.running = False
                continue

            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False

            # Procesar la imagen con MediaPipe Face Mesh
            results = self.face_mesh_detector.process(image_rgb)

            image_rgb.flags.writeable = True

            # Dibujar los puntos clave y las conexiones faciales para CADA cara detectada
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS, # FACEMESH_CONTOURS para un contorno detallado
                        landmark_drawing_spec=self.landmark_drawing_spec,
                        connection_drawing_spec=self.connection_drawing_spec
                    )
            
            # Redimensionar la imagen de salida
            if self.output_size_factor != 1.0:
                width = int(frame.shape[1] * self.output_size_factor)
                height = int(frame.shape[0] * self.output_size_factor)
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

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
        self.face_mesh_detector.close()
        self.thread.join()

# Función para leer las fuentes de las cámaras desde un archivo de texto
def leer_lineas_en_lista(nombre_archivo):
    lineas = []
    try:
        with open(nombre_archivo, 'r') as archivo:
            for linea in archivo:
                lineas.append(linea.strip())
    except FileNotFoundError:
        print(f"Error: El archivo '{nombre_archivo}' no se encontró.")
        return []
    return lineas

# Función Principal (Manejo de 4 Cámaras)
def deteccion_cuatro_camaras_facial():
    # Nombre del archivo que quieres leer
    archivo_a_leer = 'listaCamaras.txt'
    cam_sources = leer_lineas_en_lista(archivo_a_leer)

    if not cam_sources:
        print("No se encontraron fuentes de cámara en el archivo. Saliendo.")
        return

    # El factor de escala para la imagen de salida
    output_scale_factor = 0.958

    # Crear una lista de hilos para las cámaras
    camera_threads = []
    for i, source in enumerate(cam_sources):
        thread = CameraThreadFace(
            camera_source=source,
            window_name=f"Camara {i+1}: Deteccion Facial",
            output_size_factor=output_scale_factor
        )
        if thread.cap is not None:
            camera_threads.append(thread)
        else:
            print(f"La cámara '{source}' no pudo ser iniciada, no se añadirá al procesamiento.")

    if not camera_threads:
        print("Ninguna cámara se pudo iniciar. Saliendo.")
        return

    # Iniciar todos los hilos de las cámaras
    for thread in camera_threads:
        thread.start()

    print("\nPresiona 'q' para salir de cualquier ventana.")

    while True:
        all_stopped = True
        for thread in camera_threads:
            ret, frame = thread.get_frame()
            if ret:
                cv2.imshow(thread.window_name, frame)
                all_stopped = False
            elif thread.running:
                all_stopped = False

        if (cv2.waitKey(1) & 0xFF == ord('q')) or all_stopped:
            break

    for thread in camera_threads:
        thread.stop()

    cv2.destroyAllWindows()
    print("Programa finalizado.")

if __name__ == '__main__':
    deteccion_cuatro_camaras_facial()