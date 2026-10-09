import cv2
import mediapipe as mp
import threading
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Clase para manejar el flujo de una cámara en un hilo separado (Detección de Objetos) ---
class CameraThreadObject:
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

        # --- Inicializar el detector de objetos ---
        model_path = 'efficientdet_lite0.tflite'
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
        self.object_detector = vision.ObjectDetector.create_from_options(options)
        
        self.frame = None
        self.ret = False
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True

        print(f"Cámara '{camera_source}' inicializada para ventana '{window_name}'.")
        print(f"Utilizando MediaPipe Object Detection para cámara '{camera_source}'.")

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
            
            # Convertir a MP Image para MediaPipe Tasks API
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Procesar la imagen con MediaPipe Object Detection
            detection_result = self.object_detector.detect(mp_image)
            
            # --- Dibujar las detecciones en el frame ---
            if detection_result.detections:
                for detection in detection_result.detections:
                    bbox = detection.bounding_box
                    category = detection.categories[0]
                    
                    x_min = bbox.origin_x
                    y_min = bbox.origin_y
                    x_max = x_min + bbox.width
                    y_max = y_min + bbox.height

                    # Dibujar el rectángulo
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    # Dibujar la etiqueta y la puntuación
                    label = f"{category.category_name} ({category.score:.2f})"
                    cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
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
        self.object_detector.close()
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

# Función Principal (Manejo de N Cámaras con Detección de Objetos)
def deteccion_multicamaras_objetos():
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
        try:
            # Intentar convertir la fuente a un entero (para IDs de cámara USB)
            source_id = int(source)
            thread = CameraThreadObject(
                camera_source=source_id,
                window_name=f"Camara {i+1}: Deteccion de Objetos",
                output_size_factor=output_scale_factor
            )
        except ValueError:
            # Si no es un entero, asumimos que es una URL (RTSP, HTTP, etc.)
            thread = CameraThreadObject(
                camera_source=source,
                window_name=f"Camara {i+1}: Deteccion de Objetos",
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
    deteccion_multicamaras_objetos()