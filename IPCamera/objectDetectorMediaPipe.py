import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Clase para la Detección de Objetos ---
class ObjectDetector:
    def __init__(self, model_path='efficientdet_lite0.tflite', use_gpu=False):
        """
        Inicializa el detector de objetos de MediaPipe.

        Args:
            model_path (str): Ruta al archivo del modelo .tflite.
            use_gpu (bool): Si es True, intenta usar el delegado de GPU para acelerar.
        """
        # Configuración del delegado de hardware (GPU o CPU)
        base_options = python.BaseOptions(
            model_asset_path=model_path,
            delegate=python.BaseOptions.Delegate.GPU if use_gpu else python.BaseOptions.Delegate.CPU
        )

        # Opciones para el detector de objetos
        options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)

        # Crear el detector de objetos
        self.detector = vision.ObjectDetector.create_from_options(options)

        print(f"Detector de objetos cargado. Usando GPU: {use_gpu}")

    def detect(self, frame):
        """
        Realiza la detección de objetos en un solo fotograma.

        Args:
            frame (numpy.ndarray): El fotograma de entrada en formato BGR de OpenCV.
        
        Returns:
            list: Una lista de objetos detectados, con sus cuadros delimitadores y etiquetas.
        """
        # Convertir el fotograma de BGR a RGB, que es el formato esperado por MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Crear un objeto Image de MediaPipe a partir del fotograma RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Realizar la detección de objetos
        detection_result = self.detector.detect(mp_image)
        
        return detection_result.detections

    def draw_detections(self, frame, detections):
        """
        Dibuja los cuadros delimitadores y las etiquetas de los objetos detectados en el fotograma.
        
        Args:
            frame (numpy.ndarray): El fotograma de entrada en formato BGR de OpenCV.
            detections (list): La lista de detecciones de objetos.
        
        Returns:
            numpy.ndarray: El fotograma con los cuadros y etiquetas dibujados.
        """
        if not detections:
            return frame

        height, width, _ = frame.shape
        for detection in detections:
            bbox = detection.bounding_box
            category = detection.categories[0]
            
            # Obtener las coordenadas del cuadro delimitador
            x_min = bbox.origin_x
            y_min = bbox.origin_y
            x_max = x_min + bbox.width
            y_max = y_min + bbox.height

            # Dibujar el rectángulo
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Dibujar la etiqueta y la puntuación
            label = f"{category.category_name} ({category.score:.2f})"
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        return frame

# --- Función Principal ---
def main():
    # Inicializar el detector de objetos
    # MediaPipe intentará descargar el modelo automáticamente. Si no lo hace,
    # puedes descargarlo manualmente de:
    # https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite
    detector = ObjectDetector(model_path='efficientdet_lite0.tflite')

    # Inicializar la cámara web
    cap = cv2.VideoCapture(0)  # Usa 0 para la primera cámara web

    if not cap.isOpened():
        print("Error: No se puede abrir la cámara.")
        return

    print("Presiona 'q' para salir.")

    while True:
        # Leer un fotograma de la cámara
        ret, frame = cap.read()
        if not ret:
            print("Error: No se puede recibir el fotograma (stream end?). Saliendo ...")
            break

        # Invertir el fotograma horizontalmente para una vista de espejo
        frame = cv2.flip(frame, 1)

        # Medir el tiempo de inicio para calcular los FPS
        start_time = time.time()

        # Realizar la detección de objetos
        detections = detector.detect(frame)
        
        # Dibujar los resultados en el fotograma
        output_frame = detector.draw_detections(frame, detections)
        
        # Calcular los FPS (Frames por Segundo)
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        cv2.putText(output_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Mostrar el fotograma con los resultados
        cv2.imshow('Deteccion de Objetos con MediaPipe', output_frame)

        # Salir del bucle si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la cámara y destruir todas las ventanas
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()