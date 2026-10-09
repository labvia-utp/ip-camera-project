import cv2
import mediapipe as mp
import threading
import time
import av  # PyAV (usa FFmpeg por debajo)
from mediapipe.framework.formats import landmark_pb2


# --- Clase para manejar el flujo de una cámara en un hilo separado (Legacy) ---
class CameraThreadLegacy:
    def __init__(self, camera_source, window_name, output_size_factor=0.5, rtsp_transport="udp"):
        self.last_pose_time = 0.0          # (nuevo) cuándo fue la última detección válida
        self.pose_ttl_sec = 0.35           # (nuevo) cuánto tiempo mantener el dibujo (ajústalo 0.2–0.6)
        self.camera_source = camera_source
        self.window_name = window_name
        self.output_size_factor = output_size_factor # Factor para reducir el tamaño de salida
        self.rtsp_transport = rtsp_transport  # (PyAV) 'udp' suele dar menos latencia; si se corta, usar 'tcp'

        # Inicializar la captura de video
        # (PyAV) En vez de cv2.VideoCapture, abrimos RTSP con av.open dentro del hilo para poder reconectar mejor.
        # self.cap se deja como "bandera" para que el código principal no cambie tanto.
        self.cap = True

        # (PyAV) Opciones mínimas (en tus pruebas, esto dio la menor latencia)
        self.av_opts = {
            "rtsp_transport": self.rtsp_transport,
            "stimeout": "5000000",  # 5s
        }

        # Prueba rápida de conexión (opcional). Si prefieres evitarla, puedes comentar este bloque.
        try:
            _tmp = av.open(self.camera_source, options=self.av_opts)
            _tmp.close()
        except Exception as e:
            print(f"Error: No se pudo abrir la cámara {camera_source}. Asegúrate de que la URL o ID sea correcta y accesible.")
            print(f"Detalle PyAV: {e}")
            self.cap = None
            return

        # Inicializar MediaPipe Pose para este hilo (Legacy)
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0, # Puedes probar con 0 o 2 para rendimiento/precisión
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Utilidades de dibujo de MediaPipe
        self.mp_drawing = mp.solutions.drawing_utils
        # --- CORRECCIÓN AQUÍ: Definir DrawingSpec directamente ---
        self.landmark_drawing_spec = self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2)
        self.connection_drawing_spec = self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        # --- FIN DE CORRECCIÓN ---

        self.frame = None
        self.ret = False
        self.last_landmarks = None          # (nuevo) últimos landmarks para dibujar en frames skipped
        self.smoothed_landmarks = None      # (nuevo) landmarks suavizados
        self.smooth_alpha = 0.5             # (nuevo) 0.3-0.7 típico (más alto = más reactivo)
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True # El hilo se cerrará cuando el programa principal termine

        print(f"Cámara '{camera_source}' inicializada para ventana '{window_name}'.")
        print(f"Utilizando MediaPipe Pose (Legacy) para cámara '{camera_source}'.")

    def _ema_landmarks(self, prev, new, alpha=0.5):
        """Suavizado exponencial (EMA) de landmarks para reducir jitter/flicker."""
        if prev is None:
            return new

        out = landmark_pb2.NormalizedLandmarkList()
        for p, n in zip(prev.landmark, new.landmark):
            lm = landmark_pb2.NormalizedLandmark(
                x=alpha * n.x + (1 - alpha) * p.x,
                y=alpha * n.y + (1 - alpha) * p.y,
                z=alpha * n.z + (1 - alpha) * p.z,
                visibility=alpha * n.visibility + (1 - alpha) * p.visibility
            )
            out.landmark.append(lm)
        return out


    def _run(self):
        # (PyAV) Reemplazo del bucle cap.read() por decode() con reconexión
        frame_idx = 0              # (nuevo) contador de frames
        SKIP = 2                   # (nuevo) 2 = procesa pose 1 de cada 2; 3 = 1 de cada 3
        # !!!WARNING!!! AJUSTAR SKIP A 1 AGREGARA MUCHO DELAY
        while self.running:
            try:
                container = av.open(self.camera_source, options=self.av_opts)

                # (PyAV) IMPORTANTE: NO seteamos stream.thread_type="AUTO"
                # porque a ustedes les aumentó la latencia en pruebas.
                for f in container.decode(video=0):
                    if not self.running:
                        break

                    frame_idx += 1
                    frame = f.to_ndarray(format="bgr24")
                    ret = True

                    # Voltear la imagen horizontalmente (opcional, común para cámaras web)
                    # frame = cv2.flip(frame, 1)

                    # (nuevo) Calculamos Pose solo cada SKIP frames para reducir delay y mantener sync
                    do_pose = (frame_idx % SKIP == 0)

                    if do_pose:
                        # Convertir la imagen de BGR a RGB, ya que MediaPipe espera imágenes RGB
                        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # Hacer la imagen no editable para mejorar el rendimiento con MediaPipe
                        image_rgb.flags.writeable = False

                        # Procesar la imagen con MediaPipe Pose
                        results = self.pose_detector.process(image_rgb)

                        # Hacer la imagen editable de nuevo
                        image_rgb.flags.writeable = True

                        # (nuevo) Guardar landmarks (y suavizar) si hay detección
                        if results.pose_landmarks:
                            self.last_pose_time = time.time()
                            self.last_landmarks = results.pose_landmarks
                            self.smoothed_landmarks = self._ema_landmarks(
                                self.smoothed_landmarks,
                                results.pose_landmarks,
                                alpha=self.smooth_alpha
                            )

                    # (nuevo) Dibujar SIEMPRE el último pose "fresco" para evitar flicker
                    now = time.time()
                    pose_is_fresh = (now - self.last_pose_time) <= self.pose_ttl_sec

                    if not pose_is_fresh:
                        # (nuevo) expiró: borrar pose guardado para que no quede pegado
                        self.last_landmarks = None
                        self.smoothed_landmarks = None

                    if pose_is_fresh:
                        if self.smoothed_landmarks is not None:
                            self.mp_drawing.draw_landmarks(
                                frame,
                                self.smoothed_landmarks,
                                self.mp_pose.POSE_CONNECTIONS,
                                landmark_drawing_spec=self.landmark_drawing_spec,
                                connection_drawing_spec=self.connection_drawing_spec
                            )
                        elif self.last_landmarks is not None:
                            self.mp_drawing.draw_landmarks(
                                frame,
                                self.last_landmarks,
                                self.mp_pose.POSE_CONNECTIONS,
                                landmark_drawing_spec=self.landmark_drawing_spec,
                                connection_drawing_spec=self.connection_drawing_spec
                            )

                    # --- REDIMENSIONAR LA IMAGEN DE SALIDA ---
                    if self.output_size_factor != 1.0:
                        width = int(frame.shape[1] * self.output_size_factor)
                        height = int(frame.shape[0] * self.output_size_factor)
                        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

                    self.frame = frame # Almacenar el frame procesado
                    self.ret = ret
                    time.sleep(0.001) # Pequeña pausa para evitar saturar el CPU/GPU

                container.close()

            except Exception as e:
                print(f"Advertencia: No se pudo leer el frame de la cámara {self.camera_source}. Intentando reconectar en 3 segundos...")
                # (PyAV) Igual que el comportamiento anterior, esperamos y reconectamos
                time.sleep(3) # Espera y intenta reconectar
                continue

    def start(self):
        if self.cap is not None:
            self.thread.start()

    def get_frame(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        self.pose_detector.close()
        self.thread.join()


def leer_lineas_en_lista(nombre_archivo):
    """
    Lee un archivo de texto y devuelve una lista con cada línea como un elemento.

    Args:
        nombre_archivo (str): El nombre del archivo de texto a leer.

    Returns:
        list: Una lista donde cada elemento es una línea del archivo.
    """
    lineas = []
    # Usamos 'with' para asegurarnos de que el archivo se cierre correctamente
    with open(nombre_archivo, 'r') as archivo:
        # Itera sobre cada línea del archivo
        for linea in archivo:
            # Elimina los espacios en blanco y saltos de línea al inicio y final
            linea = linea.strip()
            # (PyAV) opcional: ignorar líneas vacías o comentarios
            if not linea or linea.startswith("#"):
                continue
            lineas.append(linea)
    return lineas


## Función Principal (Manejo de 4 Cámaras)

def deteccion_cuatro_camaras_legacy():
    # --- ¡CONFIGURA AQUÍ LAS FUENTES DE TUS 4 CÁMARAS! ---
    # Puedes mezclar IDs de cámaras USB y URLs de cámaras IP.
    # Si no tienes 4 cámaras físicas, puedes repetir el ID de una cámara USB existente para probar,
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
    '''cam_sources = ['rtsp://admin:321_monzad@192.168.1.110/Streaming/channels/901',
                   'rtsp://admin:321_monzad@192.168.1.110/Streaming/channels/1001',
                   'rtsp://admin:321_monzad@192.168.1.110/Streaming/channels/802',
                   'rtsp://admin:321_monzad@192.168.1.110/Streaming/channels/1602'
                    ]'''

    # Nombre del archivo que quieres leer
    archivo_a_leer = 'listaCamaras.txt'

    # Llama a la función y guarda el resultado en una variable
    cam_sources = leer_lineas_en_lista(archivo_a_leer)

    # El factor de escala para la imagen de salida (0.5 significa la mitad del tamaño)
    output_scale_factor = 0.958

    # (PyAV) En Wi-Fi, si UDP te da frames corruptos o caídas, cambia a "tcp"
    rtsp_transport = "udp"

    # Crear una lista de hilos para las cámaras
    camera_threads = []
    for i, source in enumerate(cam_sources):
        thread = CameraThreadLegacy(
            camera_source=source,
            window_name=f"Camara {i+1}: Deteccion de Pose",
            output_size_factor=output_scale_factor,
            rtsp_transport=rtsp_transport
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
    deteccion_cuatro_camaras_legacy()

# Sirve bien, solo que ahora cuando una persona sale del frame, queda el dibujo de su pose en la pantalla siempre, hasta que vuelva a entrar, lo cual no es deseable.