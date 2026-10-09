import cv2
import mediapipe as mp
import threading
import time
import numpy as np # Necesario para manejar los arreglos de bytes de ffmpeg
import ffmpeg        # ¡Nueva importación!
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2 

# --- Clase para manejar el flujo de una cámara en un hilo separado (MediaPipe Tasks API) ---
class CameraThreadTasks:
    def __init__(self, camera_source, window_name, model_path, output_size_factor=0.5):
        self.camera_source = camera_source
        self.window_name = window_name
        self.model_path = model_path
        self.output_size_factor = output_size_factor 
        
        self.ffmpeg_process = None
        self.width = None
        self.height = None
        self.frame_size_bytes = None

        try:
            # --- Paso 1: Sondear el stream para obtener ancho y alto ---
            # Esto puede tardar unos segundos, especialmente para streams de red.
            # Los parámetros '-v quiet' suprimen la salida de depuración de ffmpeg.
            # '-select_streams v:0' selecciona solo la primera pista de video.
            # '-show_entries stream=width,height,pix_fmt' obtiene la información de la corriente.
            # '-of json' para la salida en formato JSON.
            probe = ffmpeg.probe(str(camera_source), select_streams='v:0')
            video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
            
            if video_stream:
                self.width = video_stream['width']
                self.height = video_stream['height']
                # Usaremos 'bgr24' para la salida de FFmpeg para que sea compatible directamente con OpenCV
                # y NumPy sin conversiones de color adicionales después de leer los bytes.
                self.pixel_format = 'bgr24' 
                self.frame_size_bytes = self.width * self.height * 3 # 3 bytes por píxel para BGR24

                print(f"Stream '{camera_source}' probed: {self.width}x{self.height}, {self.pixel_format}")

                # --- Paso 2: Configurar el proceso FFmpeg ---
                self.ffmpeg_process = (
                    ffmpeg
                    .input(str(camera_source), 
                           # Puedes añadir opciones aquí, ej:
                           # rtsp_transport='tcp' para RTSP,
                           # vsync='vfr' para asegurar que los frames se entreguen lo más rápido posible.
                           ) 
                    .output('pipe:',                             # Salida a stdout
                            format='rawvideo',                   # Formato de video crudo
                            pix_fmt=self.pixel_format,           # Formato de píxeles BGR24
                            loglevel='quiet'                     # Suprimir mensajes de ffmpeg
                            )
                    .run_async(pipe_stdout=True, pipe_stderr=True) # Ejecutar asíncronamente y capturar stdout/stderr
                )
                print(f"FFmpeg process started for '{camera_source}'.")

            else:
                print(f"Error: No se encontró una pista de video en el stream '{camera_source}'.")
                self.cap = None # Señalamos que no se pudo inicializar
                return

        except ffmpeg.Error as e:
            print(f"Error al iniciar FFmpeg para '{camera_source}':")
            print(f"Stderr: {e.stderr.decode('utf8')}")
            self.ffmpeg_process = None
            self.cap = None # Señalamos que no se pudo inicializar
            return
        except Exception as e:
            print(f"Error inesperado al sondear/iniciar FFmpeg para '{camera_source}': {e}")
            self.ffmpeg_process = None
            self.cap = None # Señalamos que no se pudo inicializar
            return


        # Inicializar MediaPipe Pose para este hilo (MediaPipe Tasks API)
        base_options = python.BaseOptions(
            model_asset_path=self.model_path,
            delegate=python.BaseOptions.Delegate.GPU # ¡Aquí se especifica la GPU!
        )
        
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            min_pose_detection_confidence=0.5, 
            min_tracking_confidence=0.5       
        )

        self.detector = vision.PoseLandmarker.create_from_options(options)

        self.frame = None
        self.ret = False
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True 

        print(f"Cámara '{camera_source}' inicializada para ventana '{window_name}'.")
        print(f"Utilizando MediaPipe Tasks API y modelo '{self.model_path}' para cámara '{camera_source}'.")


    def _run(self):
        while self.running and self.ffmpeg_process and self.ffmpeg_process.poll() is None:
            # --- Leer un frame de FFmpeg ---
            # Leemos la cantidad exacta de bytes para un frame completo
            in_bytes = self.ffmpeg_process.stdout.read(self.frame_size_bytes)
            
            if not in_bytes: # Si no hay bytes, el stream ha terminado o hay un problema
                print(f"Advertencia: No se pudieron leer bytes de FFmpeg para {self.camera_source}. Stream terminado o error.")
                self.running = False
                break

            # Convertir los bytes crudos a un arreglo NumPy (BGR para OpenCV)
            # reshape((height, width, channels))
            frame = np.frombuffer(in_bytes, np.uint8).reshape([self.height, self.width, 3])

            # Voltear la imagen horizontalmente (opcional, común para cámaras web)
            frame = cv2.flip(frame, 1)

            # Convertir a RGB y crear mp.Image para la nueva API
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.RGB, data=image_rgb)

            # Realizar la detección
            detection_result = self.detector.detect(mp_image)

            # Dibujar los landmarks si se detectaron
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
            
            # Redimensionar la imagen de salida
            if self.output_size_factor != 1.0:
                width = int(frame.shape[1] * self.output_size_factor)
                height = int(frame.shape[0] * self.output_size_factor)
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            self.frame = frame 
            self.ret = True # Si llegamos aquí, el frame es válido
            time.sleep(0.001)
        
        # Si el bucle termina, el proceso FFmpeg podría haber terminado
        if self.ffmpeg_process:
            self.ffmpeg_process.wait() # Esperar a que el proceso termine completamente
            print(f"FFmpeg process for '{self.camera_source}' exited with code {self.ffmpeg_process.returncode}")
            # Opcional: leer stderr si hubo errores
            # print(f"FFmpeg Stderr: {self.ffmpeg_process.stderr.read().decode('utf8')}")


    def start(self):
        # Asegurarse de que el proceso ffmpeg se inició correctamente
        if self.ffmpeg_process:
            self.thread.start()
        else:
            print(f"No se pudo iniciar el hilo para '{self.camera_source}' porque FFmpeg no se inicializó correctamente.")

    def get_frame(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        if self.ffmpeg_process:
            self.ffmpeg_process.terminate() # Terminar el proceso ffmpeg
            self.ffmpeg_process.wait(timeout=5) # Esperar a que termine (con timeout)
            if self.ffmpeg_process.poll() is None: # Si no terminó, matarlo
                self.ffmpeg_process.kill()
        self.detector.close() 
        self.thread.join()

# --- Función Principal (Manejo de 4 Cámaras con Pose Detection GPU usando FFmpeg) ---
def deteccion_cuatro_camaras_pose_gpu_ffmpeg():
    # EL MODELO DE MEDIA PIPE POSE LANDMARKER (asegúrate de que este archivo esté en tu directorio)
    model_path = 'pose_landmarker_heavy.task' 

    # --- ¡CONFIGURA AQUÍ LAS FUENTES DE TUS 4 CÁMARAS! ---
    # Para FFmpeg, los IDs de cámara USB suelen ser rutas como '/dev/video0' en Linux
    # o el nombre del dispositivo en Windows si es compatible (aunque IDs numéricos suelen funcionar con cv2).
    # Las URLs de IP (rtsp://, http://) son el fuerte de FFmpeg.
    
    # EJEMPLOS:
    # cam_sources = [0, 1, 2, 3] # Esto NO funcionará directamente con ffmpeg.input si son IDs de CV2.
                               # Para webcams con FFmpeg en Linux, usaría algo como:
                               # '/dev/video0', '/dev/video1', ...
                               # Para Windows, podría ser necesario un backend como dshow:
                               # 'video=Integrated Webcam:dshow' (requiere opciones ffmpeg adicionales)

    cam_sources = [
        'rtsp://user:pass@192.168.1.100:554/stream', # URL de tu Cámara IP 1
        'rtsp://user:pass@192.168.1.101:554/stream', # URL de tu Cámara IP 2
        'rtsp://user:pass@192.168.1.102:554/stream', # URL de tu Cámara IP 3
        'rtsp://user:pass@192.168.1.103:554/stream'  # URL de tu Cámara IP 4
        # Si no tienes 4 IPs, puedes repetir una para probar, o usar menos hilos.
    ]
    # Si solo tienes 1 cámara USB y quieres probar FFmpeg:
    # cam_sources = ['/dev/video0'] # Para Linux
    # cam_sources = ['video=Integrated Webcam:dshow'] # Ejemplo para Windows (puede requerir más opciones ffmpeg)


    output_scale_factor = 0.5 

    camera_threads = []
    for i, source in enumerate(cam_sources):
        thread = CameraThreadTasks( 
            camera_source=source, 
            window_name=f"Camara {i+1}: Deteccion de Pose (FFmpeg + GPU intentado)",
            model_path=model_path, 
            output_size_factor=output_scale_factor
        )
        # Verificamos si ffmpeg_process se inició correctamente
        if thread.ffmpeg_process: 
            camera_threads.append(thread)
        else:
            print(f"La cámara '{source}' no pudo ser iniciada con FFmpeg, no se añadirá al procesamiento.")

    if not camera_threads:
        print("Ninguna cámara se pudo iniciar. Saliendo.")
        return

    for thread in camera_threads:
        thread.start()

    print("\nPresiona 'q' para salir de cualquier ventana.")
    print("Recordatorio: MediaPipe Pose Landmarker en Python aún tiene un problema conocido con el uso efectivo de la GPU en algunas plataformas.")

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
    deteccion_cuatro_camaras_pose_gpu_ffmpeg()