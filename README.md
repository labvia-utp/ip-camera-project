# Sistema de Detección de Poses Humanas mediante Cámaras IP

## Descripción

Este proyecto implementa un sistema de **detección y estimación de poses humanas en tiempo real** utilizando cámaras IP conectadas a un Network Video Recorder (NVR).

El sistema recibe transmisiones de video mediante RTSP, procesa los fotogramas con OpenCV y utiliza MediaPipe Pose para detectar puntos clave del cuerpo humano (*landmarks*) y visualizar las conexiones del esqueleto sobre el video.

El repositorio contiene diferentes implementaciones y pruebas para la lectura de cámaras IP, el procesamiento de video y la detección de poses humanas con una o múltiples cámaras.

## Tecnologías utilizadas

* **Python 3.11**
* OpenCV: procesamiento y visualización de video.
* MediaPipe Pose: detección y estimación de poses humanas.
* PyAV: recepción y decodificación de transmisiones RTSP mediante FFmpeg.
* FFmpeg: infraestructura de decodificación de video utilizada por PyAV.
* Multithreading: procesamiento independiente de las transmisiones de varias cámaras.

## Estructura del proyecto

Algunos de los principales archivos y directorios son:

| Archivo o directorio                          | Descripción                                                                                                        |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `poseEstimationLegacy4CamerasDict.py`         | Implementación de estimación de poses humanas con múltiples cámaras.                                               |
| `poseEstimation4CamFFMPEG.py`                 | Implementación de procesamiento de cuatro cámaras con FFmpeg.                                                      |
| `ipCameraPoseDetection.py`                    | Detección de poses humanas con una cámara IP.                                                                      |
| `ipCameraPoseDetectionMultipleLegacy4Cams.py` | Variante heredada para cuatro cámaras.                                                                             |
| `readIPCamera.py`                             | Lectura y visualización de una transmisión de cámara IP.                                                           |
| `readIPCameraAV.py`                           | Prueba de lectura de cámara IP utilizando PyAV.                                                                    |
| `readIPCameraAV2.py`                          | Variante de la prueba de lectura mediante PyAV.                                                                    |
| `IPCamera.py`                                 | Script auxiliar relacionado con cámaras IP.                                                                        |
| `DefinicionDeResolucionDeCamara.py`           | Pruebas relacionadas con la resolución de las cámaras.                                                             |
| `listaCamaras.txt`                            | Archivo de configuración para cámaras, sin credenciales ni direcciones IP privadas. 			         			 |
| `listaCamaras2.txt`                           | Archivo de configuración para cámaras, sin credenciales ni direcciones IP privadas. 			                     |
| `requirements.txt`                            | Dependencias de Python del proyecto.                                                                               |
| `2401.15616v1.pdf`                            | Documento de referencia relacionado con el proyecto.                                                               |
| `Kafka/`                                      | Directorio con recursos adicionales del proyecto.                                                                  |

Los archivos que contienen implementaciones similares se conservan para facilitar la comparación, las pruebas y el desarrollo experimental. No todos los scripts tienen que utilizarse para ejecutar la implementación principal.

## Requisitos

* Windows 10/11.
* Python 3.11.x.
* Visual Studio Code, recomendado.
* Acceso a la red local donde se encuentran las cámaras IP o el NVR.
* Credenciales válidas para acceder a las transmisiones RTSP.
* Dependencias indicadas en `requirements.txt`.

Se recomienda utilizar Python 3.11 para mantener la compatibilidad con las versiones de las dependencias utilizadas durante el desarrollo.

## Instalación del entorno

Clonar el repositorio:

Crear un entorno virtual:

```powershell
py -3.11 -m venv .venv
```

Activar el entorno virtual en PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Si PowerShell bloquea la ejecución del script de activación, puede utilizarse temporalmente la siguiente alternativa en la terminal actual:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Instalar las dependencias:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Comprobar la versión de Python:

```powershell
python --version
```

Comprobar las dependencias principales:

```powershell
python -c "import cv2, mediapipe, av; print('OpenCV:', cv2.__version__); print('MediaPipe:', mediapipe.__version__); print('PyAV:', av.__version__)"
```

**Nota:** si una implementación heredada no funciona con las versiones instaladas, deben revisarse las versiones de sus dependencias antes de modificarlas.

## Configuración de las cámaras

Las cámaras se configuran mediante archivos de texto que contienen una dirección RTSP por línea.

Por ejemplo, el formato de una dirección RTSP es:

```text
rtsp://USUARIO:CONTRASENA@IP_DEL_NVR:554/Streaming/channels/CANAL
```

La ruta concreta depende del NVR, de la cámara y del stream que se desee utilizar.

Algunas implementaciones leen las fuentes desde `listaCamaras.txt`. Antes de ejecutar uno de estos scripts, comprueba qué archivo de configuración utiliza y cómo interpreta sus entradas.

### Configuración del laboratorio

La computadora del laboratorio debe conservar su archivo local `listaCamaras.txt`, que contiene las direcciones IP, rutas RTSP y credenciales necesarias para conectarse a las cámaras.

Este archivo debe configurarse localmente en la computadora autorizada para acceder al sistema de cámaras.

**No se deben subir al repositorio público las direcciones RTSP reales, las contraseñas ni otros datos sensibles.**

Los archivos de ejemplo del repositorio deben utilizar valores ficticios o permanecer libres de credenciales reales.

## Ejecución

### Estimación de poses humanas con múltiples cámaras

Para ejecutar una implementación multicámara, primero identifica el script que se utilizará y verifica qué archivo de configuración lee.

Por ejemplo, si se utiliza `poseEstimationLegacy4CamerasDict.py`:

```powershell
python poseEstimationLegacy4CamerasDict.py
```

El nombre del archivo debe coincidir con el que exista en el repositorio y con la implementación que se haya validado.

Si el script utiliza `listaCamaras.txt`, asegúrate de que el archivo esté configurado correctamente antes de iniciarlo.

### Prueba de una cámara IP

Para probar la recepción de video de una sola cámara:

```powershell
python readIPCamera.py
```

Las implementaciones alternativas basadas en PyAV pueden ejecutarse de forma similar, utilizando el nombre del archivo correspondiente.

Para cerrar las ventanas de video, utiliza la tecla `q` cuando el script lo permita.

## Latencia y rendimiento

La latencia y la fluidez dependen de la resolución, el framerate, la calidad de la conexión de red, la configuración del NVR y la carga de procesamiento.

En las pruebas de desarrollo se evaluaron las siguientes estrategias:

* Lectura RTSP mediante PyAV.
* Transporte UDP para reducir la latencia en la red local.
* Procesamiento de cada cámara en un hilo independiente.
* Omisión de fotogramas para reducir la carga de MediaPipe Pose.
* Redimensionamiento de las imágenes para mejorar el rendimiento.
* Suavizado temporal de landmarks para reducir el parpadeo visual.

La omisión de fotogramas permite procesar la pose con menor frecuencia sin dejar de actualizar la imagen de video. Sin embargo, un valor de omisión demasiado alto puede reducir la capacidad de respuesta de la detección.

Los parámetros óptimos dependen de la computadora, las cámaras y las condiciones de la red.

## Seguridad y control de versiones

El repositorio debe contener código fuente, documentación, modelos necesarios y archivos de configuración de ejemplo.

No deben versionarse:

* Credenciales de cámaras, contraseñas o tokens.
* Archivos locales que contengan direcciones RTSP reales.
* Entornos virtuales (`.venv/`).
* Archivos temporales, cachés o archivos de ejecución.
* Configuraciones personales de Visual Studio Code que no sean necesarias para el equipo.

El archivo `requirements.txt` debe mantenerse actualizado con las dependencias necesarias para reproducir el entorno de ejecución.

## Objetivo del proyecto

Desarrollar y evaluar un sistema de visión por computadora capaz de recibir video de varias cámaras IP y estimar poses humanas en tiempo real, con especial atención a la sincronización visual, la latencia y la estabilidad del procesamiento multicámara.

El proyecto está orientado a la experimentación y al desarrollo dentro del Laboratorio de Visión e Inteligencia Artificial (LabVIA).
