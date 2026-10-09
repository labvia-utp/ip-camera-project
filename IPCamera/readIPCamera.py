import cv2

url_rtsp = 'rtsp://admin:321_monzad@192.168.1.110/Streaming/channels/1001' # Reemplaza con tu URL RTSP  rtsp://admin:Admin8350@192.168.31.168:65003/8000

# rtsp://usuario:contraseña@direcciónIP:puerto/ruta

# Para conectar directo a una camara conectada a un router se usa
# --> rtsp://admin:321_monzad@192.168.31.28/8000'  Indicando la ip de la camara y el puerto 8000

# Para conectar a una camara que esta conectada al NVR se usa
# --> rtsp://admin:Admin8350@192.168.31.168/Streaming/channels/201
# --> /Streaming/channels/ Indica que voy a acceder a os canales de streaming del NVR. Esto puede cambiar entre marcas
# --> 201  Indica que se toma el canal 2 o puerto 2 y el substream 1, el 0 es un separador. Substream 2 es menor reslucion
# https://youtu.be/JtKiyWE9xwI?si=2CbAqONWyGWapDpm&t=1326

cap = cv2.VideoCapture(url_rtsp)

if not cap.isOpened():
    print("No se pudo abrir la cámara RTSP")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo recibir el fotograma (fin del flujo?). Saliendo ...")
        break

    cv2.imshow('Camara IP', frame)

    if cv2.waitKey(1) == ord('q'): # Presiona 'q' para salir
        break

cap.release()
cv2.destroyAllWindows()