import cv2
import numpy as np


# inicializamos nuestra camara
camera = cv2.VideoCapture('rtsp://admin:321_monzad@192.168.31.28/8000')#cv2.VideoCapture(0)
# definir ancho y alto de la camara
camera.set(3, 640) 
camera.set(4, 480)
camera.set(cv2.CAP_PROP_FPS, 1)


while True:
    _, frame = camera.read()
    
    cv2.imshow("Imagen de Camara", frame)
                
    # cerrar el programar pulsando la tecla p
    if cv2.waitKey(1) == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()