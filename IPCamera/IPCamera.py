import cv2
cap = cv2.VideoCapture('rtsp://admin:321_monzad@192.168.1.110/Streaming/channels/1601')
output_size_factor = 0.6

while True:
    ret, frame = cap.read()

    width = int(frame.shape[1] * output_size_factor)
    height = int(frame.shape[0] * output_size_factor)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    if ret == True:
        cv2.imshow('video output', frame)
        k = cv2.waitKey(10)& 0xff
        if k == 27:
            break
cap.release()
cv2.destroyAllWindows()