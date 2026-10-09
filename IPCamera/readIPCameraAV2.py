import av
import cv2

url = "rtsp://admin:321_monzad@192.168.1.110:554/Streaming/Channels/1001"
opts = {"rtsp_transport": "udp", "stimeout": "5000000"}

container = av.open(url, options=opts)

for frame in container.decode(video=0):
    img = frame.to_ndarray(format="bgr24")
    cv2.imshow("PyAV", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

container.close()
cv2.destroyAllWindows()