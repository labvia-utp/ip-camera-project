import av, cv2, time

url = "rtsp://admin:321_monzad@192.168.1.110:554/Streaming/Channels/1001"
opts = {"rtsp_transport": "tcp", "stimeout": "5000000"}

container = av.open(url, options=opts)
stream = container.streams.video[0]

t0 = time.time()
for frame in container.decode(stream):
    img = frame.to_ndarray(format="bgr24")
    cv2.imshow("PyAV", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

container.close()
cv2.destroyAllWindows()
