import cv2
import numpy as np
from imutils.video import FPS

PROTO = 'models/faces.prototxt'
MODEL = 'models/faces.caffemodel'

IMG_SIZE = (300,300)
FRAME_SKIP = 3

CONF_THRES = 0.4

net = cv2.dnn.readNetFromCaffe(PROTO, MODEL)

cap = cv2.VideoCapture('videos/face2.mov')
fps = FPS().start()

counter = 0

while True:
    _, frame = cap.read()
    counter += 1
    fps.update()

    if frame is None:
        break
    if counter % FRAME_SKIP != 0:
        continue

    (h, w) = frame.shape[:2]
    rz = cv2.resize(frame, IMG_SIZE)
    blob = cv2.dnn.blobFromImage(rz, 1.0, IMG_SIZE, (104., 177., 123.))

    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONF_THRES:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')

            text = "{:2f}".format(confidence)
            y = startY - 10 if startY - 10 > 10 else startY + 10

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0,255,0), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xff == 27:
        break

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cap.release()
cv2.destroyAllWindows()