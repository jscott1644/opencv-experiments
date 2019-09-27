from imutils.video import FPS
import numpy as np
import imutils
import dlib
import cv2


# model detection classes. Order of entries matter!
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

CONFIDENCE = 0.50
COLOR = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL = 'car'

# prototex and model definitions
prototex = 'models/MobileNetSSD_deploy.prototxt'
model = 'models/MobileNetSSD_deploy.caffemodel'

# load net
net = cv2.dnn.readNetFromCaffe(prototex, model)

#cap = cv2.VideoCapture('north_parking.mov')
#cap.set(1, 260)

cap = cv2.VideoCapture('videos/gu_intersection.mov')
cap.set(1, 450)

tracker = None
fps = FPS().start()

def drawTrack(frame, label, startX, startY, endX, endY):
    cv2.rectangle(frame, (startX, startY), (endX, endY), COLOR, 2)
    cv2.putText(frame, label, (startX, startY-15), FONT, 0.45, COLOR, 2)

def get_pos(pos):
    startX = int(pos.left())
    startX = int(pos.left())
    startY = int(pos.top())
    endX = int(pos.right())
    endY = int(pos.bottom())
    return startX, startY, endX, endY

while True:
    # read a frame
    _, frame = cap.read()

    if frame is None:
        break

    frame = imutils.resize(frame, width=400)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert colors

    if tracker is None:
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (w,h), 127.5)

        net.setInput(blob)
        detections = net.forward()

        if len(detections) > 0:
            # find best detection
            i = np.argmax(detections[0, 0, :, 2])

            confidence = detections[0, 0, i, 2]
            label = CLASSES[int(detections[0, 0, i, 1])]

            if confidence > CONFIDENCE and label == LABEL:
                # get detection box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype('int')

                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)
                drawTrack(frame, label, startX, startY, endX, endY)
        else:
            # update tracker and get position of tracked object
            tracker.update(rgb)
            pos = tracker.get_position()

            (startX, startY, endX, endY) = get_pos(pos)

            drawTrack(frame, label, startX, startY, endX, endY)
    # show frame
    cv2.imshow('frame', frame)
    fps.update()

    if cv2.waitKey(1) & 0xff == 27:
        break

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cap.release()
cv2.destroyAllWindows()

