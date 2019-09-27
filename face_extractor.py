import cv2
import numpy as np
from imutils.video import FPS

# model from opencv_extras
DETECTOR_MODEL = 'models/res10_300x300_ssd_iter_140000.caffemodel'
DETECTOR_PROTO  = 'models/opencv_face_detector.prototxt'

FACES_PATH = 'faces'
IMG_SIZE = (300,300)
FRAME_SKIP = 30

CONF_THRES = 0.4

detector = cv2.dnn.readNetFromCaffe(DETECTOR_PROTO, DETECTOR_MODEL)

cap = cv2.VideoCapture('videos/face2.mov')
fps = FPS().start()

counter = 0

print('Start extracting faces from video. ESC to exit')

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

    detector.setInput(blob)
    detections = detector.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONF_THRES:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype('int')
            fW = x2 - x1
            fH = y2 - y1

            x1 = int(np.maximum(0, x1 - fW/2))
            x2 = int(np.minimum(w, x2 + fW/2))
            y1 = int(np.maximum(0, y1 - fH/2))
            y2 = int(np.minimum(w, y2 + fH/2))

            cv2.imwrite(f'{FACES_PATH}/{counter}-{i}.jpg', frame[y1:y2, x1:x2])

    #cv2.imshow('frame', frame)
    print(f'Processing frame {counter}')
    if cv2.waitKey(1) & 0xff == 27:
        break

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cap.release()
cv2.destroyAllWindows()