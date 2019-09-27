import numpy as np
import imutils
import pickle
import cv2
from imutils.video import FPS


CONF_THRESH = 0.3
PROP_THRESH = 0.1

# model from opencv_extras
DETECTOR_MODEL = 'models/res10_300x300_ssd_iter_140000.caffemodel'
DETECTOR_PROTO  = 'models/opencv_face_detector.prototxt'
detector = cv2.dnn.readNetFromCaffe(DETECTOR_PROTO, DETECTOR_MODEL)

# model from OpenFace
EMBEDDER_MODEL = 'models/nn4.small2.v1.t7'
embedder = cv2.dnn.readNetFromTorch(EMBEDDER_MODEL)

REC_PATH = 'output/recognizer.pickle'
recognizer = pickle.loads(open(REC_PATH, 'rb').read())

LE_PATH  = 'output/le.pickle'
le = pickle.loads(open(LE_PATH, 'rb').read())

EMB_PATH = 'output/embeddings.pickle'

# load image
cap = cv2.VideoCapture('videos/face2.mov')

fps = FPS().start()

while True:
    #image = cv2.imread(IMG_PATH)
    _, image = cap.read()
    if image is None:
        break

    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # Detect the face candidate
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300,300)),
        1.0,
        (300,300),
        (104., 177., 123.)
    )
    detector.setInput(blob)
    detections = detector.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > CONF_THRESH:
            bbox = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = bbox.astype('int')

            face = image[y1:y2, x1:x2]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue

            # Generate face embedding
            face_blob = cv2.dnn.blobFromImage(face, 1./255, (96,96), (0,0,0), swapRB=True)
            embedder.setInput(face_blob)
            vec = embedder.forward()

            # get the most like prediction, by closeness to embedding
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            if proba > PROP_THRESH:
                text = f"{name}: {proba}"
                y = y1 - 10 if y1 > 20 else y1 + 10
                cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(image, text, (x1,y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 2)
    fps.update()
    cv2.imshow('image', image)
    if cv2.waitKey(1) & 0xff == 27:
        break

fps.stop()
print("Elapsed time: {:.2f}".format(fps.elapsed()))
print("Approx. FPS: {:.2f}".format(fps.fps()))

cap.release()
cv2.destroyAllWindows()