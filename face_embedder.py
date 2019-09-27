import os
import pickle
import numpy as np
import imutils
import cv2
from imutils import paths

# model from OpenCV_extra models
DETECTOR_MODEL = 'models/res10_300x300_ssd_iter_140000.caffemodel'
DETECTOR_PROTO  = 'models/opencv_face_detector.prototxt'
detector = cv2.dnn.readNetFromCaffe(DETECTOR_PROTO, DETECTOR_MODEL)

# model from OpenFace project
EMBEDDER_MODEL = 'models/nn4.small2.v1.t7'
embedder = cv2.dnn.readNetFromTorch(EMBEDDER_MODEL)

EMB_PATH = 'output/embeddings.pickle'
FACES_PATH = 'faces'

CONF_THRESH = 0.5
MIN_FACE = 20
DEBUG = False

def init_faces():
    print('Quantifying faces...')
    imagePaths = list(paths.list_images(FACES_PATH))

    known_embeddings = []
    known_names = []

    total = 0

    for (i, imagePath) in enumerate(imagePaths):
        print(f'Processing image {i+1}/{len(imagePaths)}')

        # each person should have their own folder, with reference image(s) inside
        name = imagePath.split(os.path.sep)[-2]

        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=600)

        (h,w) = image.shape[:2]

        # get detections
        rz = cv2.resize(image, (300,300))
        blob = cv2.dnn.blobFromImage(rz, 1., (300, 300), (104., 177., 123.))
        detector.setInput(blob)
        detections = detector.forward()

        if len(detections) > 0:
            # find detection w/ largest prop
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]

            if confidence > CONF_THRESH:
                # get bounding box, params 3-7 of last column
                bbox = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = bbox.astype('int')

                if DEBUG:
                    text = '{:.2f}'.format(confidence)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(image, text, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0),2)
                    cv2.imshow(f'{name}', image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                # extract face ROI
                face = image[y1:y2, x1:x2]
                (fH, fW) = face.shape[:2]

                # filter faces below size
                if fW < MIN_FACE or fH < MIN_FACE:
                    continue
                faceblob = cv2.dnn.blobFromImage(face, 1./255, (96, 96), (0,0,0), swapRB=True, crop=False)
                embedder.setInput(faceblob)
                vec = embedder.forward()

                known_names.append(name)
                known_embeddings.append(vec.flatten())
                total += 1 # TODO: total should be the length of known_x, no?

    data = {'embeddings': known_embeddings, 'names': known_names}
    # TODO: store it
    with open(EMB_PATH, 'wb') as f:
        print(f'Serializing {len(known_names)} encodings...')
        f.write(pickle.dumps(data))

    return data

if __name__ == '__main__':
    init_faces()