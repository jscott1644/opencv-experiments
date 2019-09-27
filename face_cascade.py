import cv2
import imutils
from imutils.video import FPS

face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture('videos/face2.mov')
fps = FPS().start()

while True:
    _, frame = cap.read()
    if frame is None:
        break

    frame = imutils.resize(frame, width=600)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow('frame', frame)
    fps.update()
    if cv2.waitKey(1) & 0xff == 27:
        break

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cap.release()
cv2.destroyAllWindows()