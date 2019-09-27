import cv2
import imutils
from imutils.video import FPS, VideoStream

cap = cv2.VideoCapture('videos/north_parking.mov')
cap.set(1, 260)
ok, frame = cap.read()
frame = imutils.resize(frame, width=600)
(H, W) = frame.shape[:2]


OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create,
    "goturn": cv2.TrackerGOTURN_create
}

TRACKER = 'csrt'

tracker = OPENCV_OBJECT_TRACKERS[TRACKER]()

bbox = cv2.selectROI(frame, False)
fps = FPS().start()

tracker.init(frame, bbox)

while True:
    ok, frame = cap.read()
    frame = imutils.resize(frame, width=600)

    ok, box = tracker.update(frame)

    if ok:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      (0, 255, 0), 2)

    fps.update()
    fps.stop()
    
    info = [
        ("Tracker", TRACKER),
        ("Success", "Yes" if ok else "No"),
        ("FPS", "{:.2f}".format(fps.fps())),
    ]

    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow('tracking', frame)

    if cv2.waitKey(1) & 0xff == 27:
        break

cap.release()
cv2.destroyAllWindows()
