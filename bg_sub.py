import cv2
import imutils
from imutils.video import FPS

cap = cv2.VideoCapture('videos/gu_intersection.mov')
ret,_ = cap.read()

MIN_SIZE = 500

#backSub = cv2.createBackgroundSubtractorKNN()
backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=80)

fps = FPS().start()


def refine_segments(img, mask, iters=3):
    
    mask = cv2.dilate(mask, None, iterations=iters)
    mask = cv2.erode(mask, None, iterations=iters*2)
    mask = cv2.dilate(mask, None, iterations=iters)
    #_, mask = cv2.threshold(mask, 60, 255, cv2.THRESH_BINARY)

    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in cnts:
        if cv2.contourArea(contour) < MIN_SIZE:
            continue
        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
    #return cv2.drawContours(img, cnts, -1, (0,255,0), thickness=2)
    return img


while ret:
    ret,frame = cap.read()
    frame = imutils.resize(frame, width=600)

    fgMask = backSub.apply(frame)

    frame = refine_segments(frame, fgMask)

    fps.update()
    cv2.imshow('frame', frame)
    cv2.imshow('mask', fgMask)
    if cv2.waitKey(1) & 0xff == 27:
        break

fps.stop()
print(f'elapsed time {fps.elapsed()}')
print(f'fps {fps.fps()}')
cap.release()
cv2.destroyAllWindows()
