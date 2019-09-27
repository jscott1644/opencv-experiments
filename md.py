import cv2
import numpy as np

ALPHA=0.02

LINE_COLOR=(0,255,0)
LINE_THICK=3

THRESH_LVL = 30
CONT_SIZE = 8000

cap = cv2.VideoCapture('videos/gu_intersection.mov')

ret, img = cap.read()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bgr = cv2.GaussianBlur(img, (21,21), 0)

while(ret):
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21,21), 0)

    dframe = cv2.absdiff(bgr, gray)

    _, tframe = cv2.threshold(dframe, THRESH_LVL, 255, cv2.THRESH_BINARY)
    tframe = cv2.dilate(tframe, None, iterations=2)

    cnts, _ = cv2.findContours(tframe.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < CONT_SIZE:
            continue
        
        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w, y+h), LINE_COLOR, LINE_THICK)

    cv2.imshow('frame', frame)
    #cv2.imshow('tframe', tframe)
    #cv2.imshow('dframe', dframe)
    if cv2.waitKey(1) & 0xff == 27:
        break

cap.release()
cv2.destroyAllWindows()