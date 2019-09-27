import numpy as np
import cv2

cap = cv2.VideoCapture('videos/traffic-mini.mp4')

feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

lk_params = dict(winSize=(15,15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                
color = np.random.randint(0,255,(100,3))

ret, frame1 = cap.read()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)

mask = np.zeros_like(frame1)

while(ret):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray, p0, None, **lk_params)

    good_new = p1[st==1]
    good_old = p0[st==1]

    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b), (c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a,b), 5, color[i].tolist(), -1)
    
    img = cv2.add(frame,mask)

    cv2.imshow('frame', img) 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    gray1 = gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cap.release()
cv2.destroyAllWindows()