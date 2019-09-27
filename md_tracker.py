'''
Background-sub + Mosse tracking
'''

import cv2
import numpy as np
import dlib

ALPHA=0.02

LINE_COLOR=(0,255,0)
LINE_THICK=2
FONT = cv2.FONT_HERSHEY_SIMPLEX

TRACK_THRESH = .2
NMS_THRESH = .2
THRESH_LVL = 30
CONT_SIZE = 1000

#cap = cv2.VideoCapture('videos/north_parking.mov')
cap = cv2.VideoCapture('videos/traffic-mini.mp4')

ret, img = cap.read()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bgr = cv2.GaussianBlur(img, (21,21), 0)


trackers = []

def get_pos(pos):
    startX = int(pos.left())
    startX = int(pos.left())
    startY = int(pos.top())
    endX = int(pos.right())
    endY = int(pos.bottom())

    return startX, startY, endX, endY

def drawTrack(frame, label, startX, startY, endX, endY):
    cv2.rectangle(frame, (startX, startY), (endX, endY), LINE_COLOR, LINE_THICK)
    cv2.putText(frame, label, (startX, startY-15), FONT, 0.45, LINE_COLOR, LINE_THICK)

def str_box(box):
    return f"({box[0]},{box[1]}),({box[2]},{box[3]})"

while(ret):
    if len(trackers) > 10: break

    detections = []
    scores = []
    old_tracks = []

    ret,frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # run dlib trackers
    if len(trackers) > 0:
        for t in trackers:
            # score for track, higher=more confident
            # score is 'peak to side-lobe' ratio
            # option second param is 'guess', the region to look in
            score = t.update(rgb)
            score = np.minimum(score, 25.) / 25.
            
            i = trackers.index(t)

            # check if the score is high enough
            if score > TRACK_THRESH:
                pos = t.get_position()
                (startX, startY, endX, endY) = get_pos(pos)
                box = (startX, startY, endX, endY)

                # save the detection
                detections.append(box)
                scores.append(score)
                #print(f' -- t{i}: {str_box(box)} - {score}')
            else:
                # mark traker for removal if score is too low (lost track)
                old_tracks.append(t)

    ## MD section
    # gray it, blur it
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21,21), 0)

    # background subtract
    dframe = cv2.absdiff(bgr, gray)

    # threshold and dilate
    _, tframe = cv2.threshold(dframe, THRESH_LVL, 255, cv2.THRESH_BINARY)
    tframe = cv2.dilate(tframe, None, iterations=2)

    # find contours (continous shapes)
    cnts, _ = cv2.findContours(tframe.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    i=0
    for contour in cnts:
        # find contour area, check it is big enough
        if cv2.contourArea(contour) < CONT_SIZE:
            continue
        
        # find a box that surrounds the contour
        (x,y,w,h) = cv2.boundingRect(contour)

        # save this box for display
        box = (x, y, x+w, y+h)
        detections.append(box)
        scores.append(.15)
        #print(f' -- m{i}: {str_box(box)} - 0.15')
        i += 1

    # eliminate overlapping boxes
    idxs = cv2.dnn.NMSBoxes(detections, scores, .1, NMS_THRESH)

    for idx in idxs:
        i = idx[0]
        box = detections[i]
        score = scores[i]
        #print(f' --- {i}: {str_box(box)} - {score}')

        if score < .2: # this track is from MD, not tracker
            # start a dlib tracker for this box
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(box[0], box[1], box[2], box[3])
            tracker.start_track(rgb, rect)
            #print(f"adding tracker {str_box(box)}")
            trackers.append(tracker)

        # display non-overlapping boxes
        str_score = "{:.2f}".format(score)
        drawTrack(frame, str_score, box[0], box[1], box[2], box[3])

    # remove trackers
    for t in old_tracks:
        #print("removing tracker")
        trackers.remove(t)

    cv2.imshow('frame', frame)
    #cv2.imshow('tframe', tframe)
    #cv2.imshow('dframe', dframe)
    if cv2.waitKey(1) & 0xff == 27:
        break

cap.release()
cv2.destroyAllWindows()