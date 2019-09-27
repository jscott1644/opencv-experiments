# Various CV experiments

videos/ holds test videos

## Motion detection

bg_sub.py - simple backgroup subtraction - KNN or MOG2
md.py - motion detector using dilation, background subtraction, and contours to identify 'objects' (like vae-md)

## Flow estimators

dense_flow.py - dense flow est. Color as direction
point_flow.py - point flow est.

## Trackers

object_trackers.py - multiple object detectors, given an initial bounding box
dnn_track.py - MobileNet SSD tracker
md_tracker.py - complex tracker, using MD to identify potential things to track, then dlib's Mosse pixel correspondance tracker

## Face detection / rec

face_cascade.py - Face detection using Haar
face_dnn.py - Face detection using ___ (Resnet10?)

face_extractor.py - utility to extract all faces (at a interval, e.g. every 30 frames) and save them, for building face templates
face_embedder.py - utility to generate embeddings of all faces
face_train_model.py - train Support Vector Machine using embeddings
face_recognizer.py - detect faces in video, run them through face embedder, compare to SVM for recognition

### File structure

model/ holds models + weights
output/ holds pickle'd embeddings / vectors
faces/ holds folders per-person, and a number of faces per



## Models

GOTURN:
`wget http://cs.stanford.edu/people/davheld/public/GOTURN/trained_model/tracker.caffemodel`

