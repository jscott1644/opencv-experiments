import json
import torch
import cv2
import numpy as np
from torchvision import transforms
from imutils.video import FPS
from PIL import Image

THRES = 0.3
LINE_COLOR = (0, 255, 0)

model = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b0', pretrained=True)
image_size = (224,224)

labels_map = json.load(open('models/efficentnet_labels.json'))
labels_map = [labels_map[str(i)] for i in range(len(labels_map))]

cap = cv2.VideoCapture('videos/gu_intersection.mov')

# get first image
ret, img = cap.read()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bgr = cv2.GaussianBlur(img, (21,21), 0)

fps = FPS().start()

def make_preds(img):
    '''
    Makes predictions with the model against the image
    '''

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    tfms = transforms.Compose([
        transforms.Resize(image_size), transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    pil_img = tfms(pil_img).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        logits = model(pil_img)

    preds = torch.topk(logits, k=5).indices.squeeze(0).tolist()
    return preds, logits

def draw_preds(img, preds, logits):
    for (i, idx) in enumerate(preds):
        label = labels_map[idx]
        prob = torch.softmax(logits, dim=1)[0, idx].item()
        
        if prob > THRES:
            text = '{:<75} ({:2f}%)'.format(label, prob*100)
            cv2.putText(img, text, (50,50+i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, LINE_COLOR, 2)

def find_motion(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21,21), 0)

    dframe = cv2.absdiff(bgr, gray)

    _, tframe = cv2.threshold(dframe, 40, 255, cv2.THRESH_BINARY)
    tframe = cv2.dilate(tframe, None, iterations=2)

    cnts, _ = cv2.findContours(tframe.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 2000:
            continue
        
        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
        yield (x,y,w,h)


while True:
    _, img = cap.read()
    if img is None:
        break

    potentials = []
    for (x,y,w,h) in find_motion(img):
        w = np.maximum(w,h)
        crop = img[y:y+w, x:x+w]
        
        preds, logits = make_preds(img)
        
        for (i, idx) in enumerate(preds):
            label = labels_map[idx]
            prob = torch.softmax(logits, dim=1)[0, idx].item()
            
            if prob > THRES:
                text = '{} ({:2f}%)'.format(label, prob*100)
                cv2.rectangle(img, (x,y), (x+w, y+w), LINE_COLOR, 2)
                cv2.putText(img, text, (x,y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, LINE_COLOR, 2)

    cv2.imshow('img', img)
    fps.update()
    if cv2.waitKey(1) & 0xff == 27:
        break

fps.stop()
print("Elapsed time: {:.2f}".format(fps.elapsed()))
print("Approx. FPS: {:.2f}".format(fps.fps()))

cap.release()
cv2.destroyAllWindows()