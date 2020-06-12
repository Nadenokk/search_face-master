import cv2
import os
import re

path = 'dataset2/'

fds = sorted(os.listdir(path))
face_detector = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
count = 0
for img1 in fds:

    print(str(img1))
    img = cv2.imread(path+img1)
    rgb = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
    faces = face_detector.detectMultiScale(rgb, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        # Сохраняем лицо
        roi_gray = rgb[y:y + h, x:x + w]
        f = img[y:y+h,x:x+w]
        small = cv2.resize(f, (150, 150))
        eyes = eyeCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.2,
            minNeighbors=4,
            minSize=(10, 10),
        )
        if len(eyes) > 0:
            count += 1
            cv2.imwrite('datasetdone/unknown.' + str(count) + '.jpg', small)

