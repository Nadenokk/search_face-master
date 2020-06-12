import cv2
import numpy as np
import tensorflow as tf

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

model = tf.keras.models.load_model('my.h5')


def face_detect(img):

    img = tf.keras.preprocessing.image.img_to_array(img)
    img_arr = np.expand_dims(img, axis=0)/255.0
    result = model.predict(img_arr)
    b = result.argmax(axis=1)[0]
    t = result[0][b] * 100
    r = ("Test: %.2f" % t).encode()
    return t

cascadePath = "cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
# Тип шрифта
font = cv2.FONT_HERSHEY_SIMPLEX


cam = cv2.VideoCapture(0)
cam.set(3, 400)  # set video width
cam.set(4, 300)  # set video height
cap = cv2.VideoCapture('hlop.avi')
#cap.set(3, 400)  # set video width
#cap.set(4, 300)
while True:
    ret, img = cam.read()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret1, img1 = cap.read()
    rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        rgb,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(10, 10),
    )

    faces1 = faceCascade.detectMultiScale(
        rgb1,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(10, 10),
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        f = img[y:y + h, x:x + w]
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        small = cv2.resize(f, (150, 150))
        t = face_detect(small)
        cv2.putText(img, "%.5f" % t + '%', (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
        if (t<50):
            cv2.putText(img, "Unknown", (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        else:
            cv2.putText(img, "Nadya", (x + 5, y - 5), font, 1, (255, 255, 255), 2)
    for (x, y, w, h) in faces1:
        cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        f1 = img1[y:y + h, x:x + w]
        f1 = cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)
        small1 = cv2.resize(f1, (150, 150))
        t = face_detect(small1)
        cv2.putText(img1, "%.5f" % t + '%', (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
        if (t<50):
            cv2.putText(img1, "Unknown", (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        else:
            cv2.putText(img1, "Nadya", (x + 5, y - 5), font, 1, (255, 255, 255), 2)

    cv2.imshow('camera', img)
    cv2.namedWindow('video', cv2.WINDOW_NORMAL)
    cv2.imshow('video', img1)

    k = cv2.waitKey(10) & 0xff  # 'ESC' для Выхода
    if k == 27:
        break

cam.release()
cap.release()
cv2.destroyAllWindows()