import cv2
import os
cam = cv2.VideoCapture(0)
#cam = cv2.VideoCapture('hlop.avi')
cam.set(3, 400)  # set video width
cam.set(4, 300)


face_detector = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
# Вводим id лица которое добавляется в имя и потом будет использовать в распознавание.
#face_id = input('\n enter user id end press  ==>  ')
#print("\n [INFO] Initializing face capture. Look the camera and wait …")
count = 0
while True:
    #ret, img = cam.read()
    ret,img = cam.read()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
            cv2.imwrite('datasetdone/nadya.' + str(count) + '.jpg', small)
    cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff #  'ESC'
    if k == 27:
        break
    elif count >= 83:
        break

cam.release()
cv2.destroyAllWindows()

print("\n [INFO] Exiting Program and cleanup stuff")
exit(0)
