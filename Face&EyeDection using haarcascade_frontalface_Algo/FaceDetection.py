import cv2
import time
import imutils ## image resize

alg="day5/haarcascade_frontalface_default.xml"


face_cascade = cv2.CascadeClassifier(alg) #loading model

eye_cascade = cv2.CascadeClassifier('day5/haar-cascade-files-master/haarcascade_eye.xml')
cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayImg, 1.3, 5) #get coordinates of face

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = grayImg[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow('Facedetecton', img)
    key = cv2.waitKey(1) & 0xFF
    # Exit on key press
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()