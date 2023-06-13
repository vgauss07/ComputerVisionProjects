"""
Facial Recognition Program implementation in python
"""

# Import Libraries
import cv2

# Loading Cascades
"""
Cascades are the series of filters that are applied to the 
face to detect the face and eyes.
"""
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # detect face
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')  # detect eyes on the webcam


def faceDetect(gray, frame):
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roiGray = gray[y:y + h, x:x + w]  # roi = region of interest
        roiColor = frame[y:y + h, x:x + w]
        eyes = eyeCascade.detectMultiScale(roiGray, 1.1, 3)
        for (lx, ly, lw, lh) in eyes:
            cv2.rectangle(roiColor, (lx, ly), (lx + lw, ly + lh), (0, 255, 0), 2)
    return frame  # this shows the regional frame


# Executing the face recognition with the webcam
videoCapture = cv2.VideoCapture(0)
while True:  # repeat infinitely until a break
    _, frame = videoCapture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = faceDetect(gray, frame)
    cv2.imshow('Video', canvas)  # display processed images in an animated way
    if cv2.waitKey(1) & 0xFF == ord('q'):  # To break the facial recognition process
        break

videoCapture.release()
cv2.destroyAllWindows()