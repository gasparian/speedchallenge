import cv2 as cv
import numpy as np

cap = cv.VideoCapture("../data/train.mp4")
ret, first_frame = cap.read()
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

while(cap.isOpened()):
    ret, frame = cap.read()
    # cv.imshow("input", frame)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow("grayscale", gray)
    prev_gray = gray
    if cv.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
