import numpy as np
import cv2

# Captures Video from Integrated Webcam
cap = cv2.VideoCapture(0)

while True:
    # Reading frame by frame what is happenign
    ret, frame = cap.read()

    # Converts video to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Shows the frame captured
    cv2.imshow('Color Frame', frame)

    # Shows the gray captured film
    cv2.imshow('Gray Frame', gray)

    # Breaks when Q has been held for 20 ms
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
