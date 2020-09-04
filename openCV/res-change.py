import numpy as np
import cv2

# Selects Integrated Web Cam for Video Feed
cap = cv2.VideoCapture(0)

# Functions for changing the resolution of the video feed
def make_1080p():
    cap.set(3, 1920)
    cap.set(4, 1080)

def make_720():
    cap.set(3, 1280)
    cap.set(4, 720)

def make_480p():
    cap.set(3, 640)
    cap.set(4, 480)

def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)

#Changes Resolution
make_1080p()

#The Frame by Frame Processing
while(True):
    # Captures frame by frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('Video Feed', frame)

    # Breaks code when q is pressed
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()