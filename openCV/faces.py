import numpy as np
import cv2
import pickle

# Rectangle Colors
B = 255
G  = 0
R = 0

# Sets up facial cascade
facialCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

# Sets up recognizer for face recognition
recognizer = cv2.face.LBPHFaceRecognizer_create()

# recognizer reads trainner.yml
recognizer.read("trainner.yml")

# Reads Pickle File for Labels
labels = {"person_name": 1}
with open("labels.pkl", 'rb') as f:
    #Loads labels from Pickle File
    og_labels = pickle.load(f)

    #Inverts Label Values
    labels = {v:k for k,v in og_labels.items()}

# Uses integrated webcam for video feed
cap = cv2.VideoCapture(0)

# Makes res 1080p
def res():
    cap.set(3, 500)
    cap.set(4, 500)
res()

while (True):
    # Captures frame by frame
    ret, frame = cap.read()

    # Converts to Gray Scale for Image Processing
    grayScale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find all faces within faces
    faces = facialCascade.detectMultiScale(grayScale, scaleFactor=1.3, minNeighbors=2)

    # Does processes based off of the X Y W H of faces
    for (x, y, w, h) in faces:
        roi_gray = grayScale[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Predicts the face by giving confidence
        id_, conf = recognizer.predict(roi_gray)
        if conf>= 45 and conf <= 100:
            print(str(round(conf)))
            print(labels[id_])

            # Creates label name next to rectangle
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
            cv2.putText(grayScale, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
            B = 0
            R = 255

        # Creates Rectangle
        color = (B, G, R)
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
        cv2.rectangle(grayScale, (x, y), (end_cord_x, end_cord_y), color, stroke)

    # Displays the frames
    cv2.imshow('Video Feed', frame)
    cv2.imshow('Gray Scale Feed', grayScale)

    # If Q is pressed then the code ends
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Ends the video feed
cap.release()
cv2.destroyAllWindows()
