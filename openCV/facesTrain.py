import os
from PIL import Image
import numpy as np
import cv2
import pickle

# Sets up facial cascade
facialCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

# Sets up recognizer for face recognition
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Scans directories for images folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

# Arrays for labels and image paths
y_labels = []
x_train = []

# Dictionary for ids
current_id = 0
label_ids = {}

# Check the images within the images folder for PNG and JPG files
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            # Finds individual paths of images
            path = os.path.join(root, file)

            # Sets the labels to work with the directory names ie the names
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()

            # Creates new label ids for new people
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1

            # Sets the image label to id_
            id_ = label_ids[label]

            # Converts to Gray Scale
            pil_image = Image.open(path).convert("L")

            # Resize images before saving
            size = (500, 500)
            final_image = pil_image.resize(size, Image.ANTIALIAS)

            # Converts Gray Scale to Numbers
            image_array = np.array(final_image, "uint8")

            # Find all faces within images
            faces = facialCascade.detectMultiScale(image_array, scaleFactor=1.2, minNeighbors=5)

            for (x, y, w, h) in faces:
                # Sets region of interest(ROI)
                roi = image_array[y:y + h, x:x + w]

                # Appends the training list for ROI
                x_train.append(roi)
                y_labels.append(id_)

# Dumps all ids into the pickle id
with open("labels.pkl", 'wb') as f:
    pickle.dump(label_ids, f)

# Trains the recognizer to know each person for each label
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")

# Notifies when done
print("All People Were Updated")
