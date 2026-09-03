import os
import cv2
import numpy as np
from ultralytics import YOLO
from IPython.display import display
from PIL import Image

cap = cv2.VideoCapture(0)

names = {0: "Afrin", 1: "Anisha"}
max_distace = 70

model = YOLO("yolov8m-face.pt")
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_recognizer.yml")

while cv2.waitKey(1) != ord("x") :
    _, frame = cap.read()
    face_result = model(frame, verbose=False)
    processed_feed = face_result[0].plot()

    for box in face_result[0].boxes.xyxy :
        left, top, right, bottom = box.int()
        face = frame[top:bottom, left:right]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (200, 200))

        predicted_label, distance_from_org = face_recognizer.predict(face)

        if distance_from_org > max_distace :
            name = "Unknown"
        else:
            name = names[predicted_label]

        cv2.putText(
            processed_feed,
            name + " | " + str(int(distance_from_org)),
            (int(left), int(bottom) + 30),
            0,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.imshow("CV Toolkit", processed_feed)

cv2.waitKey(5000)
cap.release()



#Code for training Model
# folder = "faces/Anisha"
# folders = {0: "faces/Afrin", 1: "faces/Anisha"}
# faces = []
# labels = []

# for label, folder in folders.items():
#     files = os.listdir(folder)

#     for file in files:
#         photo = cv2.imread(folder + "/" + file)
#         face_result = model(photo, verbose = False)
#         processedImg = face_result[0].plot()

#         left, top, right, bottom = face_result[0].boxes.xyxy[0].int()
#         face = photo[top:bottom, left:right]
#         face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
#         face = cv2.resize(face, (200, 200))

#         faces.append(face)
#         labels.append(label)

#         # cv2.imshow("Image", faces[0])
#         # cv2.waitKey(0)
# face_recognizer.train(faces, np.array(labels))
# face_recognizer.write("face_recognizer.yml")

# cv2.destroyAllWindows()