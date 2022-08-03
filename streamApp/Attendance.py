import math
import time
from sklearn import neighbors
import os
import os.path
import pickle
import face_recognition
import cv2
import numpy as np
import random, string
import datetime
import sqlite3

# Create a db
conn = sqlite3.connect('employee.db')
c = conn.cursor()
# Create a table in the db
c.execute('''CREATE TABLE IF NOT EXISTS attendancee (Date text, emp_name text, attendance text, arrival_time text)''')

sql1 = 'DELETE FROM attendancee'
c.execute(sql1)

conn.commit()
conn.close()


def predict(img, knn_clf=None, model_path=None, distance_threshold=0.4):
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    # X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test image
    faces_encodings = face_recognition.face_encodings(img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
            zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


print("\n Looking for faces via webcam...")

video_capture = cv2.VideoCapture(0)
# time.sleep(2.0)
name_exists = 'unjhuhhb'  # random name string

while True:
    # Using the trained classifier, make predictions for unknown images
    ret, frame = video_capture.read()
    predictions = predict(frame, model_path="trained_model.clf")

    # Print results on the console
    for name, (top, right, bottom, left) in predictions:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # if match occurs save employee image and write in database
        if name != "unknown" and name != name_exists:
            face_image = frame[top:bottom, left:right]
            # path where employees' detected faces are stored
            path = " "
            cv2.imwrite(os.path.join(path, name + '.jpg'), face_image)
            print("- Found {} at ({}, {})".format(name, left, top))

            k = datetime.datetime.now()

            tim = str(k.hour) + ':' + str(k.minute) + ':' + str(k.second)
            dat = str(k.day) + '-' + str(k.month) + '-' + str(k.year)
            ff = 'Present'
            # Connect to the database
            conn = sqlite3.connect('employee.db')
            c = conn.cursor()
            # Insert a row of data
            c.execute("INSERT INTO attendancee VALUES (?,?,?,?)", (dat, name, ff, tim))
            conn.commit()
            conn.close()
            print('Attendance marked')

            name_exists = name

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()