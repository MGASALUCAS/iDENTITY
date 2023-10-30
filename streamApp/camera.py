# # importing libraries and packages.
# import datetime
# import os
# import os.path
# import pickle
# import sqlite3
# import pyttsx3
# import time

# import cv2
# import face_recognition
# from flask import Flask, render_template, Response

# # Create a database.
# conn = sqlite3.connect('registered.db')
# c = conn.cursor()
# # Create a table in the db
# c.execute(
#     '''CREATE TABLE IF NOT EXISTS attendancee (Date text, student_name text, attendance text, arrival_time text)''')

# sql1 = 'DELETE FROM attendancee'
# c.execute(sql1)

# conn.commit()
# conn.close()


# def predict(img, knn_clf=None, model_path=None, distance_threshold=0.4):
#     if knn_clf is None and model_path is None:
#         raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

#     # Load a trained KNN model (if one was passed in)
#     if knn_clf is None:
#         with open(model_path, 'rb') as f:
#             knn_clf = pickle.load(f)

#     # Load image file and find face locations
#     # X_img = face_recognition.load_image_file(X_img_path)
#     X_face_locations = face_recognition.face_locations(img)

#     # If no faces are found in the image, return an empty result.
#     if len(X_face_locations) == 0:
#         return []

#     # Find encodings for faces in the test image
#     faces_encodings = face_recognition.face_encodings(img, known_face_locations=X_face_locations)

#     # Use the KNN model to find the best matches for the test face
#     closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
#     are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

#     # Predict classes and remove classifications that aren't within the threshold
#     return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
#             zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


# print("\n Looking for faces via webcam...")

# # video_capture = cv2.VideoCapture("https://192.168.43.107:8080/video")
# video_capture = cv2.VideoCapture(0)



# # video_capture = cv2.VideoCapture(0)


# time.sleep(2.0)


# def gen_frames():
#     while True:
#         # Using the trained classifier, make predictions for unknown images
#         ret, frame = video_capture.read()
#         predictions = predict(frame, model_path="trained_model.clf")

#         name_exists = 'unjhuhhb'  # random name string
#         # Print results on the console
#         for name, (top, right, bottom, left) in predictions:
#             cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

#             # Draw a label with a name below the face
#             cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
#             font = cv2.FONT_HERSHEY_DUPLEX
#             cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

#             # if match occurs save employee image and write in database
#             if name != name_exists:
#                 face_image = frame[top:bottom, left:right]
#                 # path where employees' detected faces are stored
#                 path = " "
#                 cv2.imwrite(os.path.join(path, name + '.jpg'), face_image)
#                 print("- Found {} at ({}, {})".format(name, left, top))

#                 k = datetime.datetime.now()

#                 tim = str(k.hour) + ':' + str(k.minute) + ':' + str(k.second)
#                 dat = str(k.day) + '-' + str(k.month) + '-' + str(k.year)
#                 ff = 'Present'

#                 # Connect to the database
#                 conn = sqlite3.connect('registered.db')
#                 c = conn.cursor()

#                 # Insert a row of data
#                 c.execute("INSERT INTO attendancee VALUES (?,?,?,?)", (dat, name, ff, tim))
#                 conn.commit()
#                 conn.close()
#                 print('Attendance marked')

#                 name_exists = name
#                 # markAttendance(name)

#                 # Voice Analysis part.
#                 engine = pyttsx3.init()
#                 voices = engine.getProperty('voices')
#                 engine.setProperty('voice', voices[1].id)
#                 engine.setProperty('rate', 150)

#                 # engine.say("A person is recognized.")
#                 engine.runAndWait()

#             elif name != "unknown":
#                 # Voice Analysis part.
#                 engine = pyttsx3.init()
#                 voices = engine.getProperty('voices')
#                 engine.setProperty('voice', voices[1].id)
#                 engine.setProperty('rate', 150)
#                 engine.say("Unknown person has recognized")
#                 engine.runAndWait()

#         cv2.imshow('Video', frame)

#         # if cv2.waitKey(1) & 0xFF == ord('q'):
#         #     break

#         # # Release handle to the webcam
#         # video_capture.release()
#         # cv2.destroyAllWindows()

#         ret, buffer = cv2.imencode('.jpg', frame)
#         img = buffer.tobytes()
#         yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')  # concat frame one by one and
#         # show result


# app = Flask(__name__)


# @app.route('/video_feed')
# def video_feed():
#     # Video streaming route. Put this in the src attribute of an img tag
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# @app.route('/')
# def index():
#     # """Video streaming home page."""
#     return render_template('home.html')


# if __name__ == '__main__':
#     app.run(debug=True)







# # importing libraries and packages.
# import datetime
# import os
# import os.path
# import pickle
# import sqlite3
# import time

# import cv2
# import face_recognition
# from flask import Flask, render_template, Response


# def predict(img, knn_clf=None, model_path=None, distance_threshold=0.4):
#     if knn_clf is None and model_path is None:
#         raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

#     # Load a trained KNN model (if one was passed in)
#     if knn_clf is None:
#         with open(model_path, 'rb') as f:
#             knn_clf = pickle.load(f)

#     # Load image file and find face locations
#     # X_img = face_recognition.load_image_file(X_img_path)
#     X_face_locations = face_recognition.face_locations(img)

#     # If no faces are found in the image, return an empty result.
#     if len(X_face_locations) == 0:
#         return []

#     # Find encodings for faces in the test image
#     faces_encodings = face_recognition.face_encodings(img, known_face_locations=X_face_locations)

#     # Use the KNN model to find the best matches for the test face
#     closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
#     are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

#     # Predict classes and remove classifications that aren't within the threshold
#     return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
#             zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


# print("\n Looking for faces via webcam...")

# video_capture = None
# def init_video_capture():
#     global video_capture
#     video_capture = cv2.VideoCapture("https://192.168.43.91:8080/video")
#     # video_capture = cv2.VideoCapture(0)

# knn_clf = None
# def init_knn_clf():
#     global knn_clf
#     with open("trained_model.clf", 'rb') as f:
#         knn_clf = pickle.load(f)


# def gen_frames():
#     if video_capture is None:
#         init_video_capture()
#     if knn_clf is None:
#         init_knn_clf()

#     name_exists = 'unjhuhhb'  # random name string

#     while True:
#         ret, frame = video_capture.read()
#         predictions = predict(frame, knn_clf=knn_clf)

#         # Print results on the console
#         for name, (top, right, bottom, left) in predictions:
#             cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

#             # Draw a label with a name below the face
#             cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
#             font = cv2.FONT_HERSHEY_DUPLEX
#             cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

#             # if match occurs save employee image and write in database
#             if name != name_exists:
#                 face_image = frame[top:bottom, left:right]
#                 # path where employees' detected faces are stored
#                 path = " "
#                 cv2.imwrite(os.path.join(path, name + '.jpg'), face_image)
#                 print("- Found {} at ({}, {})".format(name, left, top))

#                 k = datetime.datetime.now()

#                 tim = str(k.hour) + ':' + str(k.minute) + ':' + str(k.second)
#                 dat = str(k.day) + '-' + str(k.month) + '-' + str(k.year)
#                 ff = 'Present'

#                 # Connect to the database
#                 conn = sqlite3.connect('registered.db')
#                 c = conn.cursor()

#                 # Insert a row of data
#                 c.execute("INSERT INTO attendancee VALUES (?,?,?,?)", (dat, name, ff, tim))
#                 conn.commit()
#                 conn.close()
#                 print('Attendance marked')

#                 name_exists = name
#                 # markAttendance(name)

#                 # Voice Analysis part.
#                 engine = pyttsx3.init()
#                 voices = engine.getProperty('voices')
#                 engine.setProperty('voice', voices[1].id)
#                 engine.setProperty('rate', 150)

#                 # engine.say("A person is recognized.")
#                 engine.runAndWait()

#             elif name != "unknown":
#                 # Voice Analysis part.
#                 engine = pyttsx3.init()
#                 voices = engine.getProperty('voices')
#                 engine.setProperty('voice', voices[1].id)
#                 engine.setProperty('rate', 150)
#                 engine.say("Unknown person has recognized")
#                 engine.runAndWait()

#         # cv2.imshow('Video', frame)

#         # if cv2.waitKey(1) & 0xFF == ord('q'):
#         #     break

#         # # Release handle to the webcam
#         # video_capture.release()
#         # cv2.destroyAllWindows()

#         ret, buffer = cv2.imencode('.jpg', frame)
#         img = buffer.tobytes()
#         yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')  # concat frame one by one and
#         # show result


# app = Flask(__name__)


# @app.route('/video_feed')
# def video_feed():
#     # Video streaming route. Put this in the src attribute of an img tag
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# @app.route('/')
# def index():
#     # """Video streaming home page."""
#     return render_template('home.html')


# if __name__ == '__main__':
#     app.run(debug=True)

















# importing libraries and packages.
import datetime
import os
import os.path
import pickle
import sqlite3
import time
import pyttsx3

import cv2
import face_recognition
from flask import Flask, render_template, Response

# Import from external Loading.
from streamApp.prediction import predict, knn_clf, name_exists
from streamApp.videocapturing import capture_video
from streamApp.database import *


print("\n Looking for faces via webcam...")



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def gen_frames():
    while True:
        cap = cv2.VideoCapture("rtsp://admin:admin123@192.168.1.122/cam/realmonitor?channel=1&subtype=0")
        # cap = cv2.VideoCapture('http://10.185.1.121:8080/video')
        # cap = cv2.VideoCapture('http://192.168.43.1:8080/video')

        # cap = cv2.VideoCapture(0)

        ret, frame = cap.read()

        # Using the trained classifier, make predictions for unknown images.
        # ret, frame = capture_video()

        predictions = predict(frame, knn_clf=knn_clf)

        # Print results on the console
        for name, (top, right, bottom, left) in predictions:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # if match occurs save employee image and write in database
            if name == name_exists:
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
                conn = sqlite3.connect('registered.db')
                c = conn.cursor()

                # Insert a row of data
                c.execute("INSERT INTO attendancee VALUES (?,?,?,?)", (dat, name, ff, tim))
                conn.commit()
                conn.close()
                print('Attendance marked')


                # name_exists = name
                # markAttendance(name)

                # Voice Analysis part.
                engine = pyttsx3.init()
                voices = engine.getProperty('voices')
                engine.setProperty('voice', voices[1].id)
                engine.setProperty('rate', 150)

                # engine.say("A person is recognized.")
                engine.runAndWait()

            elif name == "unknown":
                # Capturing the unknown person image.
                face_image = frame[top:bottom, left:right]
                print("unknown person has recognizd and image saved.")
                unknown_count = 0
                unknown_count += 1
                path = "D:/Projects/docker works/identity/iDENTITY/unknown_faces"
                unknown_filename = f"unknown_{unknown_count}.jpg"
                cv2.imwrite(os.path.join(path, unknown_filename), face_image)

                # Voice Analysis part.
                engine = pyttsx3.init()
                voices = engine.getProperty('voices')
                engine.setProperty('voice', voices[1].id)
                engine.setProperty('rate', 150)
                engine.say("Unknown person has recognized")
                engine.runAndWait()

            elif name == "wanted":
                # Capturing the unknown person image.
                face_image = frame[top:bottom, left:right]
                print("Wanted person has recognizd and image saved.")
                unknown_count = 0
                unknown_count += 1
                path = "D:/Projects/docker works/identity/iDENTITY/wanted_faces"
                unknown_filename = f"unknown_{unknown_count}.jpg"
                cv2.imwrite(os.path.join(path, unknown_filename), face_image)

                # Voice Analysis part.
                engine = pyttsx3.init()
                voices = engine.getProperty('voices')
                engine.setProperty('voice', voices[1].id)
                engine.setProperty('rate', 150)
                time.delay(60)
                engine.say("WANTED! WANTED! WANTED! WANTED! person has recognized")
                engine.runAndWait()

        # cv2.imshow('Video', frame)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        # # Release handle to the webcam
        # video_capture.release()
        # cv2.destroyAllWindows()

        ret, buffer = cv2.imencode('.jpg', frame)
        img = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')  # concat frame one by one and
        # show result
    return Response(mimetype='multipart/x-mixed-replace; boundary=frame')


app = Flask(__name__)


# @app.route('/video_feed')
# def video_feed():
#     # Video streaming route. Put this in the src attribute of an img tag
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/')
def index():
    # """Video streaming home page."""
    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)





# import cv2

# # Open the camera capture device
# cap = cv2.VideoCapture('http://192.168.1.108')

# # Get the default frame size
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# print("Default frame size:", frame_width, "x", frame_height)

# # Release the camera capture device
# cap.release()


# ****************************************************

# import cv2

# # Set the desired frame size
# frame_width = 0
# frame_height = 0

# # Open the camera capture device and set the frame size
# cap = cv2.VideoCapture("http://admin:mgasa1234!.@192.168.1.108")
# # cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# print('Succesfully Read!')
# while True:
#     ret, frame = cap.read()

#     if ret:  # Check if frame is valid
#         cv2.imshow('frame', frame)
#     else:
#         print('Error reading frame')
#         break

#     if cv2.waitKey(1) == ord('q'):
#         break
# print('failed to open direct!')
# cap.release()
# cv2.destroyAllWindows()


# Uisng password and user name for dahua test one:


# import cv2

# cap = cv2.VideoCapture("rtsp://admin:mgasa1234!.@192.168.1.108/cam/realmonitor?channel=1&subtype=0")

# while True:
#     ret, frame = cap.read()

#     if ret:  # Check if frame is valid
#         print('Done reading!')
#         if frame.size != (0, 0):  # Check if frame is not empty
#             print('Looking for frames!')
#             cv2.imshow('frame', frame)
#     else:
#         print('Error reading frame')
#         break

#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
