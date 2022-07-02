import os
import pandas as pd
import cv2
import pyttsx3
import datetime
import time
import sys

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voices', voices[0].id)


# text to speech
def speak(audio):
    engine.say(audio)
    engine.runAndWait()


# face recognition

recognizer = cv2.face.LBPHFaceRecognizer_create()  # Local Binary Patterns Histograms
recognizer.read('trainer/trainer.yml')  # load trained model
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)  # initializing haar cascade for object detection approach

font = cv2.FONT_HERSHEY_SIMPLEX  # denotes the font type

df = pd.read_csv("StudentDetails" + os.sep + "Details.csv")
col_names = ['ID', 'NAME', 'DATE', 'TIMESTAMP']

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # cv2.CAP_DSHOW to remove warning
cam.set(3, 640)  # set video FrameWidht
cam.set(4, 480)  # set video FrameHeight

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

# flag = True

g_name = ""
g_id = ""

if 1:

    ret, img = cam.read()  # read the frames using the above created object

    converted_image = cv2.cvtColor(img,
                                   cv2.COLOR_BGR2GRAY)  # The function converts an input image from one color space to another

    faces = faceCascade.detectMultiScale(
        converted_image,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # used to draw a rectangle on any image

        id, accuracy = recognizer.predict(converted_image[y:y + h, x:x + w])  # to predict on every single image
        print(f"id = {id}\naccuracy = {accuracy}")
        name = ["unknown"]
        # Check if accuracy is less them 100 ==> "0" is perfect match
        try:

            if ((accuracy) > 40):

                aa = df.loc[df['Id'] == id]['Name'].values
                name = aa
                g_name = aa[0]
                g_id = id
                accuracy = "  {0}%".format(round(100 - accuracy))
                tt = str(id) + "-" + aa

            else:
                id = "unknown"
                accuracy = "  {0}%".format(round(accuracy))

                tt = str(id)
        except:
            # os.abort()
            print("Some error ocuured, try again")
            sys.exit(5)

        cv2.putText(img, str(name[0]), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(accuracy), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('camera', img)

    k = cv2.waitKey() & 0xff  # Press 'ESC' for exiting video

# attendance = pd.read_csv('StudentDetails/')

ts = time.time()
date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
Hour, Minute, Second = timeStamp.split(":")

import csv

header = ["ID", "NAME", "DATE", "TIMESTAMP"]
row = [g_id, g_name, date, timeStamp]
if (os.path.isfile("StudentDetails" + os.sep + "attendance.csv")):
    with open("StudentDetails" + os.sep + "attendance.csv", 'a+') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(j for j in row)
        csvFile.close()
else:
    with open("StudentDetails" + os.sep + "attendance.csv", 'a+') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(i for i in header)
        writer.writerow(j for j in row)
        csvFile.close()

# Do a bit of cleanup
print("Thanks for using this program, have a good day.")
cam.release()
cv2.destroyAllWindows()
