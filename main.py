import csv
import yagmail
import cv2
import pyttsx3
import speech_recognition as sr
import os
import sys
from csv import writer
import numpy as np
import pyautogui as p
from PIL import Image
import pandas as pd
import datetime
import time

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voices', voices[0].id)

df = pd.read_csv("StudentDetails" + os.sep + "Details.csv")
df2 = pd.read_csv("StudentDetails" + os.sep + "attendance.csv")


# text to speech
def speak(audio):
    engine.say(audio)
    engine.runAndWait()


# to convert voice into text
def takecommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("LISTENING.....")
        r.pause_thresold = 2
        audio = r.listen(source, timeout=5, phrase_time_limit=5)

    try:
        print("Recognizing....")
        query = r.recognize_google(audio, language='en-in')
        print(f"user said:{query}")

    except Exception as e:
        speak("say that again please....")
        return "none"
    return query

# wish
def wish():
    hour = int(datetime.datetime.now().hour)
    if hour >= 0 and hour <= 12:
        speak("good morning")
    elif hour > 12 and hour <= 18:
        speak("good afternoon")
    else:
        speak("good evening")

def find(s):

    a = s.casefold() in list(df['Name'])
    print(a)
    if a:
        b = s.casefold() in list(df2['NAME'])
        print(b)
        print(s)
        if b:
            speak("is present do want to schedule a meet ")
            z = takecommand()
            if(z=="yes"):
                speak("can you please state your purpose ")
                purpose = input("please state here- ")

                receiver = df.loc[df['Name'] == s]['email'].values
                mail(receiver,s,purpose)
                speak("confirming your appointment,please have a seat and wait for a moment ")


            if(z=="no"):
                speak("ok cancelling your request have a nice day ")


        else:
            speak("it seems Master is not currently available at this moment can you came back later.")

    else:
        speak("no such registerd person found can you please re-confirm")
        takecommand()


def TaskExecutioner():
    p.press('esc')

    while 1:
        query = takecommand().lower()

        if "meet" in query:
            speak("whom you want to meet ")
            s = input("enter a name ")
            find(s)


        elif "thank you" in query or "no" in query:
            speak("thanks for using me sir, have a good day ")
            sys.exit()

def mail(receiver,name,purpose):
    date = datetime.date.today().strftime("%B %d, %Y")
    sub = "Request for meet " + str(date)
    body = [
        "respected sir ",
        "a person wants to meet you",
        name ,
        "wants to meet you for",
        purpose,
        "regards"
        "receptionist"
    ]

    # # mail information
    # yag = yagmail.SMTP("smarteceptionist@gmail.com", "minor@2022")
    #
    # # sent the mail
    # yag.send(
    #     to=receiver,
    #     subject=sub,  # email subject
    #     contents=body,  # email body
    #
    # )
    print("Email Sent!")
    speak("please wait for a moment")
    speak("do you have any other work")
    TaskExecutioner()


def facerecognition():
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # Local Binary Patterns Histograms
    recognizer.read('trainer/trainer.yml')  # load trained model
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)  # initializing haar cascade for object detection approach

    df = pd.read_csv("StudentDetails" + os.sep + "Details.csv")
    col_names = ['ID', 'NAME', 'DATE', 'TIMESTAMP']
    font = cv2.FONT_HERSHEY_SIMPLEX  # denotes the font type

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # cv2.CAP_DSHOW to remove warning
    cam.set(3, 640)  # set video FrameWidht
    cam.set(4, 480)  # set video FrameHeight

    # Define min window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    # flag = True

    g_name = ""
    g_id = ""

    if True:

        ret, img = cam.read()  # read the frames using the above created object

        converted_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # The function converts an input image from one color
        # space to another

        faces = faceCascade.detectMultiScale(
            converted_image,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # used to draw a rectangle on any image

            id, accuracy = recognizer.predict(converted_image[y:y + h, x:x + w])  # to predict on every single image

            name = ["unknown"]
            # Check if accuracy is less them 100 ==> "0" is perfect match
            try:

                if (accuracy) > 35:
                    aa = df.loc[df['Id'] == id]['Name'].values
                    name = aa
                    g_name = aa[0]
                    g_id = id
                    accuracy = "  {0}%".format(round(100 - accuracy))
                    tt = str(id) + "-" + aa

                    wish()
                    speak("welcome back")
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
                    speak(aa)
                    speak(" I am a bot and I will be your personal assistant for now how can i help you")
                    TaskExecutioner()

                else:
                    id = "unknown"
                    accuracy = "  {0}%".format(round(accuracy))
                    speak(" verification unsucessful")
                    speak("registering yourself")
                    register()
                    model_trainer()
                    facerecognition()

                    tt = str(id)
            except Exception as e:
                # os.abort()
                print(type(e).__name__)
                print("Some error ocuured, try again")
                sys.exit(5)

            cv2.putText(img, str(name[0]), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(accuracy), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        cv2.imshow('camera', img)

        k = cv2.waitKey() & 0xff  # Press 'ESC' for exiting video

    # attendance = pd.read_csv('StudentDetails/')



    # Do a bit of cleanup
    print("Thanks for using this program, have a good day.")
    cam.release()
    cv2.destroyAllWindows()


# sample_generator
# def samplegenerator():
#     cam = cv2.VideoCapture(0,
#                            cv2.CAP_DSHOW)  # create a video capture object which is helpful to capture videos through webcam
#     cam.set(3, 640)  # set video FrameWidth
#     cam.set(4, 480)  # set video FrameHeight
#
#     detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#     # Haar Cascade classifier is an effective object detection approach
#     face_id = 1
#     face_id = input("enter face id ")
#     name = input("enter your name")
#     email = input("enter your email")
#     # Use integer ID for every new face (0,1,2,3,4,5,6,7,8,9........)
#
#     print("Taking samples, look at camera ....... ")
#     speak("Taking samples, look at camera ....... ")
#     count = 0  # Initializing sampling face count
#
#     while True:
#
#         ret, img = cam.read()  # read the frames using the above created object
#         converted_image = cv2.cvtColor(img,
#                                        cv2.COLOR_BGR2GRAY)  # The function converts an input image from one color space to another
#         faces = detector.detectMultiScale(converted_image, 1.3, 5)
#
#         for (x, y, w, h) in faces:
#             cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # used to draw a rectangle on any image
#             count += 1
#
#             cv2.imwrite("samples/" + str(name) + '_' + str(face_id) + '.' + str(count) + ".jpg",
#                         converted_image[y:y + h, x:x + w])
#             # To capture & Save images into the datasets folder
#
#             cv2.imshow('image', img)  # Used to display an image in a window
#
#         k = cv2.waitKey(100) & 0xff  # Waits for a pressed key
#         if k == 27:  # Press 'ESC' to stop
#             break
#         elif count >= 10:  # Take 50 sample (More sample --> More accuracy)
#             break
#
#     print("Samples taken now closing the program....")


def model_trainer():
    path = 'samples'  # Path for samples already taken

    recognizer = cv2.face.LBPHFaceRecognizer_create()  # Local Binary Patterns Histograms
    detector = cv2.CascadeClassifier(
        "haarcascade_frontalface_default.xml")  # Haar Cascade classifier is an effective object detection approach

    def Images_And_Labels(path):  # function to fetch the images and labels

        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        ids = []

        for imagePath in imagePaths:  # to iterate particular image path

            gray_img = Image.open(imagePath).convert('L')  # convert it to grayscale
            img_arr = np.array(gray_img, 'uint8')  # creating an array

            # id = int(os.path.split(imagePath).split(".")[1])
            # print(os.path.split(imagePath))
            id = int(os.path.split(imagePath)[1].split("_")[1].split(".")[0])
            faces = detector.detectMultiScale(img_arr)

            for (x, y, w, h) in faces:
                faceSamples.append(img_arr[y:y + h, x:x + w])
                ids.append(id)
        print(f"id = {ids}")
        return faceSamples, ids

    print("Training faces. It will take a few seconds. Wait ...")

    faces, ids = Images_And_Labels(path)
    recognizer.train(faces, np.array(ids))

    recognizer.write('trainer/trainer.yml')  # Save the trained model as trainer.yml

    print("Model trained, Now we can recognize your face.")
    speak("Model trained, Now we can recognize your face. Thank YOU for your patience ")


def register():
    # Take image function
    speak("enter your ID")
    Id = input("Enter Your Id: ")
    speak("enter your name")
    name = input("Enter Your Name: ").casefold()
    speak("enter your email")
    email = input("enter your email id ")
    speak("enter your status student/faculty ")
    status = input("student / faculty - ")

    cam = cv2.VideoCapture(0,
                           cv2.CAP_DSHOW)  # create a video capture object which is helpful to capture videos through webcam
    cam.set(3, 640)  # set video FrameWidth
    cam.set(4, 480)  # set video FrameHeight
    face_id = 0

    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    sampleNum = 0

    while True:

        ret, img = cam.read()  # read the frames using the above created object
        converted_image = cv2.cvtColor(img,
                                       cv2.COLOR_BGR2GRAY)  # The function converts an input image from one color space to another
        faces = detector.detectMultiScale(converted_image, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # used to draw a rectangle on any image
            sampleNum += 1

            cv2.imwrite("samples/" + str(name) + '_' + str(Id) + '.' + str(sampleNum) + ".jpg",
                        converted_image[y:y + h, x:x + w])
            # To capture & Save images into the datasets folder

            cv2.imshow('image', img)  # Used to display an image in a window

        k = cv2.waitKey(100) & 0xff  # Waits for a pressed key
        if k == 27:  # Press 'ESC' to stop
            break
        elif sampleNum >= 10:  # Take 50 sample (More sample --> More accuracy)
            break

    print("Samples taken now closing the program....")
    cam.release()
    cv2.destroyAllWindows()

    res = "Images Saved for ID : " + Id + " Name : " + name
    speak(res)
    header = ["Id", "Name", "email", "status"]
    row = [Id, name, email, status]
    if os.path.isfile("StudentDetails" + os.sep + "Details.csv"):
        with open("StudentDetails" + os.sep + "Details.csv", 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(j for j in row)
            csvFile.close()
    else:
        with open("StudentDetails" + os.sep + "Details.csv", 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(i for i in header)
            writer.writerow(j for j in row)
            csvFile.close()

facerecognition()
