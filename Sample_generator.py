import csv
import cv2
import os
import speech_recognition as sr
import os.path
import pyttsx3
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voices',voices[0].id)

#text to speech
def speak(audio):
    engine.say(audio)
    engine.runAndWait()


# Take image function


Id = input("Enter Your Id: ")
name = input("Enter Your Name: ")
email =input("enter your email id ")
status=input("student / faculty")




cam = cv2.VideoCapture(0, cv2.CAP_DSHOW) #create a video capture object which is helpful to capture videos through webcam
cam.set(3, 640) # set video FrameWidth
cam.set(4, 480) # set video FrameHeight
face_id=0

harcascadePath = "haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(harcascadePath)
sampleNum = 0

while True:

    ret, img = cam.read() #read the frames using the above created object
    converted_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #The function converts an input image from one color space to another
    faces = detector.detectMultiScale(converted_image, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2) #used to draw a rectangle on any image
        sampleNum += 1


        cv2.imwrite("samples/" + str(name) + '_' + str(Id) + '.' + str(sampleNum) + ".jpg",
                    converted_image[y:y + h, x:x + w])
        # To capture & Save images into the datasets folder

        cv2.imshow('image', img) #Used to display an image in a window

    k = cv2.waitKey(100) & 0xff # Waits for a pressed key
    if k == 27: # Press 'ESC' to stop
        break
    elif sampleNum >= 10: # Take 50 sample (More sample --> More accuracy)
         break

print("Samples taken now closing the program....")
cam.release()
cv2.destroyAllWindows()

res = "Images Saved for ID : " + Id + " Name : " + name
speak(res)
header=["Id", "Name","email","status"]
row = [Id, name,email,status]
if(os.path.isfile("StudentDetails"+os.sep+"Details.csv")):
    with open("StudentDetails"+os.sep+"Details.csv", 'a+') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(j for j in row)
        csvFile.close()
else:
    with open("StudentDetails"+os.sep+"Details.csv", 'a+') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(i for i in header)
        writer.writerow(j for j in row)
        csvFile.close()

