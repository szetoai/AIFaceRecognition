import cv2
import numpy as np
import face_recognition as fr
import os
from datetime import datetime

# Encode: Image -> Encoding
# Encodes the given Image
def Encode(img):
    newImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encoded = fr.face_encodings(newImg)[0]
    return encoded

# MarkAttendence: String -> None
# Marks the given name if it is not already in our attendence sheet
def MarkAttendence(name):
    with open("Attendence.csv", "r+") as f:
        dataList = f.readlines()
        # names
        nameList = []
        # for every entry in our attendence, add it to our nameList
        for line in dataList:
            # split the name/date
            entry = line.split(",")
            nameList.append(entry[0])
        # if this name isnt in our sheet, add it in
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime("%H:%M:%S")
            f.writelines(f"\n{name},{dtString}")

# PicToDB: None -> None
# Takes a picture and adds it to our database of pictures
def PicToDB():
    name = "Faces/" + input("What is your name? ") + ".png"
    cam = cv2.VideoCapture(0)
    result, image = cam.read()
    if result:
        cv2.imwrite(name, image)
    else:
        print("Error. Try again!")
        PicToDB()

# --------------------------------------------------------------------------- #
# FaceRecognition: None -> None
# Encodes the images in our Faces folder and checks the webcam for those faces,
# adding them into our attendence sheet if they are present.
def FaceRecognition():
    # Variable for the name of the folder that stores our faces
    path = "Faces"
    # Lists for our Images and their Names
    images = []
    names = []
    # Get a list of all the files in our images folder
    imageFiles = os.listdir(path)
    # for each file in the folder
    for file in imageFiles:
        # Get the image
        curImg = cv2.imread(f"{path}/{file}")
        # Encode image and append it into our images list
        curImg = Encode(curImg)
        images.append(curImg)
        # Append the name of the image to our names
        names.append(os.path.splitext(file)[0])
    print("Encoding Complete!")
    # Video Capturing
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        # resize image and encode it
        imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        # Get the face locations and encode them
        curFrameFaces = fr.face_locations(imgS)
        encodedFrame = fr.face_encodings(imgS, curFrameFaces)
        # Loop through each face
        for encodedFace, faceLoc in zip(encodedFrame, curFrameFaces):
            matches = fr.compare_faces(images, encodedFace)
            faceDis = fr.face_distance(images, encodedFace)
            closestMatch = np.argmin(faceDis)
            # If the closestMatch is determined to be a match
            if matches[closestMatch]:
                name = names[closestMatch].upper()
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                MarkAttendence(name)
        cv2.imshow("Webcam", img)
        cv2.waitKey(1)

def main():
    choice = input("Add New Entry? Y/N ")
    newEntry = True if choice == "Y" or choice == "y" else False
    while newEntry:
        PicToDB()
        choice = input("Add New Entry? Y/N ")
        newEntry = True if choice == "Y" or choice == "y" else False
    print("Thinking...")
    FaceRecognition()
main()