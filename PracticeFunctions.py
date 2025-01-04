import cv2
import numpy as np
import face_recognition as fr

# BaseImage: String -> Image
# Loads the given Path to an image
def BaseImage(path: str):
    # loads file
    file = fr.load_image_file(path)
    # converts to colors
    file = cv2.cvtColor(file, cv2.COLOR_BGR2RGB)
    return file

# LoadEncode: String -> List
# Loads and Encodes the given Image path, returning the image and its encoding
def LoadEncode(path: str) -> list:
    img = BaseImage(path)
    # encodes
    encoding = fr.face_encodings(img)[0]
    return [img, encoding]

# DrawImage: Image -> Image
# Draws the given Image
def DrawImage(img):
    faceLoc = fr.face_locations(img)[0]
    cv2.rectangle(img, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    
# CompareFaces: Encoding Encoding -> Image
# Determines how similar encoding 2 is to encoding 1
def CompareFaces(en1, en2):
    result = fr.compare_faces([en1], en2)
    dist = fr.face_distance([en1], en2)
    match = "matched!" if result[0] else "did not match."
    return f"The faces {match} The second image was {dist[0]} away from the first."
