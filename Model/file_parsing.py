import os
import cv2
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def gatherImagePaths(folder):

    imagePaths = []
    for root, dirs, files in os.walk(folder):

        for name in files:

            imagePaths.append(os.path.abspath(os.path.join(root, name)))
                              
    return imagePaths

def loadImages(imagePaths, width, height):

    imageFiles = []
      
    for filename in imagePaths:

        img = cv2.imread(filename)
    
        if img is not None:
            resized_img = cv2.resize(img, (width, height))
            imageFiles.append(resized_img)

    return imageFiles

def resizeImagePath(imagePath, width = 500, height = 500):

    img = cv2.imread(imagePath)

    if img is not None:
        resized_img = cv2.resize(img, (width, height))
        return resized_img
    

def resizeImage(img, width = 500, height = 500):

    if img is not None:
        resized_img = cv2.resize(img, (width, height))
        return resized_img


def cropImage(img, borderCropPerc):

    if (borderCropPerc < 0.0 or borderCropPerc >= 0.5):
        raise AssertionError("borderCropPerc out of range")
    
    if (img is None):
        raise AssertionError("Image provided is invalid")

    x, y, z = img.shape
    x_border = round(x * borderCropPerc)
    y_border = round(y * borderCropPerc)

    cropped_img = img[x_border: x - x_border, y_border : y - y_border]
    return cropped_img


def greyscaleImage(img):

    if img is not None:
        greyscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return greyscaled_img

def preprocessImage(img):

    if img is not None:

        img = cropImage(img, 0.1)
        img = resizeImage(img, 500, 500)
        img = greyscaleImage(img)
        img = img / 255.0
        
        return img