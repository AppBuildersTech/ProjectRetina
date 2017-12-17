# -*- coding: utf-8 -*-

import numpy as np
import os, sys
import pandas as pd
from config import *
import cv2
import copy
from matplotlib import pyplot as plt
from scipy.stats import entropy
from scipy.ndimage.morphology import binary_opening
from skimage.exposure import adjust_gamma
from skimage.morphology import remove_small_objects
import pywt

def imagesAndLabels(trndataPath,df):

	# https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    i = 0
    featureMap = pd.DataFrame(columns=('bvMean','bvMax','bvNumber','bvArea','opticDist','entropy','label'))

    for (f,label) in zip(df['image'],df['level']):
        progressBar(i)

        img = cv2.imread(trndataPath + '/' + f + '.jpeg')

        # Images should be normalized.
        resizedImage = cv2.resize(img,(512,512))

        # Split images into their channels. Usually green channel is used, since related work suggests there is more information on green channel compared to others.
        blue, green, red = cv2.split(resizedImage)

        # Adaptive histogram equalization is applied to green channel.
        heqGreen = clahe.apply(green)
        heqRed = clahe.apply(red)

        bvMean,bvMax,bvNumber,bvArea = vessels(heqGreen,clahe)
        opticDist = opticalDisk(img)
        entropy = shannonEntropy(green)
        featureMap.loc[f] = (bvMean,bvMax,bvNumber,bvArea,opticDist,entropy,label)

        i += 1
        sys.stdout.flush()

    return featureMap

def progressBar(i):
    verbose1 = "Extracting features for .. "
    bar = '█' * int(50 * (i+1) // 5000) + '-' * (50 - int(50 * (i+1) // 5000))
    template = ("{0:." + str(1) + "f}").format(100 * ((i+1) / float(5000)))
    verbose2 = "Complete"
    sys.stdout.write('\r%s\t |%s| %s%% %s' % (verbose1, bar, template, verbose2))

def shannonEntropy(img):
	hist = cv2.calcHist([img],[0],None,[256],[0,256])/(np.prod(img.shape))
	return entropy(hist)[0]

def opticalDisk(image):
    blue, green, red = cv2.split(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(155,155))

    tophat = cv2.morphologyEx(green, cv2.MORPH_TOPHAT, kernel)
    tophat = cv2.medianBlur(tophat,5)

    green = cv2.equalizeHist(green)
    red = cv2.equalizeHist(red)

    # Bright spots in green and red
    _, green = cv2.threshold(green,green.max()-5,255,cv2.THRESH_BINARY)
    _, red = cv2.threshold(red,red.max()-5,255,cv2.THRESH_BINARY)

    # Remove small spots using median filter. Bright spots both in red and green are yellow. So we can take bitwise and of red and green channels and again remove undesired bright spots.
    red = cv2.medianBlur(red,5)
    green = cv2.medianBlur(green,5)

    output = cv2.bitwise_and(red,green)
    output = cv2.medianBlur(output,65)
    cont = cv2.bitwise_not(output)

    _, contours, _ = cv2.findContours(output, 1, 2)

    # The end points of optic disk
    left = tuple(green.shape);
    right = tuple([0,0]);
    top = tuple(green.shape);
    bottom = tuple([0,0]);
    for item in contours:
         area = cv2.contourArea(item)
         nl = tuple(item[item[:,:,0].argmin()][0])
         nr = tuple(item[item[:,:,0].argmax()][0])
         nt = tuple(item[item[:,:,1].argmin()][0])
         nb = tuple(item[item[:,:,1].argmax()][0])

         if nl[0] <= left[0]:
            left = nl
         if nr[0] >= right[0] :
            right = nr
         if nt[1] <= top[1] :
            top = nt
         if nb[1] >= bottom[1] :
            bottom = nb

    center = list(map(sum, zip(top,bottom,left,right)))

    center[0] = int(center[0]/4);
    center[1] = int(center[1]/4);
    center = np.array(center)

    # Find largest distance to center
    distTop = np.linalg.norm(center-np.array(top))
    distBot = np.linalg.norm(center-np.array(bottom))
    distLeft = np.linalg.norm(center-np.array(left))
    distRight = np.linalg.norm(center-np.array(right))
    distance = np.array([distTop,distBot,distLeft,distRight])
    radius = np.max(distance)
    center = tuple(center);
    tophat = cv2.circle(tophat,center, int(radius)+40, (255,255,255), -1)

    return np.sqrt((image.shape[0]/2 - center[0])**2 + (image.shape[1]/2 - center[1])**2)

def vessels(green,clahe):
	# Open-Close
    openClose = cv2.morphologyEx(green, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    openClose = cv2.morphologyEx(openClose, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    openClose = cv2.morphologyEx(openClose, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)), iterations = 1)
    openClose = cv2.morphologyEx(openClose, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)), iterations = 1)
    openClose = cv2.morphologyEx(openClose, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(19,19)), iterations = 1)
    openClose = cv2.morphologyEx(openClose, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(19,19)), iterations = 1)
    openClose = cv2.subtract(openClose,green)
    openClose = clahe.apply(openClose)

    # Remove small contours
    _ ,thresholdedOpenClose = cv2.threshold(openClose,15,255,cv2.THRESH_BINARY)
    mask = np.ones(openClose.shape[:2], dtype="uint8") * 255
    _, contours, _ = cv2.findContours(thresholdedOpenClose.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) <= 200:
           cv2.drawContours(mask, [cnt], -1, 0, -1)
    img = cv2.bitwise_and(openClose, openClose, mask=mask)
    _ ,img = cv2.threshold(img,15,255,cv2.THRESH_BINARY_INV)
    img = cv2.erode(img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)

    # Remove clustered white zones
    fundusRemoved = cv2.bitwise_not(img)
    mask = np.ones(green.shape[:2], dtype="uint8") * 255
    _, contours, _ = cv2.findContours(fundusRemoved.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        app = cv2.approxPolyDP(cnt, 0.04 * perimeter, False)
        if len(app) > 4 and cv2.contourArea(cnt) <= 3000 and cv2.contourArea(cnt) >= 100:
	        shape = "circle"
        else:
	        shape = "veins"
        if shape == "circle":
            cv2.drawContours(mask, [cnt], -1, 0, -1)
    vessel = cv2.bitwise_and(fundusRemoved,fundusRemoved,mask=mask)
    vessel = cv2.bitwise_not(vessel)

    kernel = np.ones((5,5),np.uint8)

    _, contours, _ = cv2.findContours(vessel, 1, 2)
    thickness = [];
    nvessels = len(contours);
    totalArea = 0;
    for items in contours:
        area = cv2.contourArea(items)
        totalArea = totalArea + area;
        perimeter = cv2.arcLength(items,True)
        if perimeter != 0:
            thickness.append(area/perimeter);
    if len(thickness) != 0:
        maxVessel = np.asarray(thickness).max();
    else:
        maxVessel = 0;
    mean = np.average(np.asarray(thickness));

    return (mean,maxVessel,nvessels,totalArea)

df = pd.read_csv(classificationLabelPath)
featureMap = imagesAndLabels(trndataPath,df)
featureMap.to_csv(classificationFeatures)
