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

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    i = 0
    featureMap = pd.DataFrame(columns=('bvMean','bvMax','bvNumber','bvArea','opticDist','exArea','exMax','exDistance','exNumber','ex1','ex2','ex3','entropy','label'))

    for (f,label) in zip(df['image'],df['level']):
        # progressBar(i)
        img = cv2.imread(trndataPath + '/' + f + '.jpeg')
        img = cv2.resize(img,(512,512))

        blue, green, red = cv2.split(img)

		# https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
        heqImage = clahe.apply(green)

        (bvMean,bvMax,bvNumber,bvArea) = extract_bv(heqImage,clahe)
        opticDist = opticalDisk(img)
        (exArea,exMax,exDistance,exNumber,ex1,ex2,ex3) = exudates(heqImage,red)
        entropy = shannonEntropy(green)
        output = hemorrhage(green,clahe)
        print(output,f)
        # featureMap.loc[f] = (bvMean,bvMax,bvNumber,bvArea,opticDist,exArea,exMax,exDistance,exNumber,ex1,ex2,ex3,entropy,label)
        # print(hemorrhage(green,clahe))
        # resizedImage = cv2.resize(blood_vessel,cA.shape)

        # res = np.hstack((blood_vessel,ext))
        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('image', 1500,1500)
        # cv2.imshow('image',cA)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        i += 1
        # sys.stdout.flush()

    return featureMap

def hemorrhage(green,clahe):
    testGreen = copy.copy(green)
    greenMax = green.max()
    screen_res = 1366, 768
    scale_width = screen_res[1] / green.shape[1]
    scale_height = screen_res[0] / green.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(green.shape[1] * scale)
    window_height = int(green.shape[0] * scale)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #preprocessing of the images
    green[green<20] = 255;
    green = cv2.bitwise_not(green);
    green = clahe.apply(green);
    green = cv2.equalizeHist(green)
    _, green = cv2.threshold(green,green.max()-10,255,cv2.THRESH_BINARY);
    exudate = copy.copy(green);
    #cv2.imshow('output',dilation);

    _,contours,_ = cv2.findContours(green, 1, 2);
    left = green.shape
    right = [0,0]
    top = green.shape
    bottom = [0,0]
    for item in contours:
        area = cv2.contourArea(item)
        perimeter = cv2.arcLength(item,True)
        if perimeter != 0:
            R = 4 * np.pi * area / perimeter**2
            if R > 0.1:
            	nl = tuple(item[item[:,:,0].argmin()][0])
            	nr = tuple(item[item[:,:,0].argmax()][0])
            	nt = tuple(item[item[:,:,1].argmin()][0])
            	nb = tuple(item[item[:,:,1].argmax()][0])
            	if nl[0] <= left[0]:
            		left = nl
            	if nr[0] >= right[0]:
            		right = nr
            	if nt[1] <= top[1]:
            		top = nt
            	if nb[1] >= bottom[1]:
            		bottom = nb
    center = [0,0];
    center[1] = int((top[1] + bottom[1])/2)
    center[0] = int((left[0] + right[0])/2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(75,75))
    r = np.sqrt((top[1]-2-center[1])**2 + (left[0]-2-center[0])**2)
    r = int(r)

    if r < 25:
       r = 25;
    elif r > 40:
    	r = 40;

    center = tuple(center);
    cv2.circle(green,center, int(r)+5, (0,0,0), -1);

    kernel = np.ones((10,10),np.uint8)
    tophat = cv2.morphologyEx(green, cv2.MORPH_TOPHAT, kernel)
    tophat = cv2.equalizeHist(tophat)
    _ , tophat = cv2.threshold(tophat,tophat.max()-10,255,cv2.THRESH_BINARY)
    tophat = cv2.morphologyEx(tophat, cv2.MORPH_CLOSE, kernel)

    exudate = copy.copy(tophat);
    _,contours,_ = cv2.findContours(tophat, 1, 2)
    mask = np.ones(green.shape, dtype="uint8") * 255

    area = []
    distance = 0
    number = 0
    totalArea = 0

    for items in contours:
        tempArea = cv2.contourArea(items)
        perimeter = cv2.arcLength(items,True)

        if perimeter != 0:
            R= 4*np.pi*tempArea/(perimeter**2);
        else:
            R = 0
        print(R,tempArea)
        if (R<0.02) or  (tempArea<4)  :
            cv2.drawContours(mask, [items], -1, 0, -1)
        else:
            number = number + 1
            M = cv2.moments(items)
            if(M['m00'] != 0):
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                distance = distance + np.sqrt((center[1]-cy)**2 + (center[0]-cx)**2)
            area.append(tempArea)
            totalArea = totalArea + tempArea

        exudate = cv2.bitwise_and(exudate,exudate, mask=mask)
        exudate = cv2.medianBlur(exudate,5)

        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('image', 1500,1500)
        # cv2.imshow('image',exudate)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        test = cv2.bitwise_and(testGreen,exudate)
        b1 = test.max()
        test[(test > (b1 - 10))] = 0
        b2 = test.max()
        test[(test > (b2 - 10))] = 0
        b3 = test.max()

        if number != 0:
            distance = distance/number
            output = [totalArea,np.asarray(area).max(),distance,number,b1/greenMax,b2/greenMax,b3/greenMax]
        else:
            output = [totalArea,0,0,number,b1/greenMax,b2/greenMax,b3/greenMax]
        return output

    return output

def progressBar(i):
    verbose1 = "Extracting features for .. "
    bar = '█' * int(50 * (i+1) // 2400) + '-' * (50 - int(50 * (i+1) // 2400))
    template = ("{0:." + str(1) + "f}").format(100 * ((i+1) / float(2400)))
    verbose2 = "Complete"
    sys.stdout.write('\r%s\t |%s| %s%% %s' % (verbose1, bar, template, verbose2))

def shannonEntropy(img):
	hist = cv2.calcHist([img],[0],None,[256],[0,256])/(np.prod(img.shape))
	return entropy(hist)[0]

def exudates(green,red):
    testGreen = copy.copy(green)
    greenMax = green.max();

    green = cv2.equalizeHist(green)
    red = cv2.equalizeHist(red)

	#detection of optical disc
    #finding the brightest spots in red and green channels
    _, green = cv2.threshold(green,green.max()-5,255,cv2.THRESH_BINARY)
    _, red = cv2.threshold(red,red.max()-5,255,cv2.THRESH_BINARY)

	# median filtering to remove the unwanted bright spots
    red = cv2.medianBlur(red,5)
    green = cv2.medianBlur(green,5)

	# usually the bright spot is yellow coloured so it should be bright in red and green channels
    # so taking bitwise and to remove the unwanted spots from both the channels
    output = cv2.bitwise_and(red,green)

	# median filtering to remove the residual (unwanted) bright spots;
    output = cv2.medianBlur(output,7)

    kernel = np.ones((5,5),np.uint8)
    output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel)

	#copying image before applying findcontour as findcontour modifies the image
    _,contours,_ = cv2.findContours(output, 1, 2)
    left = green.shape
    right = [0,0]
    top = green.shape
    bottom = [0,0]
    for item in contours:
        area = cv2.contourArea(item)
        perimeter = cv2.arcLength(item,True)
        if perimeter != 0:
            R = 4 * np.pi * area / perimeter**2
            if R > 0.3:
                nl = tuple(item[item[:,:,0].argmin()][0])
                nr = tuple(item[item[:,:,0].argmax()][0])
                nt = tuple(item[item[:,:,1].argmin()][0])
                nb = tuple(item[item[:,:,1].argmax()][0])
                if nl[0] <= left[0]:
                    left = nl
                if nr[0] >= right[0]:
                    right = nr
                if nt[1] <= top[1]:
                    top = nt
                if nb[1] >= bottom[1]:
                    bottom = nb
    center = [0,0];
    center[1] = int((top[1] + bottom[1])/2)
    center[0] = int((left[0] + right[0])/2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(75,75))
    r = np.sqrt((top[1]-2-center[1])**2 + (left[0]-2-center[0])**2)
    r = int(r)

    if r < 25:
       r = 25;
    elif r > 40:
        r = 40;

    center = tuple(center);
    cv2.circle(green,center, int(r)+5, (0,0,0), -1);

    kernel = np.ones((10,10),np.uint8)
    tophat = cv2.morphologyEx(green, cv2.MORPH_TOPHAT, kernel)
    tophat = cv2.equalizeHist(tophat)
    _ , tophat = cv2.threshold(tophat,tophat.max()-10,255,cv2.THRESH_BINARY)
    tophat = cv2.morphologyEx(tophat, cv2.MORPH_CLOSE, kernel)

    exudate = copy.copy(tophat);
    _,contours,_ = cv2.findContours(tophat, 1, 2)
    mask = np.ones(green.shape, dtype="uint8") * 255

    # removing false exudates
    area = []
    distance = 0
    number = 0
    totalArea = 0
    for items in contours:
        tempArea = cv2.contourArea(items)
        perimeter = cv2.arcLength(items,True)

        if perimeter != 0:
            R = 4*np.pi*tempArea/np.power(perimeter,2)
        else :
            R = 0
        if R < 0.3 or tempArea > 3000:
            cv2.drawContours(mask, [items], -1, 0, -1)
        else:
            number = number + 1
            M = cv2.moments(items)
            if(M['m00'] != 0):
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                distance = distance + np.sqrt((center[1]-cy)**2 + (center[0]-cx)**2)
            area.append(tempArea)
            totalArea = totalArea + tempArea

    exudate = cv2.bitwise_and(exudate,exudate, mask=mask)
    exudate = cv2.medianBlur(exudate,5)
    #
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('image', 1500,1500)
    # cv2.imshow('image',exudate)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    test = cv2.bitwise_and(testGreen,exudate)
    b1 = test.max()
    test[(test > (b1 - 10))] = 0
    b2 = test.max()
    test[(test > (b2 - 10))] = 0
    b3 = test.max()

    if number != 0:
        distance = distance/number
        output = [totalArea,np.asarray(area).max(),distance,number,b1/greenMax,b2/greenMax,b3/greenMax]
    else:
        output = [totalArea,0,0,number,b1/greenMax,b2/greenMax,b3/greenMax]
    return output

def opticalDisk(image):
    blue, green, red = cv2.split(image)
    screen_res = 1280, 720
    scale_width = screen_res[1] / blue.shape[1]
    scale_height = screen_res[0] / blue.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(blue.shape[1] * scale)
    window_height = int(blue.shape[0] * scale)

    cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('output', window_width, window_height)

    #cv2.namedWindow('true', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('true', window_width, window_height)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(155,155))
    #erosion = cv2.erode(green,kernel,iterations = 2)
    tophat = cv2.morphologyEx(green, cv2.MORPH_TOPHAT, kernel)
    tophat = cv2.medianBlur(tophat,25)
    #tophat = cv2.medianBlur(tophat,5)


    #cv2.imshow('true',green)
    #cv2.imshow('erosion',tophat)

    '''
    cv2.namedWindow('red', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('red', window_width, window_height)
    cv2.namedWindow('blue', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('blue', window_width, window_height)
    '''

    #preprocessing of the images
    retina_g = cv2.equalizeHist(green)
    retina_b = cv2.equalizeHist(blue)
    retina_r = cv2.equalizeHist(red)

    #detection of optical disc
    #finding the brightest spots in red and green channels
    _, green = cv2.threshold(retina_g,retina_g.max()-5,255,cv2.THRESH_BINARY)
    _, red = cv2.threshold(retina_r,retina_r.max()-5,255,cv2.THRESH_BINARY)

    # median filtering to remove the unwanted bright spots
    red = cv2.medianBlur(red,5)
    green = cv2.medianBlur(green,5)

    # usually the bright spot is yellow coloured so it should be bright in red and green channels
    # so taking bitwise and to remove the unwanted spots from both the channels
    output = cv2.bitwise_and(red,green)

    # median filtering to remove the residual (unwanted) bright spots;
    output = cv2.medianBlur(output,65)
    cont = cv2.bitwise_not(output)
    #copying image befor applying findcontour as findcontour modifies the image
    images, contours,_ = cv2.findContours(output, 1, 2)

    # locating the extreme points in the image to mask out the optical disc
    leftmost = tuple(blue.shape);
    rightmost = tuple([0,0]);
    topmost = tuple(blue.shape);
    bottommost = tuple([0,0]);
    for item in contours:
     ##print "item :", item
     area = cv2.contourArea(item)
     new_left = tuple(item[item[:,:,0].argmin()][0])
     new_right = tuple(item[item[:,:,0].argmax()][0])
     new_top = tuple(item[item[:,:,1].argmin()][0])
     new_bottom = tuple(item[item[:,:,1].argmax()][0])

     if new_left[0] <= leftmost[0] :
         leftmost = new_left
         if new_right[0] >= rightmost[0] :
            rightmost = new_right
         if new_top[1] <= topmost[1] :
            topmost = new_top
         if new_bottom[1] >= bottommost[1] :
            bottommost = new_bottom
	# mask to remove optical disc
	# images after optical disc removal
	#tophat[topmost[1]-200:bottommost[1]+200 , (leftmost[0]-200):(rightmost[0]+200)] = black_spot;
    centre = list(map(sum, zip(topmost,bottommost,leftmost,rightmost)))

    centre[0] = int(centre[0]/4);
    centre[1] = int(centre[1]/4);
    centre = np.array(centre)

    # the largest point from the centre
    dist_top = np.linalg.norm(centre-np.array(topmost))
    dist_bottom = np.linalg.norm(centre-np.array(bottommost))
    dist_left = np.linalg.norm(centre-np.array(leftmost))
    dist_right = np.linalg.norm(centre-np.array(rightmost))
    distance = np.array([dist_top,dist_bottom,dist_left,dist_right])
    radius = np.max(distance)
    centre = tuple(centre);
    tophat = cv2.circle(tophat,centre, int(radius)+100, (255,255,255), -1)

    #_, output_red = cv2.threshold(retina_r,retina_r.max()-60,255,cv2.THRESH_BINARY);
    #output_gr   = cv2.bitwise_and(output_green,output_red);
    #cont2 = cv2.bitwise_not(output_gr);
    #images, contours,_ = cv2.findContours(output_gr, 1, 2)

    return np.sqrt((image.shape[0]/2 - centre[0])**2 + (image.shape[1]/2 - centre[1])**2)

def extract_bv(green,clahe):
	# applying alternate sequential filtering (3 times closing opening)
    r1 = cv2.morphologyEx(green, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
    R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
    r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
    R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
    f4 = cv2.subtract(R3,green)
    f5 = clahe.apply(f4)

    # removing very small contours through area parameter noise removal
    _ ,f6 = cv2.threshold(f5,15,255,cv2.THRESH_BINARY)
    mask = np.ones(f5.shape[:2], dtype="uint8") * 255
    im2, contours, _ = cv2.findContours(f6.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) <= 200:
           cv2.drawContours(mask, [cnt], -1, 0, -1)
    im = cv2.bitwise_and(f5, f5, mask=mask)
    _ ,fin = cv2.threshold(im,15,255,cv2.THRESH_BINARY_INV)
    newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)

    # removing blobs of unwanted bigger chunks taking in consideration they are not straight lines like blood
    #vessels and also in an interval of area
    fundus_eroded = cv2.bitwise_not(newfin)
    xmask = np.ones(green.shape[:2], dtype="uint8") * 255
    x1, xcontours, xhierarchy = cv2.findContours(fundus_eroded.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in xcontours:
        shape = "unidentified"
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, False)
        if len(approx) > 4 and cv2.contourArea(cnt) <= 3000 and cv2.contourArea(cnt) >= 100:
	        shape = "circle"
        else:
	        shape = "veins"
        if(shape=="circle"):
            cv2.drawContours(xmask, [cnt], -1, 0, -1)
    finimage = cv2.bitwise_and(fundus_eroded,fundus_eroded,mask=xmask)
    blood_vessel = cv2.bitwise_not(finimage)
    kernel = np.ones((5,5),np.uint8)
    blood_vessel = cv2.morphologyEx(blood_vessel, cv2.MORPH_CLOSE, kernel)
    _,contours,_ = cv2.findContours(blood_vessel, 1, 2)
    thickness = [];
    number_vessel = len(contours);
    totalArea = 0;
    for items in contours:
        area = cv2.contourArea(items)
        totalArea = totalArea + area;
        perimeter = cv2.arcLength(items,True)
        if(perimeter !=0) :
            thickness.append(area/perimeter);
    if(len(thickness)!=0):
        max_vessel = np.asarray(thickness).max();
    else:
        max_vessel = 0;
    mean = np.average(np.asarray(thickness));

    return (mean,max_vessel,number_vessel,totalArea)

df = pd.read_csv(detectionLabelPath)
featureMap = imagesAndLabels(trndataPath,df)
featureMap.to_csv(detectionFeatures)
