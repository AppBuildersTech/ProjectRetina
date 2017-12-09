import numpy as np
import os, sys
import pandas as pd
from config import *
import cv2
import copy

def imagesAndLabels(path):

    labels = list()
    names = list()
    images = list()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    for f in os.listdir(path):

        img = cv2.imread(path + '/' + f)

        # print(blood_vessels(img))

        # This part shows how distinctive green channel is in terms of vascularization
		# retina = copy.copy(image);
        blue, green, red = cv2.split(img)
        # res = np.hstack((red,green,blue))
        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # cv2.imshow('image',res)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # The first histogram equalization we just saw, considers the global contrast of the image. In many cases, it is not a good idea. For example, below image shows an input image and its result after global histogram equalization. It is true that the background contrast has improved after histogram equalization. But compare the face of statue in both images. We lost most of the information there due to over-brightness. It is because its histogram is not confined to a particular region as we saw in previous cases (Try to plot histogram of input image, you will get more intuition).

        # So to solve this problem, adaptive histogram equalization is used. In this, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region (unless there is noise). If noise is there, it will be amplified. To avoid this, contrast limiting is applied. If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. After equalization, to remove artifacts in tile borders, bilinear interpolation is applied.

		# https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html

        heqImage = clahe.apply(green)
        resizedImage = cv2.resize(heqImage,(512,512))

        # print(getRadius(green))
        blood_vessels = extract_bv(heqImage,clahe)
        # print(exudates(heqImage,red,blue))
        # res = np.hstack((cv2.resize(green,(512,512)),resizedImage,blood_vessels))
        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # cv2.imshow('image',res)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        images.append(np.array(heqImage))

        names.append(f)
        sample = f.split(os.path.sep)[-1].split(".")[0]
        line =  imlabels.loc[imlabels['image'] == sample]
        labels.append(int(line['level']))

    return images, np.array(labels)

def exudates(retina_green,retina_red,retina_blue):

    test_green = copy.copy(retina_green)
    a = retina_green.max();
    test_green = copy.copy(retina_green)
    retina_g = cv2.equalizeHist(retina_green)
    retina_r = cv2.equalizeHist(retina_red)
    #detection of optical disc
    #finding the brightest spots in red and green channels
    ret, green = cv2.threshold(retina_g,retina_g.max()-5,255,cv2.THRESH_BINARY)
    ret, red = cv2.threshold(retina_r,retina_r.max()-5,255,cv2.THRESH_BINARY)
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
    img,contours,hierarchy = cv2.findContours(output, 1, 2)
    leftmost = retina_blue.shape;
    rightmost = [0,0];
    topmost = retina_blue.shape;
    bottommost = [0,0];
    for item in contours:
        area = cv2.contourArea(item)
        perimeter = cv2.arcLength(item,True);
        if(perimeter !=0):
            R= 4*np.pi*area/np.power(perimeter,2);
            if (R>0.3):
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
    centre = [0,0];
    centre[1] = int((topmost[1] + bottommost[1])/2)
    centre[0] = int((leftmost[0] + rightmost[0])/2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(75,75))
    radius = np.sqrt(np.power(topmost[1]-2-centre[1],2) + np.power(leftmost[0]-2-centre[0],2));
    radius = int(radius);
    if (radius<25):
       radius = 25;
    elif (radius>40):
        radius = 40;
    centre = tuple(centre);
    cv2.circle(retina_green,centre, int(radius)+5, (0,0,0), -1);
    kernel = np.ones((10,10),np.uint8)
    tophat = cv2.morphologyEx(retina_green, cv2.MORPH_TOPHAT, kernel)
    tophat = cv2.equalizeHist(tophat)
    ret, tophat = cv2.threshold(tophat,tophat.max()-10,255,cv2.THRESH_BINARY)
    tophat = cv2.morphologyEx(tophat, cv2.MORPH_CLOSE, kernel)
    exudate = copy.copy(tophat);
    img,contours,hierarchy = cv2.findContours(tophat, 1, 2)
    mask = np.ones(retina_green.shape, dtype="uint8") * 255
    # removing false exudates
    area = [];
    distance = 0;
    number = 0;
    total_area = 0;
    for items in contours:
        temp_area = cv2.contourArea(items)
        perimeter = cv2.arcLength(items,True)
        #print temp_area
        if perimeter != 0:
                 R= 4*np.pi*temp_area/np.power(perimeter,2);
        else :
                 R=0;
        if (R<0.3) or  (temp_area>3000) :
                         cv2.drawContours(mask, [items], -1, 0, -1)
        else:
                     number = number + 1;
                     M = cv2.moments(items);
                     if(M['m00'] != 0):
                         cx = int(M['m10']/M['m00'])
                         cy = int(M['m01']/M['m00'])
                         distance = distance + np.sqrt(np.power(centre[1]-cy,2) + np.power(centre[0]-cx,2));
                     area.append(temp_area);
                     total_area = total_area + temp_area;
    exudate = cv2.bitwise_and(exudate,exudate, mask=mask);
    exudate = cv2.medianBlur(exudate,5);
    test = cv2.bitwise_and(test_green,exudate)
    b1 = test.max();
    test[(test>(b1-10))] = 0;
    b2 = test.max();
    test[(test>(b2-10))] = 0;
    b3 = test.max();
    #tophat = cv2.adaptiveThreshold(tophat,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,301,0)
    if(number!=0):
        distance = distance/number;
        output = [total_area,np.asarray(area).max(),distance,number,b1/a,b2/a,b3/a];
    else :
        output = [total_area,0,0,number,b1/a,b2/a,b3/a];
    return output

def getRadius(img):
    green = cv2.bitwise_not(img);
    ret, green = cv2.threshold(green,green.max()-1,255,cv2.THRESH_BINARY);
    green = cv2.bitwise_not(green)
    green = cv2.medianBlur(green,55)
    img,contours,hierarchy = cv2.findContours(green, 1, 2)
    area = 0;
    for item in contours:
        if cv2.contourArea(item) > area :
            area = cv2.contourArea(item)
            cnt = item
    left   = list(tuple(cnt[cnt[:,:,0].argmin()][0]))
    right  = list(tuple(cnt[cnt[:,:,0].argmax()][0]))
    top    = list(tuple(cnt[cnt[:,:,1].argmin()][0]))
    bottom = list(tuple(cnt[cnt[:,:,1].argmax()][0]))
    r1 = np.sqrt((right[0]-left[0])**2 + (right[1]-left[1])**2)/2
    r2 = np.sqrt((top[0]-bottom[0])**2 + (top[1]-bottom[1])**2)/2
    r  = (r1 + r2)/2;

    return r;
def extract_bv(green_fundus,clahe):
	# applying alternate sequential filtering (3 times closing opening)
    r1 = cv2.morphologyEx(green_fundus, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
    R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
    r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
    R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
    f4 = cv2.subtract(R3,green_fundus)
    f5 = clahe.apply(f4)

    # removing very small contours through area parameter noise removal
    ret,f6 = cv2.threshold(f5,15,255,cv2.THRESH_BINARY)
    mask = np.ones(f5.shape[:2], dtype="uint8") * 255
    im2, contours, hierarchy = cv2.findContours(f6.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) <= 200:
           cv2.drawContours(mask, [cnt], -1, 0, -1)
    im = cv2.bitwise_and(f5, f5, mask=mask)
    ret,fin = cv2.threshold(im,15,255,cv2.THRESH_BINARY_INV)
    newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)

    # removing blobs of unwanted bigger chunks taking in consideration they are not straight lines like blood
    #vessels and also in an interval of area
    fundus_eroded = cv2.bitwise_not(newfin)
    xmask = np.ones(green_fundus.shape[:2], dtype="uint8") * 255
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
    img,contours,hierarchy = cv2.findContours(blood_vessel, 1, 2)
    thickness = [];
    number_vessel = len(contours);
    total_area = 0;
    for items in contours:
        area = cv2.contourArea(items)
        total_area = total_area + area;
        perimeter = cv2.arcLength(items,True)
        if(perimeter !=0) :
            thickness.append(area/perimeter);
    if(len(thickness)!=0):
        max_vessel = np.asarray(thickness).max();
    else:
        max_vessel = 0;
    mean = np.average(np.asarray(thickness));
    print(mean,max_vessel,number_vessel,total_area);
    return blood_vessel


imlabels = pd.read_csv(sampleLabelPath)
images, labels = imagesAndLabels(samplePath)

# blood = blood_vessels(images[0])
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.imshow('image',blood)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#
# from sklearn.cross_validation import train_test_split
#
# (trainRI, testRI, trainRL, testRL) = train_test_split(img_vect, labels, test_size=0.25, random_state=42)
#
# print(trainRI)
# from sklearn import svm
# from sklearn.svm import SVC
#
# clf = svm.SVC()
# clf = svm.SVC(gamma=0.0001, C=10)
# clf.fit(trainRI,trainRL)
# acc =clf.score(testRI,testRL)
# print (acc)
# print (clf.predict(testRI))
