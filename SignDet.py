import numpy
import cv2
from random import *
import math

def angle(pt1,pt2,pt0):
    dx1 = pt1[0][0] - pt0[0][0]
    dy1 = pt1[0][1] - pt0[0][1]
    dx2 = pt2[0][0] - pt0[0][0]
    dy2 = pt2[0][1] - pt0[0][1]
    return float((dx1*dx2 + dy1*dy2))/math.sqrt(float((dx1*dx1 + dy1*dy1))*(dx2*dx2 + dy2*dy2) + 1e-10)

imag = cv2.imread("speed_40.bmp")
pts = numpy.array([(0, 0), (200, 0), (200, 300), (0, 300)], dtype = "float32")

maxHeight = 160
maxWidth = 160

dst = numpy.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
M = cv2.getPerspectiveTransform(pts, dst)
warped = cv2.warpPerspective(imag, M, (maxWidth, maxHeight))


image_original = cv2.imread('speedsign5.jpg', 1)

image = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)

image = cv2.blur( image, (3 ,3))

canny_threshold = 120

image = cv2.Canny(image, canny_threshold, canny_threshold * 2, apertureSize=3)



cv2.imshow('edge image', image)

newImage = image_original.copy()

_, contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
numberOfSquare = 0
listOfCor = 0
test = image_original.copy()
for cnts in range(0, len(contours)):
    approx = cv2.approxPolyDP(contours[cnts], len(contours[cnts])*0.08  , True)
    print len (approx)

    if len(approx) == 4:
        print (approx)

        x = randint(1, 255)
        cv2.drawContours(newImage, [approx], 0, x, 2)
        cv2.imshow('Polyder', test)
        cv2.waitKey(0)

        vtc = len(approx)
        print("New Shape")
        for j in range(2, vtc + 1):
            print("Angle")
            ang = angle(approx[j % vtc], approx[j - 2], approx[j - 1])
            print(ang)

            if (ang > 0.1 or ang < -0.1): break
            x = abs(approx[0,0,0] - approx[1,0,0])
            print (x)
            if ( x <20):break

            if (j == 4):
                vtc = len(approx)
                print "OCTAGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG"
                listOfCor = approx
                x = randint(1, 255)
                newImage = image_original.copy()
                cv2.drawContours(newImage, [approx], 0, x, 2)
                cv2.imshow('Polyd', newImage)
                numberOfSquare = numberOfSquare + 1
                #cv2.waitKey(0)




print ("*******************************")
print (numberOfSquare)
print(listOfCor)


cv2.imshow('Polyd', newImage)

cv2.waitKey(0)

















cv2.imshow("Warped", warped)
cv2.waitKey(0)