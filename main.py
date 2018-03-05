
import numpy
import math
import cv2
from random import *


def angle(pt1,pt2,pt0):
    dx1 = pt1[0][0] - pt0[0][0]
    dy1 = pt1[0][1] - pt0[0][1]
    dx2 = pt2[0][0] - pt0[0][0]
    dy2 = pt2[0][1] - pt0[0][1]
    return float((dx1*dx2 + dy1*dy2))/math.sqrt(float((dx1*dx1 + dy1*dy1))*(dx2*dx2 + dy2*dy2) + 1e-10)

def write_on_image(image, text):

    fontFace = cv2.FONT_HERSHEY_DUPLEX;

    fontScale = 2.0

    thickness = 3

    textOrg = (10, 130)

    cv2.putText(image, text, textOrg, fontFace, fontScale, thickness, 8);

    return image



# Load an color image and convert to grayscale

WARPED_XSIZE = 200
WARPED_YSIZE = 300

image_original = cv2.imread('speedsign13.jpg', 1)

image = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)

image = cv2.blur( image, (5 ,5))

#warped_result = (Size(WARPED_XSIZE, WARPED_YSIZE), image.type());

cv2.imshow('original image', image_original)

numberOfOctagons = 0

speed_40 = cv2.imread("speed_40.bmp", 0);
speed_80 = cv2.imread("speed_80.bmp", 0);



# find canny edges and display them

canny_threshold = 120

image = cv2.Canny(image, canny_threshold, canny_threshold * 2, apertureSize=3)

cv2.imshow('edge image', image)

newImage = image_original.copy()

# find all contours, and save the ten biggest

_, contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

big_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

for cnts in range(0, len(contours)):
    approx = cv2.approxPolyDP(contours[cnts], len(contours[cnts])*0.08  , True)
    print len (approx)
   # cv2.drawContours(newImage, [approx], 0, 255, 4)

    if len(approx) == 8:
        vtc = len(approx)
        print("New Shape")
        for j in range(2, vtc + 1):
            print("Angle")
            ang = angle(approx[j % vtc], approx[j - 2], approx[j - 1])
            print(ang)

            if (ang > -0.65 or ang < -0.75) : break

            if (j == 8) :

                vtc = len(approx)
                print "OCTAGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG"
                x = randint(1, 255)
                newImage =image_original.copy()
                cv2.drawContours(newImage, [approx], 0, x, 2)
                cv2.imshow('Polyd', newImage)
                numberOfOctagons = numberOfOctagons + 1
                cv2.waitKey(0)



print("The number of OCTS AREEEEEEEEEE ")
print (numberOfOctagons)

if (numberOfOctagons == 2) :
    stop = image_original.copy()
    write_on_image(stop, "Stop Sign")
    cv2.imshow('StopSign', stop)


# draw each of these contours one after the other

for c in big_contours:

    image_contour = image_original.copy()

    cv2.drawContours(image_contour, [c], 0, 255, 4)

    cv2.imshow('contour', image_contour)



    area = cv2.contourArea(c)

    print ('Area:', area)

    cv2.waitKey(0)




write_on_image(image_original ,"Result")

cv2.imwrite('speedsign13-result.jpg' ,image_original)

cv2.waitKey(0)



cv2.destroyAllWindows()
