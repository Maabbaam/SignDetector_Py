import numpy as np
import cv2

needle = cv2.imread("stop4.jpg")
haystack = cv2.imread("speedsign3.jpg")

cv2.imshow("red",needle)
cv2.imshow("re",haystack)
cv2.waitKey(0)
result = cv2.matchTemplate(haystack, needle, cv2.TM_CCOEFF_NORMED)

print (result)
