# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np


file = "imgs/admission/Cologne.png"

img = cv.imread(file)
img = 255-img
cv.imshow("img", img)
cv.waitKey(0)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 50, 150, apertureSize=3)

lines = cv.HoughLinesP(edges, 1, np.pi/180, 100)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


# cv.imshow("edges", edges)
# lines = cv.HoughLines(edges, 1, np.pi/180, 200)

# ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
# cv.imshow(thresh)
# lines = cv.HoughLines(thresh, 1, np.pi/180, 200)
#
# print(lines.shape)
# [print(line) for line in lines]

# for line in lines:
#     rho, theta = line[0]
#     print("{},{}".format(rho, theta))
    # a = np.cos(theta)
    # b = np.sin(theta)
    # x0 = a*rho
    # y0 = b*rho
    # x1 = int(x0 + 1000*(-b))
    # y1 = int(y0 + 1000*a)
    # x2 = int(x0 - 1000*(-b))
    # y2 = int(y0 - 1000*a)
    # cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv.imshow("img", img)
# #
cv.waitKey(0)
cv.destroyAllWindows()

