# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np

file = "imgs/admission/Cologne.png"

img = cv.imread(file)
img = 255-img
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
image, contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

print(len(contours))
last = 0
for i in range(len(contours)):
    area = cv.contourArea(contours[i])
    # print(area)

    if area < 100 or len(contours[i]) > 50:
        continue
    # temp = cv.imread(file)
    # temp = 255 - temp
    for point in contours[i]:

        cv.circle(img, (point[0][0], point[0][1]), 4, (0, 255, 0), 2)
        cv.imshow("img", img)
        # if index > 0:
        #     if last > 0:
        #         cv.destroyWindow(str(last))
        #     last = index
        k = cv.waitKey(0)
        if k == ord('n'):
            break


    # cv.drawContours(temp, contours, index, (0, 255, 0), 2)

    # print("-"*50)
    # print(contours[index])
    # cv.imshow(str(index), temp)

    k = cv.waitKey(0)
    if k == ord('\n'):
        continue
    elif k == 27:
        break

cv.destroyAllWindows()
