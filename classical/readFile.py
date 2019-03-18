# -*- coding:utf-8 -*-
import os
import cv2 as cv
import numpy as np


def read_img(dir_path, file):
    file_name = "/".join([dir_path, file])
    img = cv.imread(file_name)
    img = 255 - img
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    image, contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print(file)

    # print(len(contours))

    count = 0
    for contour in contours:
        count += 1
        print(contour.shape)
        cv.drawContours(img, contours, count-1, (0, 255, 0), 1)
        cv.namedWindow(str(count), cv.WINDOW_NORMAL)
        cv.imshow(str(count), img)
        print("-"*30)
    print("*"*50)
    for h in hierarchy:
        print(h)
        print("-" * 30)

    print("*"*100)

    cv.imshow("img", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


dir_path = "imgs/elements"

# for file in os.listdir(dir_path):
#     read_img(dir_path, file)
for file in ["TaskService.png"]:
    read_img(dir_path, file)



# imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# # # imgray = 255 - imgray
# # ret, thresh = cv.threshold(imgray, 127, 255, 0)
# # image, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# #
# # print(contours)

# img = cv.drawContours(img, contours, -1, (0, 255, 0), 1)


# img = 255-img

# cv.imshow("img", img)
# cv.imshow("edges", edges)


# cv.waitKey(0)
# cv.destroyAllWindows()


