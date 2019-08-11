# -*- coding:utf-8 -*-
from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse

morph_size = 0
max_operator = 4
max_elem = 2
max_kernel_size = 21
title_trackbar_operator_type = 'Operator:\n 0: Opening - 1: Closing  \n 2: Gradient - 3: Top Hat \n 4: Black Hat'
title_trackbar_element_type = 'Element:\n 0: Rect - 1: Cross - 2: Ellipse'
title_trackbar_kernel_size = 'Kernel size:\n 2n + 1'
title_window = 'Morphology Transformations Demo'
morph_op_dic = {0: cv.MORPH_OPEN, 1: cv.MORPH_CLOSE, 2: cv.MORPH_GRADIENT, 3: cv.MORPH_TOPHAT, 4: cv.MORPH_BLACKHAT}


def get_structure(morph_elem, morph_size):
    return cv.getStructuringElement(morph_elem, (2 * morph_size + 1, 2 * morph_size + 1), (morph_size, morph_size))


def morphology_operations(val):
    morph_operator = cv.getTrackbarPos(title_trackbar_operator_type, title_window)
    morph_size = cv.getTrackbarPos(title_trackbar_kernel_size, title_window)
    morph_elem = 0
    val_type = cv.getTrackbarPos(title_trackbar_element_type, title_window)
    if val_type == 0:
        morph_elem = cv.MORPH_RECT
    elif val_type == 1:
        morph_elem = cv.MORPH_CROSS
    elif val_type == 2:
        morph_elem = cv.MORPH_ELLIPSE
    element = get_structure(morph_elem, morph_size)
    operation = morph_op_dic[morph_operator]
    dst = cv.morphologyEx(src, operation, element)
    cv.imshow(title_window, dst)


def main():
    cv.namedWindow(title_window)
    cv.createTrackbar(title_trackbar_operator_type, title_window, 0, max_operator, morphology_operations)
    cv.createTrackbar(title_trackbar_element_type, title_window, 0, max_elem, morphology_operations)
    cv.createTrackbar(title_trackbar_kernel_size, title_window, 0, max_kernel_size, morphology_operations)
    morphology_operations(0)
    cv.waitKey()


if __name__ == '__main__':
    src = cv.imread("samples/imgs/test/arrows.png")
    src = 255 - src

    # # closing 2, ellipse
    # closing_ele = get_structure(cv.MORPH_ELLIPSE, 2)
    # src = cv.morphologyEx(src, cv.MORPH_CLOSE, closing_ele)
    # cv.imshow("close", src)
    #
    # # opening 3, ellipse
    # opening_ele = get_structure(cv.MORPH_ELLIPSE, 3)
    # src = cv.morphologyEx(src, cv.MORPH_OPEN, opening_ele)
    # cv.imshow("open", src)
    #
    # # cv.rectangle(src, (500, 65), (510, 75), (0, 255, 0), 2)
    # # cv.imshow("circle", src)
    # # threshold
    # src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    # _, binary = cv.threshold(src, 130, 255, cv.THRESH_BINARY)
    # print(src[65, 501])
    # print(src[110, 495])
    # print(binary[65, 501])
    # print(binary[110, 495])
    #
    # cv.imshow("binary", binary)
    # cv.waitKey(0)

    main()


