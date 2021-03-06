# -*- coding:utf-8 -*-
from __future__ import print_function
import cv2 as cv


erosion_size = 0
max_elem = 2
max_kernel_size = 5
title_trackbar_element_type = 'Element:\n 0: Rect \n 1: Cross \n 2: Ellipse'
title_trackbar_kernel_size = 'Kernel size:\n 2n +1'
title_erosion_window = 'Erosion Demo'
title_dilatation_window = 'Dilation Demo'


def erosion(val):
    erosion_size = cv.getTrackbarPos(title_trackbar_kernel_size, title_erosion_window)
    erosion_type = 0
    val_type = cv.getTrackbarPos(title_trackbar_element_type, title_erosion_window)
    if val_type == 0:
        erosion_type = cv.MORPH_RECT
    elif val_type == 1:
        erosion_type = cv.MORPH_CROSS
    elif val_type == 2:
        erosion_type = cv.MORPH_ELLIPSE
    element = cv.getStructuringElement(erosion_type, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                       (erosion_size, erosion_size))
    erosion_dst = cv.erode(src, element)
    cv.imshow(title_erosion_window, erosion_dst)


def dilatation(val):
    dilatation_size = cv.getTrackbarPos(title_trackbar_kernel_size, title_dilatation_window)
    dilatation_type = 0
    val_type = cv.getTrackbarPos(title_trackbar_element_type, title_dilatation_window)
    if val_type == 0:
        dilatation_type = cv.MORPH_RECT
    elif val_type == 1:
        dilatation_type = cv.MORPH_CROSS
    elif val_type == 2:
        dilatation_type = cv.MORPH_ELLIPSE
    element = cv.getStructuringElement(dilatation_type, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                       (dilatation_size, dilatation_size))
    dilatation_dst = cv.dilate(src, element)
    cv.imshow(title_dilatation_window, dilatation_dst)


def main():
    cv.namedWindow(title_erosion_window)
    cv.createTrackbar(title_trackbar_element_type, title_erosion_window, 0, max_elem, erosion)
    cv.createTrackbar(title_trackbar_kernel_size, title_erosion_window, 0, max_kernel_size, erosion)
    cv.namedWindow(title_dilatation_window)
    cv.createTrackbar(title_trackbar_element_type, title_dilatation_window, 0, max_elem, dilatation)
    cv.createTrackbar(title_trackbar_kernel_size, title_dilatation_window, 0, max_kernel_size, dilatation)
    erosion(0)
    dilatation(0)
    cv.waitKey()


if __name__ == '__main__':
    src = cv.imread("samples/imgs/test/027_06_FIN-STABILIZER-INSTALLATION.png")
    # src = 255 - src
    main()
