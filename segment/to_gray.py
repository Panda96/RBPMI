from __future__ import print_function
import cv2 as cv
import numpy as np
import os

erosion_size = 0
max_elem = 2
max_kernel_size = 21
title_trackbar_element_type = 'Element:\n 0: Rect \n 1: Cross \n 2: Ellipse'
title_trackbar_kernel_size = 'Kernel size:\n 2n +1'
title_erosion_window = 'Erosion Demo'
title_dilatation_window = 'Dilation Demo'


def get_structure_ele(morph_elem, morph_size):
    return cv.getStructuringElement(morph_elem, (2 * morph_size + 1, 2 * morph_size + 1), (morph_size, morph_size))


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


path = "samples/imgs"

flows_imgs = os.listdir(path)
index_list = list(range(len(flows_imgs)))
# index_list = [0, 4, 6, -1]
for index in index_list:
    img = flows_imgs[index]
    img_path = path + "/" + img
    src = cv.imread(img_path, cv.COLOR_BGR2GRAY)
    # src = 255 - src

    morph_size = 2
    morph_elem = cv.MORPH_RECT
    operation = cv.MORPH_BLACKHAT

    element = get_structure_ele(morph_elem, morph_size)
    src = cv.morphologyEx(src, operation, element)

    src.dtype = np.uint8
    src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    cv.namedWindow(title_erosion_window)
    cv.createTrackbar(title_trackbar_element_type, title_erosion_window, 0, max_elem, erosion)
    cv.createTrackbar(title_trackbar_kernel_size, title_erosion_window, 0, max_kernel_size, erosion)
    cv.namedWindow(title_dilatation_window)
    cv.createTrackbar(title_trackbar_element_type, title_dilatation_window, 0, max_elem, dilatation)
    cv.createTrackbar(title_trackbar_kernel_size, title_dilatation_window, 0, max_kernel_size, dilatation)
    erosion(0)
    # dilatation(0)
    cv.waitKey()
