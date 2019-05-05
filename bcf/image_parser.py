# -*- coding:utf-8 -*-
import os
import cv2 as cv
import numpy as np

import helper.rec_helper as rh


CONTOUR_AREA_THRESHOLD = 150


def get_one_contour_rec(contour):
    contour_poly = cv.approxPolyDP(contour, 3, True)
    bound_rect = cv.boundingRect(contour_poly)
    contour_area = cv.contourArea(contour)
    rec_area = bound_rect[2] * bound_rect[3]
    # bound_rect (x, y, width, height)(x,y) is the coordinate of the left-top point
    bound = (bound_rect, contour_area, rec_area)
    return bound


def filter_contours(image_shape, contours, area_threshold=CONTOUR_AREA_THRESHOLD, contour_list=None):
    rec1 = (0, 0, 25, 25)
    rec2 = ((image_shape[1]-30)//2, image_shape[0] - 15, 30, 15)

    contours_id = []

    if contour_list is None:
        contour_list = list(range(len(contours)))

    for i in contour_list:
        bound = get_one_contour_rec(contours[i])
        if bound[1] > area_threshold:
            contours_id.append(i)
        else:
            if rh.is_overlap(bound[0], rec1) or (rh.is_overlap(bound[0], rec2)
                                                 and (bound[0][2] > 8 or bound[0][3] > 8)):
                contours_id.append(i)

    return contours_id


def divide_layers(image_shape, hierarchy, contours):
    res = np.where(hierarchy[0, :, 3] == -1)
    layer = list(res[0])
    layer = filter_contours(image_shape, contours, CONTOUR_AREA_THRESHOLD, layer)

    layer_dic = {}
    layer_count = -1
    while len(layer) > 0:
        layer_count += 1
        # get the contours in the next layer
        next_layer = []
        curr_layer = {}
        for c in layer:
            res = np.where(hierarchy[0, :, 3] == c)
            c_children = list(res[0])
            curr_layer[c] = c_children
            next_layer.extend(c_children)

        next_layer = filter_contours(image_shape, contours, CONTOUR_AREA_THRESHOLD, next_layer)

        layer_dic[layer_count] = curr_layer
        layer = next_layer
    # {layer_num:{c_i:c_children}}
    return layer_dic


def show_layers(input_img, layers, contours):
    for i in range(len(layers)):
        layer = layers[i].keys()
        contours_drawing = draw_contours(input_img, contours, layer)
        contours_drawing = rh.dilate_drawing(contours_drawing)
        img = rh.dilate_drawing(input_img)

        cv.imshow("input", img)
        cv.imshow("contour", contours_drawing)
        k = cv.waitKey(0)

        if k == 27:
            print("break one image")
            break


def draw_contours(input_img, contours, contors):
    drawing = np.zeros_like(input_img, dtype=np.uint8)
    for i in contors:
        cv.drawContours(drawing, contours, i, (255, 255, 255), 1, cv.LINE_8)

    return drawing


def get_layers_by_file_name(file_name):
    input_img = 255 - cv.imread(file_name, cv.IMREAD_GRAYSCALE)
    return get_layers_by_img(input_img)


def get_layers_by_img(input_img):
    input_img = 255 - input_img
    _, input_binary = cv.threshold(input_img, 50, 255, cv.THRESH_BINARY)
    # cv.imshow("input_binary", rh.dilate_drawing(input_binary))
    # input_binary = cv.Canny(input_img, 50, 200)
    # cv.imshow("input_binary", rh.dilate_drawing(input_binary))
    # input_binary = cv.Canny(input_binary, 50, 200)
    # cv.imshow("input_binary_3", rh.dilate_drawing(input_binary))
    # cv.imshow(input)
    # cv.imshow("input", rh.dilate_drawing(input_binary))
    # cv.waitKey(0)
    input_binary.dtype = np.uint8
    input_binary = cv.cvtColor(input_binary, cv.COLOR_BGR2GRAY)

    _, contours, hierarchy = cv.findContours(input_binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    layers = divide_layers(input_img.shape, hierarchy, contours)

    return input_img, layers, contours


# def train():
#     # train_data = "E:/diagrams/bpmn-io/bpmn2image/data0423/data_set/train/"
#     train_data = "data/train/"
#
#     type_list = os.listdir(train_data)
#
#     for type_name in type_list:
#         # print(type_name)
#         files = os.listdir(train_data + type_name)
#         for file in files:
#             file_path = train_data + type_name + "/" + file
#             # print(file_path)
#             input_img, layers, contours = get_layers(file_path)
#             show_layers(input_img, layers, contours)
#             k = cv.waitKey()
#             if k == 27:
#                 print("break one type")
#                 break


# if __name__ == '__main__':
#     file_path = ""
#     input_img, layers, contours = get_layers(file_path)
#     show_layers(input_img, layers, contours)

    # a = rh.shrink((0, 0, 100, 100), 5)
    # print(a)
