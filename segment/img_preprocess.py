# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np

import cfg
from helper import detector_helper as helper


def draw_contours_rec(input_img, contors, contours_rec):
    drawing = np.zeros_like(input_img, dtype=np.uint8)
    for i in contors:
        bound = contours_rec[i]
        bound_rect = bound[0]
        drawing = helper.draw_one_rect(drawing, bound_rect, cfg.COLOR_WHITE, cfg.CONTOUR_THICKNESS)

        # contour index:contour area，area of bounding box， height-to-width-ratio
        text = "{}:{},{},{}".format(i, int(bound[1]), int(bound[2]), "%.2f" % (bound_rect[3] / bound_rect[2]))
        text_size = cv.getTextSize(text, cv.QT_STYLE_NORMAL, 0.3, 1)
        # put the text in the middle of a rectangle and not beyond the border of the image_mat
        org_x = bound_rect[0] + (bound_rect[2] - text_size[0][0]) // 2
        org_x = max(org_x, 2)
        org_x = min(org_x, drawing.shape[1] - text_size[0][0] - 5)
        cv.putText(drawing, text, (org_x, bound_rect[1] + (bound_rect[3] + text_size[0][1]) // 2),
                   cv.QT_STYLE_NORMAL, 0.3, cfg.COLOR_WHITE)
    return drawing


def draw_contours(input_img, contors, contours):
    drawing = np.zeros_like(input_img, dtype=np.uint8)
    for i in contors:
        cv.drawContours(drawing, contours, i, cfg.COLOR_WHITE, cfg.CONTOUR_THICKNESS, cv.LINE_8)

    return drawing


# def get_layers_img(f, show):
#     pre_process(f)
#
#     for model_i in range(len(layers)):
#         layer = layers[model_i].keys()
#         contours_drawing = draw_contours(layer)
#         rec_drawing = draw_contours_rec(layer)
#         contours_drawing = helper.dilate_drawing(contours_drawing)
#         rec_drawing = helper.dilate_drawing(rec_drawing)
#         img = helper.dilate_drawing(input_img)
#
#         if show:
#             # show_im(img, "input")
#             # show_im(contours_drawing, "contour")
#             # show_im(rec_drawing, "rect")
#             cv.imshow("input", img)
#             cv.imshow("contour", contours_drawing)
#             cv.imshow("rect", rec_drawing)
#             cv.waitKey(0)
#         else:
#             output_dir = "samples/layers/" + f.split("/")[-1][:-4]
#             if not os.path.exists(output_dir):
#                 os.makedirs(output_dir)
#             contour_file = output_dir + "/" + "layer_{}_contour.png".format(model_i)
#             rec_file = output_dir + "/" + "layer_{}_rec.png".format(model_i)
#             cv.imwrite(contour_file, contours_drawing)
#             cv.imwrite(rec_file, rec_drawing)
#     cv.destroyAllWindows()


def get_contours(image):
    # 形态学变化，将图中箭头连接处断开
    morph_size = 2
    morph_elem = cv.MORPH_RECT
    operation = cv.MORPH_BLACKHAT

    element = helper.get_structure_ele(morph_elem, morph_size)
    morph = cv.morphologyEx(image, operation, element)
    # show(morph, "morph")

    # 获取粗边框的元素的位置
    erosion_element = helper.get_structure_ele(morph_elem, 1)
    erosion = cv.erode(morph, erosion_element)
    # show(erosion, "erosion")

    erosion.dtype = np.uint8
    erosion_gray = cv.cvtColor(erosion, cv.COLOR_BGR2GRAY)
    _, erosion_binary = cv.threshold(erosion_gray, 100, 255, cv.THRESH_BINARY)
    _, erosion_contours, erosion_hierarchy = cv.findContours(erosion_binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # 获取粗边框元素边界矩形
    partial_elements_rec = []
    for i, contour in enumerate(erosion_contours):
        contour_rec = helper.get_one_contour_rec(i, erosion_contours)
        if contour_rec[2] > cfg.CONTOUR_AREA_THRESHOLD and erosion_hierarchy[0][i][3] == -1:
            partial_elements_rec.append(contour_rec[0])

    # 获取细边框元素轮廓
    morph.dtype = np.uint8
    morph_gray = cv.cvtColor(morph, cv.COLOR_BGR2GRAY)
    _, morph_binary = cv.threshold(morph_gray, 50, 255, cv.THRESH_BINARY)
    _, morph_contours, morph_hierarchy = cv.findContours(morph_binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return morph_contours, morph_hierarchy, partial_elements_rec


# remove the contours with small area
def get_contours_bt(contours, area_threshold, contour_list):
    return list(filter(lambda i: cv.contourArea(contours[i]) > area_threshold, contour_list))


def divide_layers(contours, hierarchy):
    res = np.where(hierarchy[0, :, 3] == -1)
    layer = list(res[0])
    layer = get_contours_bt(contours, cfg.CONTOUR_AREA_THRESHOLD, layer)

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
            c_children = get_contours_bt(contours, cfg.CONTOUR_AREA_THRESHOLD, c_children)
            curr_layer[c] = c_children
            next_layer.extend(c_children)

        next_layer = get_contours_bt(contours, cfg.CONTOUR_AREA_THRESHOLD, next_layer)

        layer_dic[layer_count] = curr_layer
        layer = next_layer
    # {layer_num:{c_i:c_children}}
    return layer_dic


def get_contours_rec(contours, layers):
    contours_rec = [None] * len(contours)
    layers_num = len(layers)
    for i in range(layers_num):
        layer = layers[i].keys()
        for c_i in layer:
            contours_rec[c_i] = helper.get_one_contour_rec(c_i, contours)
    return contours_rec


def pre_process(file_path):
    input_img = cv.imread(file_path)
    contours, hierarchy, partial_elements = get_contours(input_img)
    layers = divide_layers(contours, hierarchy)
    contours_rec = get_contours_rec(contours, layers)
    return input_img, layers, contours, contours_rec, partial_elements
