# -*- coding:utf-8 -*-
import os
import cv2 as cv
import numpy as np
import pickle

import rec_helper as rh
import detector
from bcf import BCF
from shape_context import shape_context
from llc import llc_coding_approx


def get_structure_ele(morph_elem, morph_size):
    return cv.getStructuringElement(morph_elem, (2 * morph_size + 1, 2 * morph_size + 1), (morph_size, morph_size))


def draw_contours(input_img, contours, contors):
    drawing = np.zeros_like(input_img, dtype=np.uint8)
    for i in contors:
        cv.drawContours(drawing, contours, i, detector.COLOR_WHITE, detector.CONTOUR_THICKNESS, cv.LINE_8)

    return drawing


def pre_process(file_name):
    input_img = 255 - cv.imread(file_name, cv.IMREAD_GRAYSCALE)
    # cv.imshow("input", rh.dilate_drawing(input_img))

    _, input_binary = cv.threshold(input_img, 50, 255, cv.THRESH_BINARY)
    # cv.imshow("input_binary", rh.dilate_drawing(input_binary))
    # input_binary = cv.Canny(input_img, 50, 200)
    # cv.imshow("input_binary", rh.dilate_drawing(input_binary))
    # input_binary = cv.Canny(input_binary, 50, 200)
    # cv.imshow("input_binary_3", rh.dilate_drawing(input_binary))
    _, contours, hierarchy = cv.findContours(input_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #
    detector.CONTOUR_AREA_THRESHOLD = 100
    layers = detector.divide_layers(hierarchy, contours)

    selected_contours_id = []
    for i in range(min(len(layers), 2)):
        selected_contours_id.extend(layers[i].keys())

    selected_contours = []
    for contour_id in selected_contours_id:
        selected_contours.append(contours[contour_id])

    shape_feature = []

    for contour in selected_contours:
        print(len(contour))
        points = []
        for point in contour:
            points.append([point[0][0], point[0][1]])
        max_curvature = 1.5
        n_contsamp = 50
        n_pntsamp = 10
        bcf = BCF()
        cfs = bcf._extr_raw_points(np.array(points), max_curvature, n_contsamp, n_pntsamp)

        num_cfs = len(cfs)
        print("Extracted %s points" % num_cfs)
        contour_feature = np.zeros((300, num_cfs))
        for i in range(num_cfs):
            cf = cfs[i]
            sc, _, _, _ = shape_context(cf)
            # shape context is 60x5 (60 bins at 5 reference points)
            sc = sc.flatten(order='F')
            sc /= np.sum(sc)  # normalize
            contour_feature[:, i] = sc

        shape_feature.append(contour_feature)


shapes_dir_1 = "E:/diagrams/bpmn-io/bpmn2image/data0423/ele_type_data/boundEvent_cancel/"
shapes_dir_2 = "E:/diagrams/bpmn-io/bpmn2image/data0423/ele_type_data/manualTask_mulInsL_seq/"
# shapes_dir_2 = "E:/diagrams/bpmn-io/bpmn2image/data0423/ele_type_data/sendTask/"

shapes_1 = os.listdir(shapes_dir_1)
shapes_2 = os.listdir(shapes_dir_2)

# for shape in shapes_1[-5:]:
#     shape_path = shapes_dir_1 + shape

#     detector.CONTOUR_AREA_THRESHOLD = 30
#     detector.get_layers_img(shape_path, True)

for shape in shapes_2:
    shape_path = shapes_dir_2 + shape
    detector.CONTOUR_AREA_THRESHOLD = 30
    pre_process(shape_path)
    # detector.get_layers_img(shape_path, True)
