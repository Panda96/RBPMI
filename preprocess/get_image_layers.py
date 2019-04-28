# -*- coding:utf-8 -*-
import numpy as np


def divide_layers(hierarchy):
    res = np.where(hierarchy[0, :, 3] == -1)
    layer = list(res[0])
    # layer = get_contours_bt(CONTOUR_AREA_THRESHOLD, layer)

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

        # next_layer = get_contours_bt(CONTOUR_AREA_THRESHOLD, next_layer)

        layer_dic[layer_count] = curr_layer
        layer = next_layer
    # {layer_num:{c_i:c_children}}
    return layer_dic
