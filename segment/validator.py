# -*- coding:utf-8 -*-
import sys

sys.path.append("..")

import os

from preprocess import count
import detector
from classifier import Classifier
from collections import defaultdict

from helper import detector_helper as helper

classifier = Classifier()
classifier_type = "vgg16"
labels = classifier.classes_57


def get_element_rec_by_path(path, pools):
    pool = pools[path[0]]
    # print(path)
    if path[3] == 0:
        elements = pool.get("elements", defaultdict(list))
        elements_i = elements.get(path[1], [])
        rect = elements_i[path[2]]
    else:
        sub_procs = pool["sub_procs"]
        sub_procs_in_lane = sub_procs.get(path[1])
        rect = sub_procs_in_lane[path[2]]
    return rect


def validate_one(bpmn_file, image_file):
    def match_ele_and_shape(e_index, e_rec, e_type, shapes_list):
        ele_area = e_rec[2] * e_rec[3]

        matched = False
        for shape_index in shapes_list:
            # [file_id, element_type, shape_bound, element_id]
            shape_label = shapes_label[shape_index]
            shape_rec = shape_label[2]
            shape_area = shape_rec[2] * shape_rec[3]
            overlap_area = helper.get_overlap_area(e_rec, shape_rec)
            if overlap_area / ele_area > 0.8 and overlap_area / shape_area > 0.8:
                shapes_list.remove(shape_index)
                image_ele_id_map[e_index] = shape_label[-1]
                shape_type = shape_label[1]
                matched = True
                # detected_num, type_right, type_wrong
                shape_type_res = validate_result[shape_type].get("detect", [0, 0, 0])
                shape_type_res[0] += 1
                if e_type == shape_type:
                    shape_type_res[1] += 1
                else:
                    shape_type_res[2] += 1
                break

        if not matched:
            image_ele_id_map[e_index] = ""
            fake_elements.append(e_index)
        return shapes_list

    validate_result = defaultdict(dict)
    fake_elements = []
    image_ele_id_map = dict()

    shapes_label, _, flows_label = count.count_one_bpmn(bpmn_file)
    _, all_elements_info, all_seq_flows, all_elements, pools = detector.detect(image_file, classifier, classifier_type)

    flow_shapes = []
    sub_p_shapes = []
    # [file_id, element_type, shape_bound, element_id]
    for shape_index, shape_label in enumerate(shapes_label):
        shape_type = shapes_label[1]
        if shape_type in labels:
            type_num = validate_result[shape_type].get("total", 0)
            type_num += 1
            validate_result[shape_type]["total"] = type_num
            flow_shapes.append(shape_index)
        else:
            if "expanded" in shape_type.split("_"):
                shapes_label[shape_index][1] = "subProcess_expanded"
                type_num = validate_result["subProcess_expanded"].get("total", 0)
                type_num += 1
                validate_result["subProcess_expanded"]["total"] = type_num
                sub_p_shapes.append(shape_index)

    # {detected element id: bpmn file element id}

    for ele_index, ele_path in enumerate(all_elements):
        ele_rec = get_element_rec_by_path(ele_path, pools)
        ele_type = all_elements_info[ele_index][0]

        if ele_path[-1] == 0:
            # flow elements
            flow_shapes = match_ele_and_shape(ele_index, ele_rec, ele_type, flow_shapes)
        else:
            # expanded sub processes
            sub_p_shapes = match_ele_and_shape(ele_index, ele_rec, ele_type, sub_p_shapes)

    # flow_rest =


def validate():
    data_dir = "E:/diagrams/bpmn-io/bpmn2image/data0423/admission/"

    bpmn_dir = data_dir + "bpmn/"
    image_dir = data_dir + "images/"

    bpmns = os.listdir(bpmn_dir)
    bpmns.sort()
    images = os.listdir(image_dir)
    images.sort()

    for i in range(len(bpmns)):
        bpmn_file = bpmn_dir + bpmns[i]
        image_file = bpmn_dir + images[i]
        validate_one(bpmn_file, image_file)

