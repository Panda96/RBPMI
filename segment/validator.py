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
classifier_type = "vgg16_57"
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
        for shape_id in shapes_list:
            # [file_id, element_type, shape_bound, element_id]
            ele_label = shapes_label[shape_id]
            shape_rec = ele_label[2]
            shape_area = shape_rec[2] * shape_rec[3]
            overlap_area = helper.get_overlap_area(e_rec, shape_rec)
            if overlap_area / ele_area > 0.5 and overlap_area / shape_area > 0.5:
                # print("here")
                shapes_list.remove(shape_id)
                image_ele_id_map[e_index] = ele_label[-1]

                ele_shape_type = ele_label[1]
                # print(ele_shape_type)
                matched = True
                # detected_num, type_right, type_wrong
                shape_type_res = validate_result[ele_shape_type].get("detect", [0, 0, 0])
                shape_type_res[0] += 1
                if e_type == ele_shape_type:
                    shape_type_res[1] += 1
                else:
                    shape_type_res[2] += 1
                validate_result[ele_shape_type]["detect"] = shape_type_res
                break

        if not matched:
            image_ele_id_map[e_index] = ""
            fake_elements.append(e_index)
        return shapes_list

    validate_result = defaultdict(dict)
    fake_elements = []

    shapes_label, _, flows_label = count.count_one_bpmn(bpmn_file)
    _, all_elements_info, all_seq_flows, all_elements, pools = detector.detect(image_file, classifier, classifier_type)

    flow_shapes = []
    sub_p_shapes = []
    # [file_id, element_type, shape_bound, element_id]
    for shape_index, shape_label in enumerate(shapes_label):
        shape_type = shape_label[1]
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
    image_ele_id_map = dict()

    for ele_index, ele_path in enumerate(all_elements):
        ele_rec = get_element_rec_by_path(ele_path, pools)
        ele_type = all_elements_info[ele_index][0]

        if ele_path[-1] == 0:
            # flow elements
            flow_shapes = match_ele_and_shape(ele_index, ele_rec, ele_type, flow_shapes)
        else:
            # expanded sub processes
            sub_p_shapes = match_ele_and_shape(ele_index, ele_rec, ele_type, sub_p_shapes)

    # [file seq num, not matched, target match, source match, others]
    seq_result = [0, 0, 0, 0, 0]

    flows_label_rest = []

    for flow_id in range(len(flows_label)):
        if flows_label[flow_id][1] == "sequenceFlow":
            flows_label_rest.append(flow_id)

    seq_result[0] = len(flows_label_rest)

    for seq_flow in all_seq_flows:
        target_ele_id = seq_flow[0]
        source_ele_id = seq_flow[-1]
        target_ele_ref = image_ele_id_map[target_ele_id]
        source_ele_ref = image_ele_id_map[source_ele_id]

        same_target_seqs = []
        for flow_id in flows_label_rest:
            # [file_id, main_type, points_label, element_id, source_ref, target_ref]
            flow_label = flows_label[flow_id]
            target_ref = flow_label[-1]
            source_ref = flow_label[-2]

            if target_ele_ref == target_ref:
                same_target_seqs.append([flow_id, source_ref])

        # matched = False
        if len(same_target_seqs) > 0:
            seq_result[2] += 1
            for same_target_seq in same_target_seqs:
                if source_ele_ref == same_target_seq[1]:
                    seq_result[3] += 1
                    flows_label_rest.remove(same_target_seq[0])
                    # matched = True
                    break

        else:
            seq_result[-1] += 1

    seq_result[1] = len(flows_label_rest)

    interested_labels = labels.copy()
    interested_labels.append("subProcess_expanded")

    print(image_file)
    for label in interested_labels:
        label_result = validate_result[label]
        # print(label)
        # print(len(label_result))
        if len(label_result) > 0:
            # print(label)
            total = label_result["total"]
            if len(label_result) == 2:
                detect = label_result["detect"]
            else:
                print("not detected")
                detect = [0, 0, 0]
            print("{}\t{},{},{},{}".format(label, total, detect[0], detect[1], detect[2]))
    seq_record = "sequenceFlow\t{},{},{},{},{}".format(seq_result[0], seq_result[1],
                                                       seq_result[2], seq_result[3], seq_result[4])
    print(seq_record)

    print("=" * 100)


def validate():
    data_dir = "E:/diagrams/bpmn-io/bpmn2image/data0423/admission/"

    bpmn_dir = data_dir + "bpmn/"
    images_dir = data_dir + "images/"

    bpmns = os.listdir(bpmn_dir)
    bpmns.sort()
    images = os.listdir(images_dir)
    images.sort()

    for i in range(len(bpmns))[0:1]:
        bpmn_file = bpmn_dir + bpmns[i]
        image_file = images_dir + images[i]
        # print(bpmn_file)
        # print(image_file)
        validate_one(bpmn_file, image_file)


validate()
