# -*- coding:utf-8 -*-
import sys
import re

sys.path.append("..")

import os
import json
from preprocess import count
import detector
from classifier import Classifier
from collections import defaultdict

from helper import detector_helper as helper


def get_type_info(type_info):
    type_info = re.sub("_[a-zA-Z]*Characteristics", "", type_info)
    type_info = type_info.replace("_cancelActivity", "")

    if type_info in ["boundaryEvent_conditional", "boundaryEvent_message", "boundaryEvent_signal",
                     "boundaryEvent_timer"]:
        type_info = type_info.replace("boundaryEvent", "intermediateCatchEvent")

    if type_info == "startEvent_message_isInterrupting":
        type_info = "startEvent_message"

    if type_info == "intermediateThrowEvent_timer":
        type_info = "intermediateCatchEvent_timer"

    if type_info == "intermediateCatchEvent":
        type_info = "intermediateThrowEvent"
    return type_info


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


def validate_one(json_file, image_file):
    def match_ele_and_shape(e_index, e_rec, e_type, shapes_list, shapes_label, fake_elements, image_ele_id_map):
        ele_area = e_rec[2] * e_rec[3]

        matched = False
        for shape_id in shapes_list:
            # [element_type, shape_bound, element_id]
            ele_label = shapes_label[shape_id]
            shape_rec = ele_label[1]
            shape_area = shape_rec[2] * shape_rec[3]
            overlap_area = helper.get_overlap_area(e_rec, shape_rec)
            if overlap_area / ele_area > 0.5 and overlap_area / shape_area > 0.5:
                # print("here")
                shapes_list.remove(shape_id)
                image_ele_id_map[e_index] = ele_label[-1]

                ele_shape_type = get_type_info(ele_label[0])
                # print(ele_shape_type)
                matched = True
                # detected_num, type_right, type_wrong
                shape_type_res = shapes_result[ele_shape_type].get("detect", [0, 0, 0])
                shape_type_res[0] += 1
                if e_type == ele_shape_type:
                    shape_type_res[1] += 1
                else:
                    if e_type == "exclusiveGateway_fork" and ele_shape_type == "exclusiveGateway":
                        shape_type_res[1] += 1
                    else:
                        shape_type_res[2] += 1
                shapes_result[ele_shape_type]["detect"] = shape_type_res
                break

        if not matched:
            image_ele_id_map[e_index] = ""
            fake_elements.append(e_index)
        return shapes_list

    _, all_elements_info, all_seq_flows, all_elements, pools = detector.detect(image_file, classifier, classifier_type)

    shapes_result = defaultdict(dict)
    fake_shapes = []
    fake_lanes = []
    fake_pools = []

    with open(json_file, encoding="utf-8", mode='r') as f:
        project_labels = json.load(f)

    pool_labels = project_labels["pool_labels"]
    shape_labels = project_labels["shape_labels"]
    flow_labels = project_labels["flow_labels"]

    base_point = None
    if img_type == "png":
        base_point = [6, 6]
    elif img_type == "jpg":
        base_point = [30, 30]
    else:
        print("invalid img_type")
        exit(0)

    x_min = project_labels["x_min"]
    y_min = project_labels["y_min"]
    x_offset = base_point[0] - x_min
    y_offset = base_point[1] - y_min

    flow_shapes = []
    sub_p_shapes = []
    lane_shapes = []
    pool_shapes = []
    # [element_type, shape_bound, element_id]
    for shape_index, shape_label in enumerate(shape_labels):
        shape_type = get_type_info(shape_label[0])
        shape_rect = shape_label[1]
        shape_rect[0] += x_offset
        shape_rect[1] += y_offset
        shape_label[1] = shape_rect

        if shape_type in labels:
            type_num = shapes_result[shape_type].get("total", 0)
            type_num += 1
            shapes_result[shape_type]["total"] = type_num
            flow_shapes.append(shape_index)
        else:
            if "expanded" in shape_type.split("_"):
                shape_labels[shape_index][0] = "subProcess_expanded"
                type_num = shapes_result["subProcess_expanded"].get("total", 0)
                type_num += 1
                shapes_result["subProcess_expanded"]["total"] = type_num
                sub_p_shapes.append(shape_index)

            if shape_type == "lane":
                type_num = shapes_result["lane"].get("total", 0)
                type_num += 1
                shapes_result["lane"]["total"] = type_num
                lane_shapes.append(shape_index)

    for pool_index, pool_label in enumerate(pool_labels):
        pool_rect = pool_label[1]
        pool_rect[0] += x_offset
        pool_rect[1] += y_offset
        pool_label[1] = pool_rect

        type_num = shapes_result["pool"].get("total", 0)
        type_num += 1
        shapes_result["pool"]["total"] = type_num
        pool_shapes.append(pool_index)

    # {detected element id: bpmn file element id}
    image_shape_id_map = dict()
    image_pool_id_map = dict()
    image_lane_id_map = dict()

    for ele_index, ele_path in enumerate(all_elements):
        ele_rec = get_element_rec_by_path(ele_path, pools)
        ele_type = all_elements_info[ele_index][0]

        if ele_path[-1] == 0:
            # flow elements
            flow_shapes = match_ele_and_shape(ele_index, ele_rec, ele_type, flow_shapes, shape_labels, fake_shapes,
                                              image_shape_id_map)
        else:
            # expanded sub processes
            sub_p_shapes = match_ele_and_shape(ele_index, ele_rec, ele_type, sub_p_shapes, shape_labels, fake_shapes,
                                               image_shape_id_map)

    if len(pool_labels) > 0:
        for pool_id, pool in enumerate(pools):
            pool_rect = pool["rect"]

            pool_shapes = match_ele_and_shape(pool_id, pool_rect, "pool", pool_shapes, pool_labels, fake_pools,
                                              image_pool_id_map)

            lanes = pool["lanes"]
            for lane_id, lane in enumerate(lanes):
                lane_shapes = match_ele_and_shape((pool_id, lane_id), lane, "lane", lane_shapes, shape_labels,
                                                  fake_lanes, image_lane_id_map)

    # [file seq num, not matched, target match, source match, others]
    seq_result = [0, 0, 0, 0, 0]

    flows_label_rest = []

    for flow_id in range(len(flow_labels)):
        if flow_labels[flow_id][1] == "sequenceFlow":
            flows_label_rest.append(flow_id)

    seq_result[0] = len(flows_label_rest)

    for seq_flow in all_seq_flows:
        target_ele_id = seq_flow[0]
        source_ele_id = seq_flow[-1]
        try:
            target_ele_ref = image_shape_id_map[target_ele_id]
            source_ele_ref = image_shape_id_map[source_ele_id]
        except KeyError:
            seq_result[-1] += 1
            continue
        # try:
        #     source_ele_ref = image_ele_id_map[source_ele_id]
        # except KeyError:
        #     seq_result[-1] += 1
        #     continue

        same_target_seqs = []
        for flow_id in flows_label_rest:
            # [file_id, main_type, points_label, element_id, source_ref, target_ref]
            flow_label = flow_labels[flow_id]
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
    interested_labels.extend(["lane", "pool", "subProcess_expanded"])

    one_res = {}
    file_name = image_file.split("/")[-1]
    file_label = file_name.split(".")[0]
    one_res["file_label"] = file_label
    # print(image_file)
    one_res["shapes"] = {}
    for label in interested_labels:
        label_result = shapes_result[label]
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
            # print("{}\t{},{},{},{}".format(label, total, detect[0], detect[1], detect[2]))
            one_res["shapes"][label] = [total, detect[0], detect[1], detect[2]]

    # print("fake_elements\t{}".format(len(fake_elements)))
    one_res["fake_shapes"] = len(fake_shapes)
    one_res["fake_pools"] = len(fake_pools)
    one_res["fake_lanes"] = len(fake_lanes)

    seq_record = "sequenceFlow\t{},{},{},{},{}".format(seq_result[0], seq_result[1],
                                                       seq_result[2], seq_result[3], seq_result[4])
    one_res["seq_record"] = seq_record
    # print(seq_record)

    # print("=" * 100)
    return one_res


def validate(data_dir):
    print("validate {}".format(classifier_type))
    validate_res_dir = "validate_results_2/"

    projects = os.listdir(validate_data_dir)
    projects.sort()

    results = []
    for i in range(begin, end):
        project = projects[i]
        if project in invalid_projects:
            continue
        print("----------------{}, {}----------------".format(i, project))
        project_dir = "{}{}".format(validate_data_dir, project)
        files = os.listdir(project_dir)
        png_file = ""
        jpg_file = ""
        json_file = ""
        for file in files:
            if file.endswith("png"):
                png_file = "{}/{}".format(project_dir, file)
            if file.endswith("jpeg"):
                jpg_file = "{}/{}".format(project_dir, file)
            if file.endswith("json"):
                json_file = "{}/{}".format(project_dir, file)

        if img_type == "png":
            image_file = png_file
        else:
            image_file = jpg_file

        try:
            _, all_elements_info, all_seq_flows, all_elements, pools, time_recorder = detector.detect(image_file, classifier,
                                                                                       classifier_type)
        except TypeError:
            print("invalid!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  TypeError")
            continue
        except IndexError:
            print("invalid!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  IndexError")
            continue
        for seq_flow in all_seq_flows:
            points = seq_flow[1]
            for p_id in range(len(points)):
                points[p_id] = [float(x) for x in points[p_id]]
        # print(all_seq_flows)
        detect_result = {"all_elements_info": all_elements_info,
                         "all_seq_flows": all_seq_flows,
                         "all_elements": all_elements,
                         "pools": pools,
                         "time_recorder": time_recorder}

        result_root_dir = "detect_results_all_gpu"
        image_type_result_dir = "{}/{}".format(result_root_dir, img_type)

        project_result_dir = "{}/{}_{}".format(image_type_result_dir, project, i)

        if not os.path.exists(project_result_dir):
            os.makedirs(project_result_dir, exist_ok=True)

        result_file = "{}/{}_detect.json".format(project_result_dir, project)
        # temp = json.dumps(detect_result)
        # print(result_file)
        with open(result_file, encoding="utf-8", mode="w") as f:
            json.dump(detect_result, f)

        # try:
        #     one_res = validate_one(json_file, image_file)
        # except TypeError:
        #     with open("validate_invalid_list.txt", "a+") as f:
        #         f.write("{}\t{}:{}\n".format("TypeError", i, image_file))
        #     continue
        # except IndexError:
        #     with open("validate_invalid_list.txt", "a+") as f:
        #         f.write("{}\t{}:{}\n".format("IndexError", i, image_file))
        #     continue
        # except KeyError:
        #     with open("validate_invalid_list.txt", "a+") as f:
        #         f.write("{}\t{}:{}\n".format("KeyError", i, image_file))
        #     continue
        # results.append(one_res)

    # if not os.path.exists(validate_res_dir):
    #     os.mkdir(validate_res_dir)
    #
    # data_dir_name = data_dir.split("/")[-2]
    #
    # file_id = len(os.listdir(validate_res_dir))
    # file_name = "{}_{}_{}_res.json".format(file_id, classifier_type, data_dir_name)
    # file_path = validate_res_dir + file_name
    #
    # with open(file_path, "w") as f:
    #     json.dump(results, f)


if __name__ == '__main__':

    opt = sys.argv[1]
    img_type = sys.argv[2]
    begin = int(sys.argv[3])
    end = int(sys.argv[4])
    # opt = "bcf"

    # invalid_projects = ["00142_00", "00144_00", "00170_01", "00205_03", "00222_00", "00227_00", "00473_00", "00570_00",
    #                     "00750_00", "00870_00", "00878_00", "00978_00", "01117_00", "01119_00", "01178_00", "01357_00",
    #                     "01381_00", "01595_00", "01608_00", "01634_00", "01718_00", "01811_00", "01838_00", "01847_00",
    #                     "01986_00", "02693_02", "02833_00", "02901_00", "03223_00", "03819_00", "03979_00", "04121_00",
    #                     "04136_00", "04303_00", "04378_00", "04410_03", "04429_00", "04544_01", "04618_00", "04621_00",
    #                     "04909_00", "04939_01", "04988_00", "05080_00", "05229_00", "05613_00", "05888_00", "05912_00",
    #                     "05968_00", "06126_00", "06169_00", "06298_01", "06874_00", "07198_00", "07290_00", "07603_00",
    #                     "07832_00", "08510_00", "08972_00", "09195_00", "09232_00", "09393_00", "09557_04", "09652_01"]

    # """
    # ["00750_00","01357_00","01634_00","01811_00","01912_00","02215_00","02216_00","02693_02","03819_00","04542_00",
    # "04618_00","04647_01","04666_01","04769_00","04909_00","06585_00","07901_00","09557_04"]
    # """
    # png_invalid = ["01912_00", "02215_00", "02216_00", "04542_00", "04647_01", "04666_01", "04769_00", "06585_00",
    #                "07901_00"]

    invalid_projects = ["00750_00", "01357_00", "01634_00", "01811_00", "01912_00", "02215_00", "02216_00", "02693_02",
                        "03819_00", "04542_00", "04618_00", "04647_01", "04666_01", "04769_00", "04909_00", "06585_00",
                        "07901_00", "09557_04"]

    validate_data_dir = "../merge_info_validate/"
    # classifier_types = ["bcf", "bcf_56", "bcf_57", "vgg16", "vgg16_56", "vgg16_57"]
    if opt == "vgg":
        print("validate vgg")
        # validate(validate_data_dir, "vgg16")
        # validate(validate_data_dir, "vgg16_56")
        classifier = Classifier()
        classifier_type = "vgg16_52"
        classifier.img_type = img_type
        classifier_id = classifier.classifiers.index(classifier_type)
        labels = classifier.classes[classifier_id].copy()
        validate(validate_data_dir)
    elif opt == "bcf":
        print("validate bcf")
        validate(validate_data_dir)
        validate(validate_data_dir)
        validate(validate_data_dir)
