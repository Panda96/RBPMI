import os
import json
import re
from collections import defaultdict
import numpy as np
import helper.detector_helper as helper


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

    if "expanded" in type_info:
        type_info = "subprocess_expanded"

    if type_info in ["process", "participant"]:
        type_info = "pool"

    return type_info


def get_project_info(parent_dir, project):
    project_dir = "{}/{}".format(parent_dir, project)
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

    with open(json_file, mode="r", encoding="utf-8") as f:
        project_labels = json.load(f)

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

    project_info = dict()
    project_info["png_file"] = png_file
    project_info["jpg_file"] = jpg_file
    project_info["project_labels"] = project_labels
    project_info["x_offset"] = x_offset
    project_info["y_offset"] = y_offset

    return project_info


def get_element_rec_by_path(path, pools):
    pool = pools[path[0]]
    # print(path)
    if path[3] == 0:
        elements = pool.get("elements", defaultdict(list))
        elements_i = elements.get(str(path[1]), [])
        rect = elements_i[path[2]]
    else:
        sub_procs = pool["sub_procs"]
        sub_procs_in_lane = sub_procs.get(str(path[1]))
        rect = sub_procs_in_lane[path[2]]
    return rect


def get_infer_info(parent_dir, project):
    infer_info = dict()
    if model_id == "my_detector":
        infer_projects = os.listdir(parent_dir)
        real_project = ""
        for each in infer_projects:
            if each.startswith(project):
                real_project = each
                break
        infer_dir = "{}/{}".format(parent_dir, real_project)
        infer_file = "{}/{}_detect.json".format(infer_dir, project)
        with open(infer_file, encoding="utf-8", mode="r") as f:
            detect_res = json.load(f)
            pools = detect_res["pools"]
            all_elements = detect_res["all_elements"]
            all_seq_flows = detect_res["all_seq_flows"]
            all_elements_info = detect_res["all_elements_info"]

            # print(pools)

            shapes_info = []
            for ele_id, ele_path in enumerate(all_elements):
                # print(ele_path)
                ele_type = all_elements_info[ele_id][0]
                ele_rec = get_element_rec_by_path(ele_path, pools)
                shapes_info.append([ele_type, ele_rec])
            for pool in pools:
                shapes_info.append(["pool", pool["rect"]])
                lanes = pool["lanes"]
                for lane in lanes:
                    shapes_info.append(["lane", lane])
        infer_info["shapes_info"] = shapes_info
        infer_info["all_seq_flows"] = all_seq_flows
    else:
        infer_dir = "{}/{}".format(parent_dir, project)
        infer_file = "{}/{}_infer.json".format(infer_dir, project)
        with open(infer_file, encoding="utf-8", mode="r") as f:
            shapes_info = json.load(f)
        infer_info["shapes_info"] = shapes_info

    return infer_info


def find_one_match(shape_rect, infer_info, match_mask):
    possible_matched = []
    for info_id, info in enumerate(infer_info):
        if match_mask[info_id] == 1:
            continue
        infer_rect = info[1]
        shape_area = helper.get_rect_area(shape_rect)
        infer_area = helper.get_rect_area(infer_rect)
        overlap_area = helper.get_overlap_area(shape_rect, infer_rect)

        shape_ratio = overlap_area / shape_area
        infer_ratio = overlap_area / infer_area

        if shape_ratio > 0.5 and infer_ratio > 0.5:
            shape_center = helper.get_rec_center(shape_rect)
            infer_center = helper.get_rec_center(infer_rect)
            x_offset = infer_center[0] - shape_center[0]
            y_offset = infer_center[1] - shape_center[1]
            width_diff = infer_rect[2] - shape_rect[2]
            height_diff = infer_rect[3] - shape_rect[3]
            possible_matched.append(
                [info_id, [shape_ratio, infer_ratio], [x_offset, y_offset, width_diff, height_diff]])

    if len(possible_matched) == 0:
        return -1, None
    else:
        possible_matched.sort(key=lambda x: (x[1][0], x[1][1]))
        matched_id = possible_matched[0][0]
        match_mask[matched_id] = 1
        return matched_id, possible_matched[0]


def validate_one(project):
    project_info = get_project_info(validate_dir, project)
    project_labels = project_info["project_labels"]
    shape_labels = project_labels["shape_labels"]
    pool_labels = project_labels["pool_labels"]
    x_offset = project_info["x_offset"]
    y_offset = project_info["y_offset"]

    info_info = get_infer_info(infer_result_dir, project)
    shapes_infer_info = info_info["shapes_info"]

    match_mask = [0] * len(shapes_infer_info)
    match_map = dict()

    # [total_num, detected_num, type_right, type_wrong]
    shapes_result = [[0, 0, 0, 0] for _ in range(len(categories))]
    # not detected shape ids
    not_detected = []

    shape_labels.extend(pool_labels)

    # if project == "00049_04":
    #     print("shape_labels")
    #     for shape_label in shape_labels:
    #         print(shape_label)
    #     print("shape_infers")
    #     for shape_infer in shapes_infer_info:
    #         print(shape_infer)

    for shape_id, shape_label in enumerate(shape_labels):
        shape_type = get_type_info(shape_label[0])
        shape_label[0] = shape_type
        shape_rect = shape_label[1]
        shape_rect[0] += x_offset
        shape_rect[1] += y_offset

        if shape_type not in categories:
            # print(shape_type)
            continue

        one_shape_result = shapes_result[categories.index(shape_type)]
        one_shape_result[0] += 1

        matched_id, match_info = find_one_match(shape_rect, shapes_infer_info, match_mask)
        if matched_id < 0:
            not_detected.append(shape_id)
        else:
            one_shape_result[1] += 1
            if shape_type.lower() == shapes_infer_info[matched_id][0].lower():
                one_shape_result[2] += 1
            else:
                one_shape_result[3] += 1
            match_map[shape_id] = match_info
    # print(match_mask)
    # print(not_detected)
    # for i in range(len(match_mask)):
    #     if match_mask[i] == 0:
    #         print(infer_info[i])
    one_res = {"all_shape_labels": shape_labels,
               "infer_info": shapes_infer_info,
               "shapes_result": shapes_result,
               "match_mask": match_mask,
               "match_map": match_map,
               "project": project}

    validate_result_dir = "{}/{}".format(validate_result_root_dir, project)
    if not os.path.exists(validate_result_dir):
        os.makedirs(validate_result_dir, exist_ok=True)

    result_file = "{}/validate.json".format(validate_result_dir)
    with open(result_file, mode="w", encoding="utf-8") as f:
        json.dump(one_res, f)


def validate():
    projects = os.listdir(validate_dir)
    projects.sort()
    for project in projects:
        if model_id == "my_detector" and project in invalid_projects:
            continue
        print(project)
        validate_one(project)


def get_validate_info(parent_dir, project):
    project_dir = "{}/{}".format(parent_dir, project)
    validate_result = "{}/validate.json".format(project_dir)
    with open(validate_result, mode="r", encoding="utf-8") as f:
        one_res = json.load(f)

    return one_res


def evaluate_one(project):
    one_val_res = get_validate_info(validate_result_root_dir, project)
    all_shape_labels = one_val_res["all_shape_labels"]
    shapes_infer_info = one_val_res["infer_info"]
    shapes_result = one_val_res["shapes_result"]
    match_map = one_val_res["match_map"]
    match_mask = one_val_res["match_mask"]

    shape_division = defaultdict(list)

    for shape_id, shape_label in enumerate(all_shape_labels):
        shape_division[shape_label[0]].append(shape_id)

    infer_division = defaultdict(list)

    for infer_id, infer in enumerate(shapes_infer_info):
        infer_division[infer[0]].append(infer_id)

    one_evaluate_res = dict()
    all_located = 0
    all_correct = 0
    all_match_infos = []
    all_num = 0
    all_infered = 0
    for cate_id, cate in enumerate(categories):
        [cate_num, cate_located, cate_correct, cate_error] = shapes_result[cate_id]
        cate_infered = len(infer_division[cate])
        if cate not in ["pool", "lane", "subprocess_expanded"]:
            all_num += cate_num
            all_infered += cate_infered
        if cate_num == 0:
            cate_locate_acc_rate = None
            cate_locate_recall_rate = None
            cate_classify_acc_rate = None
            x_offset = None
            y_offset = None
            width_offset = None
            height_offset = None
        else:
            # print(cate)
            # print(cate_num)
            if cate not in ["pool", "lane", "subprocess_expanded"]:
                all_located += cate_located
                all_correct += cate_correct
            cate_locate_acc_rate = cate_located / cate_num

            if cate_infered > 0:
                cate_locate_recall_rate = cate_located / cate_infered
            else:
                cate_locate_recall_rate = 0

            if cate_located > 0:
                cate_classify_acc_rate = cate_correct / cate_located
            else:
                cate_classify_acc_rate = 0

            cate_shapes = shape_division[cate]

            cate_match_infos = []
            for shape_id in cate_shapes:
                if str(shape_id) in match_map.keys():
                    cate_match_infos.append(match_map[str(shape_id)][-1])
            if len(cate_match_infos) > 0:
                all_match_infos.extend(cate_match_infos)

                cate_diff = [0] * 4
                for one in range(4):
                    cate_diff[one] = np.mean(np.array(cate_match_infos)[:, one])
            else:
                cate_diff = [None, None, None, None]
            [x_offset, y_offset, width_offset, height_offset] = cate_diff

        one_evaluate_res[cate] = [cate_locate_acc_rate, cate_locate_recall_rate, cate_classify_acc_rate, x_offset,
                                  y_offset,
                                  width_offset, height_offset]

    if all_num == 0:
        all_res = [None, None, None, None, None, None, None]
    else:
        all_locate_acc_rate = all_located / all_num
        if all_infered == 0:
            all_locate_recall_rate = 0
        else:
            all_locate_recall_rate = all_located / all_infered

        if all_located == 0:
            all_classify_acc_rate = 0
        else:
            all_classify_acc_rate = all_correct / all_located

        all_res = [all_locate_acc_rate, all_locate_recall_rate, all_classify_acc_rate]
        if len(all_match_infos) > 0:
            all_diff = [0] * 4
            for one in range(4):
                all_diff[one] = np.mean(np.array(all_match_infos)[:, one])

        else:
            all_diff = [None, None, None, None]
        all_res.extend(all_diff)
    one_evaluate_res[all] = all_res
    one_evaluate_res["num"] = all_num


def evaluate():
    projects = os.listdir(validate_result_root_dir)
    projects.sort()
    for project in projects:
        print(project)
        evaluate_one(project)


def main():
    # print("validating")
    # validate()
    print("evaluating")
    evaluate()


if __name__ == '__main__':
    validate_dir = "../merge_info_validate"
    img_type = "png"
    model_id = "ssd_resnet_03"
    infer_result_dir = "infer_results/{}/{}".format(img_type, model_id)
    # detect_result_file = "detect_results/{}/{}/validate.json".format(img_type, model_id)
    validate_result_root_dir = infer_result_dir.replace("infer_results", "detect_results")
    categories = ['task', 'sendTask', 'userTask', 'manualTask', 'scriptTask', 'receiveTask', 'serviceTask',
                  'businessRuleTask', 'endEvent', 'endEvent_cancel', 'endEvent_terminate', 'endEvent_signal',
                  'endEvent_link', 'endEvent_compensate', 'endEvent_error', 'endEvent_message', 'endEvent_escalation',
                  'startEvent', 'startEvent_compensate', 'startEvent_conditional', 'startEvent_signal',
                  'startEvent_message', 'startEvent_isInterrupting', 'startEvent_error', 'startEvent_escalation',
                  'startEvent_timer', 'boundaryEvent_error', 'boundaryEvent_compensate', 'boundaryEvent_escalation',
                  'boundaryEvent_cancel', 'intermediateCatchEvent_signal', 'intermediateCatchEvent_conditional',
                  'intermediateCatchEvent_link', 'intermediateCatchEvent_message', 'intermediateCatchEvent_timer',
                  'intermediateThrowEvent', 'intermediateThrowEvent_link', 'intermediateThrowEvent_message',
                  'intermediateThrowEvent_signal', 'intermediateThrowEvent_compensate',
                  'intermediateThrowEvent_escalation', 'subProcess', 'adHocSubProcess', 'transaction', 'callActivity',
                  'dataStoreReference', 'dataObjectReference', 'complexGateway', 'parallelGateway', 'exclusiveGateway',
                  'inclusiveGateway', 'eventBasedGateway', 'pool', 'lane', "subprocess_expanded"]

    invalid_projects = ["00750_00", "01357_00", "01634_00", "01811_00", "01912_00", "02215_00", "02216_00", "02693_02",
                        "03819_00", "04542_00", "04618_00", "04647_01", "04666_01", "04769_00", "04909_00", "06585_00",
                        "07901_00", "09557_04"]
    main()
