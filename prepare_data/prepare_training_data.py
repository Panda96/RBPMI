import os
import cv2 as cv
import json
import re

import helper.detector_helper as helper

from collections import defaultdict


def complete_id(one_id, num, prefix="0"):
    one_id = str(one_id)
    while len(one_id) < num:
        one_id = prefix + one_id
    return one_id


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


def split_one(project, img_type):
    project_dir = "{}/{}".format(root_dir, project)
    files = os.listdir(project_dir)
    jpeg_file = ""
    png_file = ""
    json_file = ""
    for file in files:
        if file.endswith(".png"):
            png_file = "{}/{}".format(project_dir, file)
        if file.endswith(".jpeg"):
            jpeg_file = "{}/{}".format(project_dir, file)
        if file.endswith(".json"):
            json_file = "{}/{}".format(project_dir, file)
    # json_file = "{}/{}/{}.json".format(root_dir, project, project)

    with open(json_file, mode="r", encoding="utf-8") as f:
        project_labels = json.load(f)

    base_point = None
    img_file = ""
    if img_type == "png":
        base_point = [6, 6]
        img_file = png_file
    elif img_type == "jpeg":
        base_point = [30, 30]
        img_file = jpeg_file

    if base_point is None:
        print("invalid image format, we only support .png and .jpeg")
        return None

    shape_labels = project_labels["shape_labels"]
    x_min = project_labels["x_min"]
    y_min = project_labels["y_min"]
    x_offset = base_point[0] - x_min
    y_offset = base_point[1] - y_min

    input_img = cv.imread(img_file)

    for shape_label in shape_labels:
        type_info = shape_label[0]
        if type_info in ["lane", "group", "textAnnotation"]:
            continue
        if "expanded" in type_info:
            continue
        type_info = get_type_info(type_info)
        if type_info not in categories:
            continue

        shape_rect = shape_label[1]
        shape_rect[0] += x_offset
        shape_rect[1] += y_offset

        one_shape = helper.truncate(input_img, helper.dilate(shape_rect, 10))
        shape_dir = "{}/{}".format(target_dir, type_info)
        if not os.path.exists(shape_dir):
            os.mkdir(shape_dir)
        shapes = os.listdir(shape_dir)
        shape_id = len(shapes)
        shape_id = complete_id(shape_id, 6)
        shape_name = "{}_{}.png".format(project, shape_id)
        shape_path = "{}/{}".format(shape_dir, shape_name)
        cv.imwrite(shape_path, one_shape)


def split():
    projects = os.listdir(root_dir)
    projects.sort()
    image_type = "png"
    for project in projects:
        split_one(project, image_type)


def count():
    types = os.listdir(target_dir)

    # count_dict = defaultdict(int)
    print(len(types))
    for shape_type in types:
        type_dir = "{}/{}".format(target_dir, shape_type)
        shapes = os.listdir(type_dir)
        print("{} \t \t {}".format(shape_type, len(shapes)))


def main():
    # split()
    count()


if __name__ == '__main__':
    root_dir = "E:/master/data_1031/merge_info_use"
    target_dir = "E:/master/data_1031/shapes_png/all_elements"
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
                  'inclusiveGateway', 'eventBasedGateway']

    main()
