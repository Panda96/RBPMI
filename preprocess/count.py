# -*- coding:utf-8 -*-
import os
import xml.etree.ElementTree as eTree
import shutil
from collections import defaultdict
from helper import detector_helper as rh


def get_tag_type(tag):
    return tag.split("}")[1]


def judge_flow_cross_pools(file):
    file_name = file.split("/")[-1]
    file_id = "_".join(file_name.split("_")[0:2])
    dom_tree = eTree.parse(file)

    definitions = dom_tree.getroot()
    res = definitions.findall("./{http://www.omg.org/spec/BPMN/20100524/DI}BPMNDiagram")
    diagram = res[0]
    plane = diagram.findall("{http://www.omg.org/spec/BPMN/20100524/DI}BPMNPlane")[0]

    participants = []

    for node in plane:
        element_id = node.attrib.get("bpmnElement", "")
        elements = definitions.findall(".//*[@id='{}']".format(element_id))
        element = elements[0]
        element_type = get_tag_type(element.tag)

        if element_type == "participant":
            bounds = node.find("{http://www.omg.org/spec/DD/20100524/DC}Bounds")
            shape_bound = [bounds.attrib["x"], bounds.attrib["y"], bounds.attrib["width"], bounds.attrib["height"]]
            shape_bound = [int(float(x)) for x in shape_bound]
            participants.append(shape_bound)

    # print(participants)
    # print("-"*100)

    if len(participants) > 0:
        for node in plane:
            element_id = node.attrib.get("bpmnElement", "")
            elements = definitions.findall(".//*[@id='{}']".format(element_id))
            element = elements[0]
            element_type = get_tag_type(element.tag)

            if element_type == "sequenceFlow":
                points = node.findall("{http://www.omg.org/spec/DD/20100524/DI}waypoint")
                points_label = []
                for point in points:
                    point_label = [int(float(point.attrib["x"])), int(float(point.attrib["y"]))]
                    points_label.append(point_label)

                # print(points_label)
                # print("=" * 100)
                for participant in participants:
                    if rh.point_is_in(participant, points_label[0]) and \
                            not rh.point_is_in(participant, points_label[-1]):
                        print(file_id, "crossed")
                        return True
    return False


def judge_one_overlapped(file):
    file_name = file.split("/")[-1]
    file_id = "_".join(file_name.split("_")[0:2])
    dom_tree = eTree.parse(file)

    definitions = dom_tree.getroot()
    res = definitions.findall("./{http://www.omg.org/spec/BPMN/20100524/DI}BPMNDiagram")
    diagram = res[0]
    plane = diagram.findall("{http://www.omg.org/spec/BPMN/20100524/DI}BPMNPlane")[0]

    participants = []

    for node in plane:
        not_blank = True
        element_id = node.attrib.get("bpmnElement", "")
        elements = definitions.findall(".//*[@id='{}']".format(element_id))
        element = elements[0]
        element_type = get_tag_type(element.tag)

        if element_type == "participant":
            bounds = node.find("{http://www.omg.org/spec/DD/20100524/DC}Bounds")
            if bounds is None:
                print(file_id)
                print("shape no bounds")
                continue
            shape_bound = [bounds.attrib["x"], bounds.attrib["y"], bounds.attrib["width"], bounds.attrib["height"]]
            shape_bound = [int(float(x)) for x in shape_bound]

            for bound in participants:
                if rh.is_overlap(bound, shape_bound):
                    print(file_id)
                    return True
            participants.append(shape_bound)
    return False


def count_one_bpmn(file):
    shapes_label = []
    flows_label = []
    texts_label = []
    file_name = file.split("/")[-1]
    file_id = "_".join(file_name.split("_")[0:2])
    # print(file_id)
    try:
        dom_tree = eTree.parse(file)
    except eTree.ParseError:
        print(file_name)
        return

    definitions = dom_tree.getroot()
    res = definitions.findall("./{http://www.omg.org/spec/BPMN/20100524/DI}BPMNDiagram")

    if len(res) != 1:
        print(file_name, " more than one diagram")
        exit(0)

    diagram = res[0]
    plane = diagram.findall("{http://www.omg.org/spec/BPMN/20100524/DI}BPMNPlane")[0]

    bound_dic = defaultdict(list)

    # blank = True
    # for node in plane:
    #     blank = False
    #     # 如果是未筛选过的数据需要做一些异常处理
    #     element_id = node.attrib.get("bpmnElement", "")
    #     elements = definitions.findall(".//*[@id='{}']".format(element_id))
    #     element = elements[0]
    #     main_type = get_tag_type(element.tag)
    #
    #     if main_type == "boundaryEvent":
    #         # 未筛选过的数据需要做异常处理
    #         task_id = element.attrib.get("attachedToRef", "")
    #         node_type = get_tag_type(node.tag)
    #         if node_type == "BPMNShape":
    #             bounds = node.find("{http://www.omg.org/spec/DD/20100524/DC}Bounds")
    #             shape_bound = [bounds.attrib["x"], bounds.attrib["y"], bounds.attrib["width"], bounds.attrib["height"]]
    #             shape_bound = [int(float(x)) for x in shape_bound]
    #             bound_dic[task_id].append(shape_bound)

    # if blank:
    #     print(file_id, "blank")
    #     return

    min_x = float("inf")
    min_y = float("inf")

    for node in plane:
        # 如果是未筛选过的数据需要做一些异常处理
        element_id = node.attrib.get("bpmnElement", "")
        elements = definitions.findall(".//*[@id='{}']".format(element_id))
        element = elements[0]

        main_type = get_tag_type(element.tag)

        # main_type = main_type.replace("intermediate", "inter")
        # main_type = main_type.replace("boundary", "bound")
        # main_type = main_type.replace("Reference", "")
        # main_type = main_type.replace("business", "busi")
        # main_type = main_type.replace("SubProcess", "Sub")
        type_info = [main_type]

        for sub_node in element:
            element_type_info = get_tag_type(sub_node.tag)
            if main_type.endswith("Event"):
                if element_type_info.endswith("Definition"):
                    info = element_type_info.replace("EventDefinition", "")
                    type_info.append(info)

                # if main_type == "boundaryEvent":
                #     type_info[0] = "intermediateCatchEvent"
            # else:
            #     if element_type_info.endswith("Characteristics"):
            #         if sub_node.attrib.get("isSequential", "") == "true":
            #             element_type_info = element_type_info + "_seq"
            #         # info = element_type_info.replace("multiInstance", "mulIns")
            #         # info = info.replace("standard", "std")
            #         # info = info.replace("oopCharacteristics", "")
            #         type_info.append(element_type_info)

        # if element_id in list(bound_dic.keys()):
        #     type_info.append("withBound")
        if main_type == "subProcess" and element.attrib.get("triggeredByEvent", "") == "true":
            type_info.append("triggeredByEvent")

        if main_type == "boundaryEvent" and element.attrib.get("cancelActivity", "") == "false":
            type_info.append("cancelActivity")

        if main_type == "startEvent" and element.attrib.get("isInterrupting", "") == "false":
            type_info.append("isInterrupting")

        if main_type in ["adHocSubProcess", "subProcess", "transaction"]:
            for sub_node in element:
                element_type_info = get_tag_type(sub_node.tag)
                lower_str = element_type_info.lower()
                if lower_str.endswith("task") or lower_str.endswith("event") or lower_str.endswith("gateway"):
                    # print(file_id)
                    # print(element_type_info)
                    type_info.append("expanded")
                    break

        element_type = "_".join(type_info)

        node_type = get_tag_type(node.tag)
        if node_type == "BPMNShape":
            bounds = node.find("{http://www.omg.org/spec/DD/20100524/DC}Bounds")
            if bounds is None:
                print(file_id)
                print("shape no bounds")
                continue
            shape_bound = [bounds.attrib["x"], bounds.attrib["y"], bounds.attrib["width"], bounds.attrib["height"]]
            shape_bound = [int(float(x)) for x in shape_bound]

            # task中的文字
            name = element.attrib.get("name", "")
            labels = node.findall("{http://www.omg.org/spec/BPMN/20100524/DI}BPMNLabel")
            process_type_list = ["participant", "lane"]
            if name != "" and len(labels) == 0 and main_type not in process_type_list:
                # 0 表示是shape的文字
                text_label = [file_id, [shape_bound[0], shape_bound[1], shape_bound[2], shape_bound[3]], element_id, 0]
                texts_label.append(text_label)

            if element_id in list(bound_dic.keys()):
                bound_recs = bound_dic[element_id]
                bound_recs.append(shape_bound)
                x_min = float("inf")
                y_min = float("inf")
                x_max = -1
                y_max = -1
                for rec in bound_recs:
                    x_min = min(rec[0], x_min)
                    y_min = min(rec[1], y_min)
                    x_max = max(rec[0] + rec[2], x_max)
                    y_max = max(rec[1] + rec[3], y_max)

                shape_bound = [x_min, y_min, x_max - x_min, y_max - y_min]

            if shape_bound[0] < min_x:
                min_x = shape_bound[0]
            if shape_bound[1] < min_y:
                min_y = shape_bound[1]

            shape_label = [file_id, element_type, shape_bound, element_id]
            shapes_label.append(shape_label)

        elif node_type == "BPMNEdge":
            points = node.findall("{http://www.omg.org/spec/DD/20100524/DI}waypoint")
            points_label = []
            for point in points:
                point_label = [int(float(point.attrib["x"])), int(float(point.attrib["y"]))]
                points_label.append(point_label)

                if point_label[0] < min_x:
                    min_x = point_label[0]
                if point_label[1] < min_y:
                    min_y = point_label[1]

            if main_type != "dataInputAssociation" and main_type != "dataOutputAssociation":
                source_ref = element.attrib.get("sourceRef", "")
                target_ref = element.attrib.get("targetRef", "")
                if source_ref != "" and target_ref != "":
                    flow_label = [file_id, main_type, points_label, element_id, source_ref, target_ref]
                else:
                    print(main_type + " " + element_id)
                    continue
            else:
                flow_label = [file_id, main_type, points_label, element_id, "", ""]
            flows_label.append(flow_label)

        name = element.attrib.get("name", "")
        labels = node.findall("{http://www.omg.org/spec/BPMN/20100524/DI}BPMNLabel")
        if name != "" and len(labels) > 0:
            label = labels[0]
            bounds = label.find("{http://www.omg.org/spec/DD/20100524/DC}Bounds")
            if bounds is not None:
                label_bound = [bounds.attrib["x"], bounds.attrib["y"], bounds.attrib["width"],
                               bounds.attrib["height"]]
                label_bound = [int(float(x)) for x in label_bound]

                if label_bound[0] < min_x:
                    min_x = label_bound[0]
                if label_bound[1] < min_y:
                    min_y = label_bound[1]
                if node_type == "BPMNEdge":
                    text_label = [file_id, label_bound, element_id, 1]
                else:
                    text_label = [file_id, label_bound, element_id, 0]
                texts_label.append(text_label)

    offset_x = 6 - min_x
    offset_y = 6 - min_y

    # print(min_x)
    # print(min_y)
    # print(offset_x)
    # print(offset_y)

    for shape_label in shapes_label:
        shape_label[2][0] += offset_x
        shape_label[2][1] += offset_y

    for text_label in texts_label:
        text_label[1][0] += offset_x
        text_label[1][1] += offset_y

    for flow_label in flows_label:
        for point in flow_label[2]:
            point[0] += offset_x
            point[1] += offset_y

    all_shapes_label.extend(shapes_label)
    all_texts_label.extend(texts_label)
    all_flows_label.extend(flows_label)


def output():
    # print("-"*100)
    with open("labels/shapes.txt", "w") as shape:
        for shape_label in all_shapes_label:
            # print(shape_label)
            shape_record = "{};{};{};{}".format(shape_label[0], shape_label[1],
                                                " ".join([str(x) for x in shape_label[2]]), shape_label[3])
            shape.write(shape_record + "\n")

    with open("labels/flows.txt", "w") as flow:
        for flow_label in all_flows_label:
            points = flow_label[2]
            points_label = " ".join([",".join([str(x) for x in point]) for point in points])
            flow_record = "{};{};{};{};{}".format(flow_label[0], flow_label[1], points_label,
                                                  flow_label[3], flow_label[4], flow_label[5])
            flow.write(flow_record + "\n")

    with open("labels/text.txt", "w") as texts:
        for text_label in all_texts_label:
            text_record = "{};{};{};{}".format(text_label[0], " ".join([str(x) for x in text_label[1]]),
                                               text_label[2], text_label[3])
            texts.write(text_record + "\n")


def count(file_dir):
    # file_dir = "E:/diagrams/bpmn-io/bpmn2image/data0423/data700_3/bpmn/"
    files = os.listdir(file_dir)
    for f in files:
        file_id = "_".join(f.split("_")[0:2])
        print(file_id)
        # if file_id == "157_08":
        file_path = file_dir + f
        count_one_bpmn(file_path)


def judge_crossed():
    file_dir = "E:/diagrams/bpmn-io/bpmn2image/data0423/files/"
    # file_dir = "E:/diagrams/bpmn-io/bpmn2image/data0423/data700_4/bpmn/"
    images_dir = "E:/diagrams/bpmn-io/bpmn2image/data0423/images/"
    files_crossed_dir = "E:/diagrams/bpmn-io/bpmn2image/data0423/files_crossed/"
    images_crossed_dir = "E:/diagrams/bpmn-io/bpmn2image/data0423/images_crossed/"

    files = os.listdir(file_dir)
    images = os.listdir(images_dir)
    files.sort()
    images.sort()

    for i in range(len(files)):
        f = files[i]
        file_path = file_dir + f
        crossed = judge_flow_cross_pools(file_path)
        if crossed:
            image_path = images_dir + images[i]
            shutil.move(file_path, files_crossed_dir)
            shutil.move(image_path, images_crossed_dir)


def judge_overlapped():
    file_dir = "E:/diagrams/bpmn-io/bpmn2image/data0423/files/"
    images_dir = "E:/diagrams/bpmn-io/bpmn2image/data0423/images/"
    files_overlapped_dir = "E:/diagrams/bpmn-io/bpmn2image/data0423/files_overlapped/"
    images_overlapped_dir = "E:/diagrams/bpmn-io/bpmn2image/data0423/images_overlapped/"

    files = os.listdir(file_dir)
    images = os.listdir(images_dir)
    files.sort()
    images.sort()

    for i in range(len(files)):
        f = files[i]
        file_path = file_dir + f
        overlapped = judge_one_overlapped(file_path)
        if overlapped:
            image_path = images_dir + images[i]
            shutil.move(file_path, files_overlapped_dir)
            shutil.move(image_path, images_overlapped_dir)


def statistic():
    for shape_id, shape_record in enumerate(all_shapes_label):
        shape_type_dict[shape_record[1]].append(shape_id)
    shape_type_list = list(shape_type_dict.keys())
    shape_type_list.sort()
    # sorted_shape_type_list = sorted(shape_type_list, key=lambda x: x.split("_")[0][-2:])
    type_dirs = []
    for shape_type in shape_type_list:
        type_dir_name = "{}:{}".format(shape_type, len(shape_type_dict[shape_type]))
        print(type_dir_name)
        type_dirs.append([shape_type, len(shape_type_dict[shape_type])])

    type_dirs.sort(key=lambda x: x[1])
    print(type_dirs)
    # print(len(type_dirs))

    # participants = shape_type_dict["participant"]
    # min_area = float("inf")
    # for participant in participants:
    #     label = all_shapes_label[participant]
    #     area = label[2][2] * label[2][3]
    #     if area < min_area:
    #         min_area = area
    #
    # print("min_area:{}".format(area))
    return type_dirs
    # sorted_shape_type_list = sorted(shape_type_list, key=lambda x: x.split("_")[0][-2:])
    # for shape_type in sorted_shape_type_list:
    #     print("{}:{}".format(shape_type, len(shape_type_dict[shape_type])))


def all_type():
    type_list = ['parallelGateway', 'subProcess', 'participant', 'group', 'manualTask', 'complexGateway', 'transaction',
                 'startEvent', 'endEvent', 'intermediateCatchEvent', 'exclusiveGateway', 'dataObjectReference',
                 'callActivity', 'dataStoreReference', 'task', 'scriptTask', 'receiveTask', 'lane', 'dataObject',
                 'sendTask', 'userTask', 'serviceTask', 'boundaryEvent', 'process', 'businessRuleTask',
                 'eventBasedGateway', 'inclusiveGateway', 'adHocSubProcess', 'textAnnotation', 'intermediateThrowEvent']
    sorted_list = sorted(type_list, key=lambda x: x[-2:])
    print(sorted_list)


all_shapes_label = []
all_flows_label = []
all_texts_label = []
shape_type_dict = defaultdict(list)

if __name__ == '__main__':
    files_dir = "E:/diagrams/bpmn-io/bpmn2image/data0423/files/"
    count(files_dir)
    #
    # for shape_label in all_shapes_label:
    #     print(shape_label)
    statistic()
    # output()
