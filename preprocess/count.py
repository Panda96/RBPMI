# -*- coding:utf-8 -*-
import os
import xml.etree.ElementTree as eTree
import shutil
from collections import defaultdict


def get_tag_type(tag):
    return tag.split("}")[1]


def is_overlap(rec1, rec2):
    return rec1[1] + rec1[3] > rec2[1] and rec1[1] < rec2[1] + rec2[3] \
           and rec1[0] + rec1[2] > rec2[0] and rec1[0] < rec2[0] + rec2[2]


# rec2 is in rec1
def is_in(rec1, rec2):
    return rec1[0] <= rec2[0] and rec1[1] <= rec2[1] \
           and rec1[0] + rec1[2] >= rec2[0] + rec2[2] and rec1[1] + rec1[3] >= rec2[1] + rec2[3]


def point_is_in(rec, point):
    return is_in(rec, [point[0], point[1], 0, 0])


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
                    if point_is_in(participant, points_label[0]) and not point_is_in(participant, points_label[-1]):
                        print(file_id, "crossed")
                        return True
    return False


# def get_element_type(node, file_id, definitions):
#     element_id = node.attrib.get("bpmnElement", "")
#     elements = definitions.findall(".//*[@id='{}']".format(element_id))
#     element = elements[0]
#     element_type = get_tag_type(element.tag)


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
                if is_overlap(bound, shape_bound):
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

    min_x = 10000
    min_y = 10000

    try:
        for node in plane:
            element_id = node.attrib.get("bpmnElement", "")
            if element_id == "":
                print(file_id)
                print("invalid shape")
                continue
            try:
                elements = definitions.findall(".//*[@id='{}']".format(element_id))
            except SyntaxError:
                print(file_id)
                print(".//*[@id='{}']".format(element_id))
                print("SyntaxError")
                continue
            if len(elements) == 0:
                print(file_id)
                print("no corresponding element")
                continue
            element = elements[0]

            main_type = get_tag_type(element.tag)
            type_info = [main_type]
            if main_type.endswith("Event"):
                for sub_node in element:
                    element_type_info = get_tag_type(sub_node.tag)
                    if element_type_info.endswith("Definition"):
                        type_info.append(element_type_info)

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

                if shape_bound[0] < min_x:
                    min_x = shape_bound[0]
                if shape_bound[1] < min_y:
                    min_y = shape_bound[1]

                shape_label = [file_id, element_type, shape_bound]
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

                flow_label = [file_id, element_type, points_label]
                flows_label.append(flow_label)

            labels = node.findall("{http://www.omg.org/spec/BPMN/20100524/DI}BPMNLabel")
            for label in labels:
                bounds = label.find("{http://www.omg.org/spec/DD/20100524/DC}Bounds")
                if bounds is not None:
                    label_bound = [bounds.attrib["x"], bounds.attrib["y"], bounds.attrib["width"],
                                   bounds.attrib["height"]]
                    label_bound = [int(float(x)) for x in label_bound]

                    if label_bound[0] < min_x:
                        min_x = label_bound[0]
                    if label_bound[1] < min_y:
                        min_y = label_bound[1]

                    text_label = [file_id, label_bound]
                    texts_label.append(text_label)

    except KeyError:
        print(file_name, " key error")
        return

    offset_x = 6 - min_x
    offset_y = 6 - min_y

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
    with open("shapes.txt", "w") as shape:
        for shape_label in all_shapes_label:
            shape_record = "{};{};{}".format(shape_label[0], shape_label[1], " ".join([str(x) for x in shape_label[2]]))
            shape.write(shape_record + "\n")

    with open("flows.txt", "w") as flow:
        for flow_label in all_flows_label:
            points = flow_label[2]
            points_label = " ".join([",".join([str(x) for x in point]) for point in points])
            flow_record = "{};{};{}".format(flow_label[0], flow_label[1], points_label)
            flow.write(flow_record + "\n")

    with open("text.txt", "w") as texts:
        for text_label in all_texts_label:
            text_record = "{};{}".format(text_label[0], " ".join([str(x) for x in text_label[1]]))
            texts.write(text_record + "\n")


def count():
    # file_dir = "E:/diagrams/bpmn-io/bpmn2image/data0423/files/"
    file_dir = "E:/diagrams/bpmn-io/bpmn2image/data0423/data700_3/bpmn/"
    files = os.listdir(file_dir)
    for f in files:
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
    sorted_shape_type_list = sorted(shape_type_list, key=lambda x: x.split("_")[0][-2:])
    for shape_type in sorted_shape_type_list:
        print("{}:{}".format(shape_type, len(shape_type_dict[shape_type])))


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
    # judge_crossed()
    # pass

    count()
    statistic()
    # output()

    # all_type()
