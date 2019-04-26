# -*- coding:utf-8 -*-
import xml.etree.ElementTree as et
import os
import shutil


def get_file_id(file_name):
    return "_".join(file_name.split("_")[0:2])


def filter_invalid():
    invalid_list = ["033_01", "033_14", "035_12", "045_10", "045_10", "056_11", "063_13", "131_17", "169_05", "179_01",
                    "179_01", "181_13", "186_10", "186_11", "186_12"]

    files_mapped = "E:/diagrams/bpmn-io/bpmn2image/files_mapped/"
    images = "E:/diagrams/bpmn-io/bpmn2image/images/"

    files_invalid = "E:/diagrams/bpmn-io/bpmn2image/files_invalid/"
    images_invalid = "E:/diagrams/bpmn-io/bpmn2image/images_invalid/"

    bpmns = os.listdir(files_mapped)
    imgs = os.listdir(images)

    for i in range(len(bpmns)):
        f_id = get_file_id(bpmns[i])
        if f_id in invalid_list:
            shutil.move(files_mapped+bpmns[i], files_invalid)
            shutil.move(images+imgs[i], images_invalid)


def map_image_and_file(file_dir, img_dir, mapped_dir, unmapped_dir):
    bpmns = os.listdir(file_dir)
    imgs = os.listdir(img_dir)

    j = 0
    for bpmn in bpmns:
        bpmn_id = get_file_id(bpmn)
        img_id = get_file_id(imgs[j])
        if bpmn_id == img_id:
            j += 1
            shutil.copy(file_dir+bpmn, mapped_dir)
        else:
            shutil.copy(file_dir+bpmn, unmapped_dir)


def filter_bad(file, bad_dir):
    bad_list = ["003_10", "004_01", "005_06", "011_17", "014_18", "017_00", "017_16", "021_04", "027_04", "035_07",
                "035_08", "040_17", "044_01", "045_18", "048_15", "050_10", "052_13", "055_13", "056_03", "056_06",
                "063_18", "067_07", "079_01", "082_03", "112_13", "123_04", "123_19", "139_10", "143_07", "143_13",
                "162_07", "164_12", "165_13", "167_11", "168_15", "170_07", "171_03", "187_10"]
    file_name = file.split("/")[-1]
    file_id = "_".join(file_name.split("_")[0:2])

    # print(file_id)

    if file_id in bad_list:
        shutil.move(file, bad_dir)


def filter_multi(file, mul_dir, sgl_dir):
    file_name = file.split("/")[-1]
    try:
        dom_tree = et.parse(file)
    except et.ParseError:
        print(file_name, " parseError")
        return

    definitions = dom_tree.getroot()
    count = 0
    for d in definitions.iter("{http://www.omg.org/spec/BPMN/20100524/DI}BPMNDiagram"):
        count += 1

    if count > 1:
        pass
        # shutil.copy(file, mul_dir)
    elif count == 1:
        pass
        # shutil.copy(file, sgl_dir)
    else:
        print(file_name, " no diagram")


def split_bpmn(file, output):
    file_name = file.split("/")[-1]
    dom_tree = et.parse(file)
    definitions = dom_tree.getroot()

    et.register_namespace("", "http://genmymodel.com/bpmn2")
    et.register_namespace("bpmn2", "http://www.omg.org/spec/BPMN/20100524/MODEL")
    et.register_namespace("bpmndi", "http://www.omg.org/spec/BPMN/20100524/DI")
    et.register_namespace("dc", "http://www.omg.org/spec/DD/20100524/DC")
    et.register_namespace("di", "http://www.omg.org/spec/DD/20100524/DI")
    count = 0

    for d in definitions.iter("{http://www.omg.org/spec/BPMN/20100524/DI}BPMNDiagram"):
        one = et.Element("defenitions")
        one.attrib = definitions.attrib
        one.tag = definitions.tag
        try:
            for plane in d.iter("{http://www.omg.org/spec/BPMN/20100524/DI}BPMNPlane"):
                collab_id = plane.attrib["bpmnElement"]
                for collab in definitions.iter("{http://www.omg.org/spec/BPMN/20100524/MODEL}collaboration"):
                    if collab.attrib["id"] == collab_id:
                        one.append(collab)
                        for participant in collab.iter("{http://www.omg.org/spec/BPMN/20100524/MODEL}participant"):
                            pf = participant.attrib["processRef"]
                            for process in definitions.iter("{http://www.omg.org/spec/BPMN/20100524/MODEL}process"):
                                if process.attrib["id"] == pf:
                                    one.append(process)
            one.append(d)
            tree = et.ElementTree(one)

            name_seg = file_name.split("_")
            name_seg.insert(2, str(count))
            output_path = output + "_".join(name_seg)
            print(output_path)
            tree.write(output_path, encoding="utf-8", xml_declaration=True)
            count += 1
        except KeyError:
            print(file_name)
            shutil.move(file, "E:/diagrams/bpmn-io/bpmn2image/files_multi_invalid")
            break


if __name__ == '__main__':
    # file_dir = "E:/diagrams/bpmn-io/bpmn2image/files/"
    # multi_dir = "E:/diagrams/bpmn-io/bpmn2image/files_multi/"
    # single_dir = "E:/diagrams/bpmn-io/bpmn2image/files_single/"
    # bpmns = os.listdir(file_dir)
    # for bpmn in bpmns:
    #     f = file_dir + bpmn
    #     filter_multi(f, multi_dir, single_dir)

    # bad_dir = "E:/diagrams/bpmn-io/bpmn2image/files_bad/"
    # single = os.listdir(single_dir)
    # for bpmn in single:
    #     f = single_dir + bpmn
    #     filter_bad(f, bad_dir)

    # image_dir = "E:/diagrams/bpmn-io/bpmn2image/images/"
    # files_mapped = "E:/diagrams/bpmn-io/bpmn2image/files_mapped/"
    # files_not_mapped = "E:/diagrams/bpmn-io/bpmn2image/files_not_mapped/"
    # map_image_and_file(single_dir, image_dir, files_mapped, files_not_mapped)

    filter_invalid()
