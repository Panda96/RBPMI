# -*- coding:utf-8 -*-
import os
import shutil
import cv2 as cv

import count
import rec_helper as rh


def filter_enough_data():
    train_data = "E:/diagrams/bpmn-io/bpmn2image/data0423/ele_type_data/"
    enough_data = "E:/diagrams/bpmn-io/bpmn2image/data0423/ele_enough_data/"
    little_data = "E:/diagrams/bpmn-io/bpmn2image/data0423/ele_little_data/"

    if not os.path.exists(enough_data):
        os.mkdir(enough_data)

    if not os.path.exists(little_data):
        os.mkdir(little_data)

    type_dirs = os.listdir(train_data)
    for dir_name in type_dirs:
        file_num = len(os.listdir(train_data + dir_name))
        if file_num >= 50:
            print(dir_name)
            shutil.copytree(train_data + dir_name, enough_data + dir_name)
        else:
            print(dir_name)
            shutil.copytree(train_data + dir_name, little_data + dir_name)


def move_complex_type():
    complex_ele_list = ["adHocSub_expanded", "group", "lane", "participant", "subProcess_expanded",
                        "subProcess_mulInsL_expanded", "subProcess_stdL_expanded", "subProcess_withBound_expanded",
                        "transaction_withBound_expanded", "textAnnotation"]
    train_data = "E:/diagrams/bpmn-io/bpmn2image/data0423/ele_type_data/"
    complex_data = "E:/diagrams/bpmn-io/bpmn2image/data0423/ele_complex_data/"

    if not os.path.exists(complex_data):
        os.mkdir(complex_data)

    type_dirs = os.listdir(train_data)

    for dir_name in type_dirs:
        if dir_name in complex_ele_list:
            print(dir_name)
            shutil.copytree(train_data + dir_name, complex_data + dir_name)
            shutil.rmtree(train_data + dir_name)


def make_type_dirs(ele_type_list):
    train_data = "E:/diagrams/bpmn-io/bpmn2image/data0423/ele_type_data/"

    if not os.path.exists(train_data):
        os.mkdir(train_data)

    for one_type in ele_type_list:
        one_type_dir = train_data + one_type
        if not os.path.exists(one_type_dir):
            os.mkdir(one_type_dir)


def get_file_id(file_name):
    return "_".join(file_name.split("_")[0:2])


def distribute_shape_data():
    # data_dir = "E:/diagrams/bpmn-io/bpmn2image/data0423/data700_3/ele_type_data/"
    # files_dir = "E:/diagrams/bpmn-io/bpmn2image/data0423/data700_3/bpmn/"
    # imgs_dir = "E:/diagrams/bpmn-io/bpmn2image/data0423/data700_3/imgs/"
    data_dir = "E:/diagrams/bpmn-io/bpmn2image/data0423/ele_type_data/"
    files_dir = "E:/diagrams/bpmn-io/bpmn2image/data0423/files/"
    imgs_dir = "E:/diagrams/bpmn-io/bpmn2image/data0423/images/"
    print("Start Counting ...")
    count.count(files_dir)
    print("Counting finished!")
    ele_type_list = count.statistic()
    make_type_dirs(ele_type_list)

    all_shapes_label = count.all_shapes_label

    imgs = os.listdir(imgs_dir)

    i = 0
    file_id = get_file_id(imgs[i])
    file_path = imgs_dir + imgs[i]
    cur_img = cv.imread(file_path, cv.COLOR_BGR2GRAY)
    for shape_id, shape_label in enumerate(all_shapes_label):
        while shape_label[0] != file_id:
            i += 1
            file_id = get_file_id(imgs[i])
            print(file_id)
            file_path = imgs_dir + imgs[i]
            cur_img = cv.imread(file_path)

        shape_rec = shape_label[2]
        if shape_rec[2] * shape_rec[3] == 0:
            print(file_id)
            continue
        shape_rec = rh.dilate(shape_rec, 5)
        # print("{}:{}".format(str(shape_id)+"_"+file_id, shape_rec))
        shape_img = rh.truncate(cur_img, shape_rec)
        shape_type = shape_label[1]
        shape_dir = ""
        for ele_type in ele_type_list:
            if ele_type.startswith(shape_type):
                shape_dir = ele_type
                break
        if len(shape_dir) > 0:
            shape_path = data_dir + shape_dir + "/" + str(shape_id) + "_" + file_id + ".png"
            cv.imwrite(shape_path, shape_img)


if __name__ == '__main__':
    # distribute_shape_data()
    # move_complex_type()
    filter_enough_data()
