# -*- coding:utf-8 -*-
import os
import cv2 as cv

import count
import rec_helper as rh


def make_type_dirs(ele_type_list):
    train_data = "E:/diagrams/bpmn-io/bpmn2image/data0423/ele_type_data/"
    # expanded_data = "E:/diagrams/bpmn-io/bpmn2image/data0423/ele_expanded_data/"
    #
    # complex_ele_list = ["adHocSub_expanded", "group", "lane", "participant", "subProcess_expanded",
    #                     "subProcess_mulInsL_expanded", "subProcess_withBound_expanded", ""]

    for one_type in ele_type_list:
        # if "_".join(one_type.split("_")[0:-1] in complex_ele_list:
        #     pass
        one_type_dir = train_data + one_type
        if not os.path.exists(one_type_dir):
            os.makedirs(one_type_dir)


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
        shape_path = data_dir + shape_label[1] + "/" + str(shape_id) + "_" + file_id + ".png"
        cv.imwrite(shape_path, shape_img)


if __name__ == '__main__':
    # make_type_dirs()
    distribute_shape_data()
