# -*- coding:utf-8 -*-
import os
import shutil
import numpy as np

import cv2 as cv

shape_data = "E:/diagrams/bpmn-io/bpmn2image/data0423/ele_enough_data/"
data_set = "E:/diagrams/bpmn-io/bpmn2image/data0423/data_set/"
train_data = data_set + "train/"
test_data = data_set + "test/"


def png_to_jpg():
    all_data = "E:/diagrams/bpmn-io/bpmn2image/data0423/ele_enough_data/"
    jpg_data = "E:/diagrams/bpmn-io/bpmn2image/data0423/ele_enough_data_jpg/"
    type_list = os.listdir(all_data)

    # for each_type in type_list:
    #     os.makedirs(jpg_data + each_type + "/")

    for each_type in type_list:
        print(each_type)
        type_dir = all_data + each_type + "/"
        for name in os.listdir(type_dir):
            image = cv.imread(type_dir + name)
            jpg_path = jpg_data + each_type + "/" + name.replace(".png", ".jpg")
            cv.imwrite(jpg_path, image)
            # print(jpg_path)
        # break


def divide_120_data():
    all_data = "E:/diagrams/bpmn-io/bpmn2image/data0423/ele_enough_data_jpg/"
    data_120 = "E:/diagrams/bpmn-io/bpmn2image/data0423/120data/"
    data_120_train = data_120 + "train/"
    data_120_val = data_120 + "val/"
    data_120_test = data_120 + "test/"

    type_list = os.listdir(all_data)

    for each_type in type_list:
        os.makedirs(data_120_train + each_type)
        os.makedirs(data_120_val + each_type)
        os.makedirs(data_120_test + each_type)

    for each_type in type_list:
        print(each_type)
        type_dir = all_data + each_type + "/"
        type_images = os.listdir(type_dir)
        size = len(type_images)
        index = np.arange(size)
        np.random.shuffle(index)

        if size > 120:
            index = index[:120]
            size = 120

        for i in index[0:int(0.6 * size)]:
            image_name = type_images[i]
            image_path = type_dir + image_name
            target_path = data_120_train + each_type + "/" + image_name
            shutil.copy(image_path, target_path)

        for i in index[int(0.6 * size): int(0.8 * size)]:
            image_name = type_images[i]
            image_path = type_dir + image_name
            target_path = data_120_val + each_type + "/" + image_name
            shutil.copy(image_path, target_path)

        for i in index[int(0.8 * size):]:
            image_name = type_images[i]
            image_path = type_dir + image_name
            target_path = data_120_test + each_type + "/" + image_name
            shutil.copy(image_path, target_path)


def divide_6_2_2_data():
    # all_data = "E:/diagrams/bpmn-io/bpmn2image/data0423/ele_merge_data_500/"
    all_data = "E:/diagrams/bpmn-io/bpmn2image/data0423/ele_type_57_aug_500/"
    data_622 = "E:/diagrams/bpmn-io/bpmn2image/data0423/57_622data/"
    data_622_train = data_622 + "train/"
    data_622_val = data_622 + "val/"
    data_622_test = data_622 + "test/"

    type_list = os.listdir(all_data)

    for each_type in type_list:
        if not os.path.exists(data_622_train+each_type):
            os.makedirs(data_622_train+each_type)
            os.makedirs(data_622_val+each_type)
            os.makedirs(data_622_test+each_type)
        else:
            print("dirs exists")
            return

    for each_type in type_list:
        print(each_type)
        type_dir = all_data + each_type + "/"
        type_images = os.listdir(type_dir)
        size = len(type_images)
        index = np.arange(size)
        np.random.shuffle(index)

        for i in index[0:int(0.6 * size)]:
            image_name = type_images[i]
            image_path = type_dir + image_name
            target_path = data_622_train + each_type + "/" + image_name
            shutil.copy(image_path, target_path)

        for i in index[int(0.6 * size): int(0.8 * size)]:
            image_name = type_images[i]
            image_path = type_dir + image_name
            target_path = data_622_val + each_type + "/" + image_name
            shutil.copy(image_path, target_path)

        for i in index[int(0.8 * size):]:
            image_name = type_images[i]
            image_path = type_dir + image_name
            target_path = data_622_test + each_type + "/" + image_name
            shutil.copy(image_path, target_path)


def divide_50_data():
    all_data = "E:/diagrams/bpmn-io/bpmn2image/data0423/ele_type_data/"
    data_50 = "E:/diagrams/bpmn-io/bpmn2image/data0423/ele_data_50/"
    type_dirs = os.listdir(all_data)
    for type_dir in type_dirs:
        print(type_dir)
        type_path = data_50 + type_dir
        if not os.path.exists(type_path):
            os.mkdir(type_path)

        shapes = os.listdir(all_data + type_dir)

        size = len(shapes)
        index = np.arange(size)
        np.random.shuffle(index)
        num = min(size, 50)

        for i in range(num):
            file_name = shapes[index[i]]
            shutil.copy(all_data + type_dir + "/" + file_name, data_50 + type_dir + "/" + file_name)


def prepare_test():
    data_set = "E:/diagrams/bpmn-io/bpmn2image/data0423/data_set_test/"
    train_data = data_set + "train/"
    print(train_data)
    type_dirs = os.listdir(shape_data)
    for type_dir in type_dirs:
        if not os.path.exists(train_data + type_dir):
            os.makedirs(train_data + type_dir)

    for type_dir in type_dirs:
        shapes = os.listdir(shape_data + type_dir)
        file_name = shapes[0]
        shutil.copy(shape_data + type_dir + "/" + file_name, train_data + type_dir + "/" + file_name)


def prepare():
    # validate_data = "E:/diagrams/bpmn-io/bpmn2image/data0423/data_set/validate/"

    type_dirs = os.listdir(shape_data)

    for type_dir in type_dirs:
        if not os.path.exists(train_data + type_dir):
            os.makedirs(train_data + type_dir)
        if not os.path.exists(test_data + type_dir):
            os.makedirs(test_data + type_dir)
        # if not os.path.exists(validate_data + type_dir):
        #     os.makedirs(validate_data + type_dir)


def divide_data():
    np.random.seed(423)

    type_dirs = os.listdir(shape_data)
    for type_dir in type_dirs:
        print(type_dir)
        shapes = os.listdir(shape_data + type_dir)
        size = len(shapes)
        index = np.arange(size)
        np.random.shuffle(index)
        num = min(int(0.7 * len(shapes)), 50)

        for i in range(num):
            file_name = shapes[index[i]]
            shutil.copy(shape_data + type_dir + "/" + file_name, train_data + type_dir + "/" + file_name)

        for i in range(num, size):
            file_name = shapes[index[i]]
            shutil.copy(shape_data + type_dir + "/" + file_name, test_data + type_dir + "/" + file_name)


def get_data_500():
    # merge_data = "E:/diagrams/bpmn-io/bpmn2image/data0423/ele_merge_data/"
    # data_500 = "E:/diagrams/bpmn-io/bpmn2image/data0423/ele_merge_data_500/"
    aug_data = "E:/diagrams/bpmn-io/bpmn2image/data0423/ele_type_57_aug_data/"
    data_57_500 = "E:/diagrams/bpmn-io/bpmn2image/data0423/ele_type_57_aug_500/"

    type_dirs = os.listdir(aug_data)
    for each_type in type_dirs:
        type_dir = aug_data + each_type + "/"
        data_500_type_dir = data_57_500 + each_type + "/"
        print(each_type)
        images = os.listdir(type_dir)
        size = len(images)
        index = np.arange(size)
        np.random.shuffle(index)

        for i in index[0:500]:
            image_name = images[i]
            image_path = type_dir + image_name
            target_path = data_500_type_dir + image_name
            shutil.copy(image_path, target_path)


if __name__ == '__main__':
    # prepare()
    # divide_data()
    # prepare_test()
    # divide_50_data()
    # png_to_jpg()
    divide_6_2_2_data()
    # divide_120_data()
    # get_data_500()
