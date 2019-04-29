# -*- coding:utf-8 -*-
import os
import shutil
import numpy as np

shape_data = "E:/diagrams/bpmn-io/bpmn2image/data0423/ele_enough_data/"
data_set = "E:/diagrams/bpmn-io/bpmn2image/data0423/data_set/"
train_data = data_set + "train/"
test_data = data_set + "test/"


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


if __name__ == '__main__':
    # prepare()
    # divide_data()
    prepare_test()
