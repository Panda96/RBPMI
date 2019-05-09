# -*- coding:utf-8 -*-
import os
import shutil
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


def augment():
    type_dirs = os.listdir(data_dir)

    for each_type in type_dirs:
        type_target_dir = augment_data_dir + each_type + "/"
        if not os.path.exists(type_target_dir):
            os.makedirs(type_target_dir)
        else:
            print("dirs exists!")
            return

    data_gen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        fill_mode="constant",
        cval=255.)

    for each_type in type_dirs:
        type_dir = data_dir + each_type + "/"
        type_target_dir = augment_data_dir + each_type + "/"
        # type_dir = "raw/"
        # augment_data_dir = "test/"
        max_size = 500
        gen_num = 600
        images = os.listdir(type_dir)
        current_size = len(images)
        if len(images) < max_size:
            print(each_type)

            loop_num = (gen_num - current_size) // current_size

            for image in images:
                image_path = type_dir + image
                img = load_img(image_path)
                x = img_to_array(img)
                x = x.reshape((1,) + x.shape)
                i = 0
                for batch in data_gen.flow(x, batch_size=1, save_to_dir=type_target_dir, save_prefix=image[0:-4]):
                    i += 1
                    if i > loop_num:
                        break


def count_data():
    type_dirs = os.listdir(augment_data_dir)
    for each_type in type_dirs:
        data_type_dir = data_dir + each_type + "/"
        augment_data_type_dir = augment_data_dir + each_type + "/"
        size_1 = len(os.listdir(data_type_dir))
        size_2 = len(os.listdir(augment_data_type_dir))
        print("{}:{}+{}={}".format(each_type, size_1, size_2, size_1 + size_2))


def count_one_dir(one_dir):
    type_dirs = os.listdir(one_dir)
    for each_type in type_dirs:
        type_dir = merge_data_dir + each_type + "/"
        print("{}:{}".format(each_type, len(os.listdir(type_dir))))


def make_dirs(target_dir):
    type_dirs = os.listdir(data_dir)
    for each_type in type_dirs:
        type_target_dir = target_dir + each_type + "/"
        if not os.path.exists(type_target_dir):
            os.makedirs(type_target_dir)
        else:
            print("dirs exists!")
            return


def merge_data():
    type_dirs = os.listdir(data_dir)
    for each_type in type_dirs:
        print(each_type)
        data_type_dir = data_dir + each_type + "/"
        aug_data_type_dir = augment_data_dir + each_type + "/"
        merge_type_dir = merge_data_dir + each_type + "/"

        data_type_images = os.listdir(data_type_dir)
        aug_data_type_images = os.listdir(aug_data_type_dir)

        for image in data_type_images:
            image_path = data_type_dir + image
            target_path = merge_type_dir + image
            shutil.copy(image_path, target_path)

        for image in aug_data_type_images:
            image_path = aug_data_type_dir + image
            target_path = merge_type_dir + image
            shutil.copy(image_path, target_path)


if __name__ == '__main__':
    data_dir = "E:/diagrams/bpmn-io/bpmn2image/data0423/ele_type_data/"
    augment_data_dir = "E:/diagrams/bpmn-io/bpmn2image/data0423/ele_type_aug_data/"
    merge_data_dir = "E:/diagrams/bpmn-io/bpmn2image/data0423/ele_merge_data/"
    merge_data_500 = "E:/diagrams/bpmn-io/bpmn2image/data0423/ele_merge_data_500/"
    # count_data()
    make_dirs(merge_data_500)
    # merge_data()
    # count_one_dir(merge_data_dir)
