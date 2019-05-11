# -*- coding:utf-8 -*-
import os

import cv2 as cv
import numpy as np
import modeler


def image_reverse(image):
    return 255 - image


data_dir = "../56_622data/"

data_train = data_dir + "train/"
data_test = data_dir + "test/"
data_val = data_dir + "val/"

labels = os.listdir(data_train)
classes = len(labels)

img_size = 150
img_shape = (img_size, img_size)

tune_model = modeler.get_vgg16_fc(img_size, classes)
tune_model.load_weights(filepath="weights/VGG16_fc_model_56.h5")
tune_model.summary()

testing_data = []
test_labels = []
type_dirs = os.listdir(data_test)
predictions = []
# {type:[[test_num, correct_num], ["mistook info"...]]}
test_res = {}

for each_type in type_dirs:
    print(each_type)
    test_res[each_type] = [[0, 0], []]
    images = os.listdir(data_test + each_type)
    for image in images:
        image_path = data_test + each_type + "/" + image
        img = cv.imread(image_path)
        img = image_reverse(img)
        img = cv.resize(img, img_shape)
        img = img.reshape((1,) + img.shape)
        test_labels.append([each_type, image])
        result = tune_model.predict(img, batch_size=1)
        try:
            predictions.append(labels[int(np.where(result == 1)[1])])
        except TypeError:
            predictions.append("None")

for (i, test_label) in enumerate(test_labels):
    type_name = test_label[0]
    image_name = test_label[1]
    test_res[type_name][0][0] += 1
    if predictions[i] == type_name:
        test_res[type_name][0][1] += 1
    else:
        test_res[type_name][1].append("Mistook {} {} for {}".format(type_name, image_name, predictions[i]))

all_total = 0
all_correct = 0
for label in labels:
    total = test_res[label][0][0]
    correct_num = test_res[label][0][1]
    all_total += total
    all_correct += correct_num
    print("{}\t{},{},{}".format(label, total, correct_num, correct_num / total))
    # if correct_num < total:
    for info in test_res[label][1]:
        print(info)
print("{}\t{},{},{}".format("all", all_total, all_correct, all_correct / all_total))
