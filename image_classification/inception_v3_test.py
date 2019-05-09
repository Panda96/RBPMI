# -*- coding:utf-8 -*-
import os

import cv2 as cv
import numpy as np
from keras.models import Model
from keras import layers
from keras.applications.inception_v3 import InceptionV3


def image_reverse(image):
    return 255 - image


data_dir = "../622data/"

data_train = data_dir + "train/"
data_test = data_dir + "test/"
data_val = data_dir + "val/"

labels = os.listdir(data_train)
classes = len(labels)

img_size = 150
img_shape = (img_size, img_size)


base_model = InceptionV3(include_top=False, input_shape=(img_size, img_size, 3))
# base_model.summary()

out = base_model.layers[-1].output
out = layers.Flatten()(out)
out = layers.Dense(1024, activation='relu')(out)
# 因为前面输出的dense feature太多了，我们这里加入dropout layer来防止过拟合
out = layers.Dropout(0.5)(out)
out = layers.Dense(512, activation='relu')(out)
out = layers.Dropout(0.3)(out)
out = layers.Dense(classes, activation='softmax')(out)
tuneModel = Model(inputs=base_model.input, outputs=out)
tuneModel.load_weights(filepath="weights/Inception_v3_fc_model.h5")


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
        result = tuneModel.predict(img, batch_size=1)
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
