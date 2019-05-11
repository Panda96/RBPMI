# -*- coding:utf-8 -*-
import sys
sys.path.append("../")
sys.path.append("../bcf/")

from image_classification import modeler
from bcf.bcf import BCF

import cv2 as cv
import numpy as np
import os


class Classifier:
    def __init__(self):
        self.vgg_16_classifier = None
        self.vgg_16_56_classifier = None
        self.vgg_16_57_classifier = None

        self.bcf_classifier = None
        self.bcf_56_classifier = None
        self.bcf_57_classifier = None

        self.classes = os.listdir("../622data/train/")
        self.classes_56 = os.listdir("../56_622data/train/")
        self.classes_57 = os.listdir("../57_622data/train/")

        self.img_size = 150

    def vgg_pre_process_image(self, image):
        if type(image) == str:
            img = cv.imread(image)
        else:
            img = image
        img = 255 - img
        img_shape = (self.img_size, self.img_size)
        img = cv.resize(img, img_shape)
        img = img.reshape((1,) + img.shape)
        return img

    def bcf_pre_process_image(self, image):
        if type(image) == str:
            img = cv.imread(image)
        else:
            img = image
        return img

    def load_vgg_16_classifier(self):
        if self.vgg_16_classifier is None:
            self.vgg_16_classifier = modeler.get_vgg16_fc(self.img_size, self.classes)
            self.vgg_16_classifier.load_weights("../image_classification/weights/VGG16_fc_model.h5")

    def load_vgg_16_56_classifier(self):
        if self.vgg_16_56_classifier is None:
            self.vgg_16_56_classifier = modeler.get_vgg16_fc(self.img_size, self.classes_56)
            self.vgg_16_56_classifier.load_weights("")

    def load_vgg_16_57_classifier(self):
        if self.vgg_16_57_classifier is None:
            self.vgg_16_57_classifier = modeler.get_vgg16_fc(self.img_size, self.classes_57)
            self.vgg_16_57_classifier.load_weights("")

    def classify_with_vgg_16(self, image):
        self.load_vgg_16_classifier()
        img = self.vgg_pre_process_image(image)
        result = self.vgg_16_classifier.predict(img, batch_size=1)
        label = self.classes[int(np.where(result == 1)[1])]
        return label

    def classify_with_vgg_16_56(self, image):
        self.load_vgg_16_56_classifier()
        img = self.vgg_pre_process_image(image)
        result = self.vgg_16_56_classifier.predict(img, batch_size=1)
        label = self.classes_56[int(np.where(result == 1)[1])]
        return label

    def classify_with_vgg_16_57(self, image):
        self.load_vgg_16_57_classifier()
        img = self.vgg_pre_process_image(image)
        result = self.vgg_16_57_classifier.predict(img, batch_size=1)
        label = self.classes_57[int(np.where(result == 1)[1])]
        return label

    def load_bcf_classifier(self):
        if self.bcf_classifier is None:
            bcf = BCF()
            bcf.CLASSIFIER_FILE = "../bcf/model/classifier_30_50"
            bcf.CODEBOOK_FILE = "../bcf/model/code_book_56_30.data"
            bcf.load_kmeans()
            bcf.load_classifier()
            self.bcf_classifier = bcf

    def load_bcf_56_classifier(self):
        if self.bcf_56_classifier is None:
            bcf = BCF()
            bcf.CLASSIFIER_FILE = "../bcf/model/classifier_56_30_50"
            bcf.CODEBOOK_FILE = "../bcf/model/code_book_56_30.data"
            bcf.load_kmeans()
            bcf.load_classifier()
            self.bcf_classifier = bcf

    def load_bcf_57_classifier(self):
        if self.bcf_57_classifier is None:
            bcf = BCF()
            bcf.CLASSIFIER_FILE = "../bcf/model/classifier_57_30_50"
            bcf.CODEBOOK_FILE = "../bcf/model/code_book_57_30.data"
            bcf.load_kmeans()
            bcf.load_classifier()
            self.bcf_classifier = bcf

    def classify_with_bcf(self, image):
        self.load_bcf_classifier()
        img = self.bcf_pre_process_image(image)
        return self.bcf_classifier.get_one_image_type(img)

    def classify_with_bcf_56(self, image):
        self.load_bcf_56_classifier()
        img = self.bcf_pre_process_image(image)
        return self.bcf_56_classifier.get_one_image_type(img)

    def classify_with_bcf_57(self, image):
        self.load_bcf_57_classifier()
        img = self.bcf_pre_process_image(image)
        return self.bcf_57_classifier.get_one_image_type(img)

    def test(self, classifier_type):
        # vgg16, vgg16_56, vgg16_57, bcf, bcf_56, bcf_57
        type_info = classifier_type.split("_")
        if len(type_info) == 1:
            data_dir = "../622data/"
        elif type_info[1] == "56":
            data_dir = "../56_622data/"
        else:
            data_dir = "../57_622data/"

        data_test = data_dir + "test/"
        # labels = os.listdir(data_test)

        test_labels = []
        predictions = []
        test_res = {}

        type_dirs = os.listdir(data_test)
        for each_type in type_dirs:
            print(each_type)
            test_res[each_type] = [[0, 0], []]
            type_dir = data_test + each_type + "/"
            images = os.listdir(type_dir)
            for image in images:
                file_path = type_dir + image
                test_labels.append([each_type, image])

                prediction = self.classify(file_path, classifier_type)
                predictions.append(prediction)

        for (i, test_label) in enumerate(test_labels):
            type_name = test_label[0]
            image_name = test_label[1]
            test_res[type_name][0][0] += 1
            if predictions[i] == type_name:
                test_res[type_name][0][1] += 1
            else:
                test_res[type_name][1].append("Mistook {} {} for {}".format(type_name, image_name, predictions[i]))

        test_logs = os.listdir("test_results/")
        file_id = len(test_logs)
        test_res_file = "test_results/{}_{}_test.txt".format(file_id, classifier_type)
        with open(test_res_file, "w") as f:
            all_total = 0
            all_correct = 0
            for label in type_dirs:
                total = test_res[label][0][0]
                correct_num = test_res[label][0][1]
                all_total += total
                all_correct += correct_num
                record = "{}\t{},{},{}".format(label, total, correct_num, correct_num / total)
                print(record)
                f.write(record+"\n")

                # if correct_num < total:
                for info in test_res[label][1]:
                    print(info)
            record = "{}\t{},{},{}".format("all", all_total, all_correct, all_correct / all_total)
            print(record)
            f.write(record+"\n")

    def classify(self, image, classifier_type):
        if classifier_type == "vgg16":
            prediction = self.classify_with_vgg_16(image)
        elif classifier_type == "vgg16_56":
            prediction = self.classify_with_vgg_16_56(image)
        elif classifier_type == "vgg16_57":
            prediction = self.classify_with_vgg_16_57(image)
        elif classifier_type == "bcf":
            prediction = self.classify_with_bcf(image)
        elif classifier_type == "bcf_56":
            prediction = self.classify_with_bcf_56(image)
        else:
            prediction = self.classify_with_bcf_57(image)

        return prediction


if __name__ == '__main__':
    # image_path = "../622data/test/dataObjectReference/4034_019_01.png"
    # image_mat = cv.imread(image_path)
    classifier = Classifier()
    # res = classifier.classify_with_vgg_16(image_path)
    # print(res)
    # res = classifier.classify_with_vgg_16(image_mat)
    # print(res)
    classifier.test("bcf")
