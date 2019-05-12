# -*- coding:utf-8 -*-
import sys
import getopt

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

    def vgg_pre_process_image(self, images):
        imgs = []
        for image in images:
            if type(image) == str:
                img = cv.imread(image)
            else:
                img = image
            img = 255 - img
            img_shape = (self.img_size, self.img_size)
            img = cv.resize(img, img_shape)
            imgs.append(img)
        imgs = np.array(imgs)
        return imgs

    def bcf_pre_process_image(self, images):
        imgs = []
        for image in images:
            if type(image) == str:
                imgs.append(cv.imread(image))
            else:
                imgs.append(image)
        # imgs = np.array(imgs)
        return imgs

    def load_vgg_16_classifier(self):
        if self.vgg_16_classifier is None:
            self.vgg_16_classifier = modeler.get_vgg16_fc(self.img_size, len(self.classes))
            self.vgg_16_classifier.load_weights("../image_classification/weights/VGG16_fc_model.h5")

    def load_vgg_16_56_classifier(self):
        if self.vgg_16_56_classifier is None:
            self.vgg_16_56_classifier = modeler.get_vgg16_fc(self.img_size, len(self.classes_56))
            self.vgg_16_56_classifier.load_weights("../image_classification/weights/VGG16_fc_model_56_2.h5")

    def load_vgg_16_57_classifier(self):
        if self.vgg_16_57_classifier is None:
            self.vgg_16_57_classifier = modeler.get_vgg16_fc(self.img_size, len(self.classes_57))
            self.vgg_16_57_classifier.load_weights("../image_classification/weights/VGG16_fc_model_57_2.h5")

    def classify_with_vgg_16(self, clf, images, labels):
        imgs = self.vgg_pre_process_image(images)
        results = clf.predict(imgs)
        predictions = []
        for result in results:
            try:
                label = labels[int(np.where(result == 1)[0])]
            except TypeError:
                label = "None"
                print(result.shape)
                print(np.where(result == 1))
            predictions.append(label)
        return predictions

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
            self.bcf_56_classifier = bcf

    def load_bcf_57_classifier(self):
        if self.bcf_57_classifier is None:
            bcf = BCF()
            bcf.CLASSIFIER_FILE = "../bcf/model/classifier_57_30_50"
            bcf.CODEBOOK_FILE = "../bcf/model/code_book_57_30.data"
            bcf.load_kmeans()
            bcf.load_classifier()
            self.bcf_57_classifier = bcf

    def classify_with_bcf(self, clf, images):
        imgs = self.bcf_pre_process_image(images)
        return clf.get_images_type(imgs)

    def classify(self, images, classifier_type):
        if classifier_type == "vgg16":
            self.load_vgg_16_classifier()
            predictions = self.classify_with_vgg_16(self.vgg_16_classifier, images, self.classes)
        elif classifier_type == "vgg16_56":
            self.load_vgg_16_56_classifier()
            predictions = self.classify_with_vgg_16(self.vgg_16_56_classifier, images, self.classes_56)
        elif classifier_type == "vgg16_57":
            self.load_vgg_16_57_classifier()
            predictions = self.classify_with_vgg_16(self.vgg_16_57_classifier, images, self.classes_57)
        elif classifier_type == "bcf":
            self.load_bcf_classifier()
            predictions = self.classify_with_bcf(self.bcf_classifier, images)
        elif classifier_type == "bcf_56":
            self.load_bcf_56_classifier()
            predictions = self.classify_with_bcf(self.bcf_56_classifier, images)
        else:
            self.load_bcf_57_classifier()
            predictions = self.classify_with_bcf(self.bcf_57_classifier, images)
        return predictions

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

        test_res = {}

        type_dirs = os.listdir(data_test)
        for each_type in type_dirs:
            print(each_type)
            test_res[each_type] = [[0, 0], []]
            type_dir = data_test + each_type + "/"
            image_names = os.listdir(type_dir)
            images = [type_dir + name for name in image_names]
            predictions = self.classify(images, classifier_type)
            test_res[each_type][0][0] = len(images)
            for i, prediction in enumerate(predictions):
                if prediction == each_type:
                    test_res[each_type][0][1] += 1
                else:
                    test_res[each_type][1].append("Mistook {} {} for {}".format(image_names[i], each_type, prediction))

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
                f.write(record + "\n")

                # if correct_num < total:
                for info in test_res[label][1]:
                    print(info)
                    f.write(info + "\n")
            record = "{}\t{},{},{}".format("all", all_total, all_correct, all_correct / all_total)
            print(record)
            f.write(record + "\n")


if __name__ == '__main__':

    classifier = Classifier()
    opt = sys.argv[1]

    if opt == "vgg":
        print("test vgg")
        classifier.test("vgg16")
        classifier.test("vgg16_56")
        classifier.test("vgg16_57")
    elif opt == "bcf":
        print("test bcf")
        classifier.test("bcf")
        classifier.test("bcf_56")
        classifier.test("bcf_57")
    else:
        print("wrong args, it should be 'test_***'")

