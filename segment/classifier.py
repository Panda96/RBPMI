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
        self.vgg_classifier = None

        self.bcf_classifier = None
        self.bcf_56_classifier = None
        self.bcf_57_classifier = None

        self.img_type = None

        self.classifiers = ["vgg16", "vgg16_56",
                            "vgg16_57", "vgg16_52"]

        # self.classes = [os.listdir("../622data/train/"), os.listdir("../56_622data/train/"),
        #                 os.listdir("../57_622data/train/"),
        #                 os.listdir("../png_training_data/train/")]
        self.classes = [[], [], [], os.listdir("../training_data_png/train/")]

        self.weights = ["../image_classification/weights_2/VGG16_fc_model_3.h5",
                        "../image_classification/weights_2/VGG16_fc_model_56_2.h5",
                        "../image_classification/weights_2/VGG16_fc_model_57_2.h5",
                        {"png": "../image_classification/weights_52/png/VGG16_fc_model_png_1.h5",
                         "jpg": "../image_classification/weights_52/jpg/VGG16_fc_model_jpg_4.h5"}]

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

    def load_vgg_classifier(self, classifier_type):
        classifier_id = self.classifiers.index(classifier_type)
        if classifier_id >= 0:
            self.vgg_classifier = modeler.get_vgg16_fc(self.img_size, len(self.classes[classifier_id]))
            one_weight = self.weights[classifier_id]
            if classifier_type == "vgg16_52":
                if self.img_type is not None:
                    one_weight = one_weight[self.img_type]
                else:
                    print("vgg16_52 classifier needs to specify one image type")
                    exit(0)
            self.vgg_classifier.load_weights(one_weight)
        else:
            print("invalid classifier type:{}".format(classifier_type))

    # def load_bcf_classifier(self, classifier_type):
    #     print("bcf classifiers are not considered from now on")

    def classify_with_vgg_16(self, clf, images, labels):
        imgs = self.vgg_pre_process_image(images)
        results = clf.predict(imgs)
        predictions = []
        for result in results:
            try:
                label = labels[int(np.where(result == 1)[0])]
            except TypeError:
                label = "task"
                print(result.shape)
                print(np.where(result == 1))
                print("default type is task")
            predictions.append(label)
        return predictions

    def load_bcf_classifier(self):
        if self.bcf_classifier is None:
            bcf = BCF()
            bcf.CLASSIFIER_FILE = "../bcf/model_0/classifier_0_30_50"
            bcf.CODEBOOK_FILE = "../bcf/model/code_book_56_30.data"
            bcf.load_kmeans()
            bcf.load_classifier()
            self.bcf_classifier = bcf

    def load_bcf_56_classifier(self):
        if self.bcf_56_classifier is None:
            bcf = BCF()
            bcf.CLASSIFIER_FILE = "../bcf/model_0/classifier_56_0_30_50"
            bcf.CODEBOOK_FILE = "../bcf/model/code_book_56_30.data"
            bcf.load_kmeans()
            bcf.load_classifier()
            self.bcf_56_classifier = bcf

    def load_bcf_57_classifier(self):
        if self.bcf_57_classifier is None:
            bcf = BCF()
            bcf.CLASSIFIER_FILE = "../bcf/model_0/classifier_57_0_30_50"
            bcf.CODEBOOK_FILE = "../bcf/model/code_book_57_30.data"
            bcf.load_kmeans()
            bcf.load_classifier()
            self.bcf_57_classifier = bcf

    def classify_with_bcf(self, clf, images):
        imgs = self.bcf_pre_process_image(images)
        return clf.get_images_type(imgs)

    def classify(self, images, classifier_type):
        classifier_id = self.classifiers.index(classifier_type)
        if classifier_type.startswith("vgg"):
            self.load_vgg_classifier(classifier_type)
            predictions = self.classify_with_vgg_16(self.vgg_classifier, images, self.classes[classifier_id])
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
        elif type_info[1] == "57":
            data_dir = "../57_622data/"
        else:
            data_dir = "../{}_training_data/".format(img_type)

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

        test_res_file = "test_results/{}_{}_{}_test.txt".format(file_id, classifier_type, weights_suffix)
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
                    f.write(str(info) + "\n")
            record = "{}\t{},{},{}".format("all", all_total, all_correct, all_correct / all_total)
            print(record)
            f.write(record + "\n")


if __name__ == '__main__':

    classifier = Classifier()
    opt = sys.argv[1]

    if opt == "vgg":
        print("test vgg")
        c_type = "vgg16_52"
        c_id = classifier.classifiers.index(c_type)

        for img_type in ["png"]:
            weights_list = classifier.weights[c_id][img_type]
            for one_weights in weights_list[4:]:
                weights_suffix = "_".join(one_weights.split("_")[-2:])
                print(weights_suffix)
                classifier.test(c_type)
        # classifier.test("vgg16")
        # classifier.test("vgg16_56")
        # classifier.test("vgg16_57")

    # elif opt == "bcf":
    #     print("test bcf")
    #     classifier.test("bcf")
    #     classifier.test("bcf_56")
    #     classifier.test("bcf_57")
    else:
        print("wrong args, it should be vgg'")
