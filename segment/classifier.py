# -*- coding:utf-8 -*-
from image_classification import modeler
from bcf.bcf import BCF

import cv2 as cv
import numpy as np
import os


class Classifier:
    def __init__(self):
        self.vgg_16_classifier = None
        self.bcf_classifier = None
        self.img_size = 150
        self.classes = os.listdir("../622data/train/")

    def load_vgg_16_classifier(self):
        if self.vgg_16_classifier is None:
            self.vgg_16_classifier = modeler.get_vgg16_fc(self.img_size, len(self.classes))
            self.vgg_16_classifier.load_weights("../image_classification/weights/VGG16_fc_model.h5")

    def load_bcf_classifier(self):
        if self.bcf_classifier is None:
            self.bcf_classifier = BCF()
            self.bcf_classifier.load_classifier()

    def classify_with_vgg_16(self, image):
        self.load_vgg_16_classifier()
        if type(image) == str:
            img = cv.imread(image)
        else:
            img = image
        img = 255 - img
        img_shape = (self.img_size, self.img_size)
        img = cv.resize(img, img_shape)
        img = img.reshape((1,) + img.shape)
        result = self.vgg_16_classifier.predict(img, batch_size=1)
        label = self.classes[int(np.where(result == 1)[1])]
        return label


if __name__ == '__main__':
    image_path = "../622data/test/dataObjectReference/4034_019_01.png"
    image = cv.imread(image_path)
    classifier = Classifier()
    res = classifier.classify_with_vgg_16(image_path)
    print(res)
    res = classifier.classify_with_vgg_16(image)
    print(res)

