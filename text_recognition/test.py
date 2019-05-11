# -*- coding:utf-8 -*-
import pytesseract
from PIL import Image
import cv2 as cv
import numpy as np
import os
from helper import detector_helper as helper

image_dir = "../622data/test/businessRuleTask/"


def format_text(text):
    res = text.strip()
    res = res.replace("\n", " ")
    res = res.replace("  ", " ")
    return res


for name in os.listdir(image_dir)[25:30]:
    print(name)
    image_path = image_dir + name
    image = cv.imread(image_path)

    # text = pytesseract.image_to_string(Image.fromarray(image_mat))
    # text = format_text(text)
    #
    # print(text)
    # print("=" * 30)
    cv.imshow("img", helper.dilate_drawing(image))
    rotated_img = cv.transpose(image)
    rotated_img = cv.flip(rotated_img, 1)
    cv.imshow("rotated", helper.dilate_drawing(rotated_img))
    cv.waitKey(0)
    # np.rot
