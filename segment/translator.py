# -*- coding:utf-8 -*-
from PIL import Image
import pytesseract
import cv2 as cv


def format_text(text):
    res = text.strip()
    res = res.replace("\n", " ")
    res = res.replace("  ", " ")
    return res


def translate(image):
    if type(image) == str:
        image = cv.imread(image)
    text = pytesseract.image_to_string(Image.fromarray(image))
    text = format_text(text)
    return text
