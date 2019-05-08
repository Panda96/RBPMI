# -*- coding:utf-8 -*-
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def image_reverse(image):
    return 255 - image


data_dir = "120data/"

data_train = data_dir + "train/"
data_test = data_dir + "test/"
data_val = data_dir + "val/"


data_gen = ImageDataGenerator(
    preprocessing_function=image_reverse,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    fill_mode="constant",
    cval=255.)

one_image = data_val + "businessRuleTask/984_004_09.jpg"
# one_image = "cat.jpg"

img = load_img(one_image)
# print(img.shape)
x = img_to_array(img)
print(x.shape)
x = x.reshape((1,) + x.shape)
print(x.shape)

i = 0
for batch in data_gen.flow(x, batch_size=1, save_to_dir="test_data", save_format="jpeg"):
    i += 1
    if i > 10:
        break

# data_gen.flow_from_directory(data_val, save_to_dir=aug_data_val)
