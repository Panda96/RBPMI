# -*- coding:utf-8 -*-
import os

from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

import modeler


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
# image generator

data_gen = ImageDataGenerator(
    rescale=1. / 255,
    preprocessing_function=image_reverse
)
val_gen = ImageDataGenerator(
    rescale=1. / 255,
    preprocessing_function=image_reverse,
)

train_iter = data_gen.flow_from_directory(data_train, class_mode='categorical',
                                          target_size=img_shape, batch_size=16)

val_iter = val_gen.flow_from_directory(data_val, class_mode='categorical',
                                       target_size=img_shape, batch_size=16)

tune_model = modeler.get_vgg16_fc(img_size, classes)
tune_model.summary()

tune_model.compile(loss="categorical_crossentropy", optimizer=optimizers.RMSprop(lr=1e-4),
                   metrics=['acc'])

history = tune_model.fit_generator(
    generator=train_iter,
    steps_per_epoch=100,
    epochs=100,
    validation_data=val_iter,
    validation_steps=32
)
tune_model.save_weights("VGG16_fc_model.h5")
