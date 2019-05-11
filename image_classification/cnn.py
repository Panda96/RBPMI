# -*- coding:utf-8 -*-
# -*- coding:utf-8 -*-
import os
import numpy as np
from keras.models import Sequential, Model
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.utils.np_utils import to_categorical
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt


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
# image_mat generator

data_gen = ImageDataGenerator(
    rescale=1. / 255,
    preprocessing_function=image_reverse
)
val_gen = ImageDataGenerator(
    rescale=1. / 255,
    preprocessing_function=image_reverse
)

train_iter = data_gen.flow_from_directory(data_train, class_mode='categorical',
                                          target_size=img_shape, batch_size=16)

val_iter = val_gen.flow_from_directory(data_val, class_mode='categorical',
                                       target_size=img_shape, batch_size=16)

model = Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(classes, activation='softmax'))
model.summary()
model.compile(loss="categorical_crossentropy", optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

history = model.fit_generator(
    generator=train_iter,
    steps_per_epoch=100,
    epochs=100,
    validation_data=val_iter,
    validation_steps=32
)
model.save_weights("CNN_fc_model.h5")

acc = history.history['acc']
print(acc.shape)
val_acc = history.history['val_acc']
print(val_acc.shape)
loss = history.history['loss']
print(loss.shape)
val_loss = history.history['val_loss']
print(val_loss.shape)

# epochs = range(1, 101)
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'r', label='Validation acc')
# plt.legend()
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.legend()
# plt.show()
