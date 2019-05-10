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

tune_model = modeler.get_inception_v3_fc(img_size, classes)
tune_model.summary()
# for layer in tune_model.layers[:19]:  # freeze the base model only use it as feature extractors
#     layer.trainable = False
tune_model.compile(loss="categorical_crossentropy", optimizer=optimizers.RMSprop(lr=1e-4),
                   metrics=['acc'])

history = tune_model.fit_generator(
    generator=train_iter,
    steps_per_epoch=100,
    epochs=100,
    validation_data=val_iter,
    validation_steps=32
)
tune_model.save_weights("Inception_v3_fc_model_2.h5")

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

for i in range(len(acc)):
    print("epoch_{},{},{},{},{}".format(i+1, loss[i], acc[i], val_loss[i], val_acc[i]))
