# -*- coding:utf-8 -*-
import os

from keras.models import Sequential, Model
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.applications.vgg16 import VGG16


def image_reverse(image):
    return 255 - image


data_dir = "120data/"

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
    preprocessing_function=image_reverse,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    fill_mode="constant",
    cval=255.)
val_gen = ImageDataGenerator(
    rescale=1. / 255,
    preprocessing_function=image_reverse,
)

train_iter = data_gen.flow_from_directory(data_train, class_mode='categorical',
                                          target_size=img_shape, batch_size=16)

val_iter = val_gen.flow_from_directory(data_val, class_mode='categorical',
                                       target_size=img_shape, batch_size=16)

base_model = VGG16(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))
base_model.summary()

out = base_model.layers[-1].output
out = layers.Flatten()(out)
out = layers.Dense(1024, activation='relu')(out)
# 因为前面输出的dense feature太多了，我们这里加入dropout layer来防止过拟合
out = layers.Dropout(0.5)(out)
out = layers.Dense(512, activation='relu')(out)
out = layers.Dropout(0.3)(out)
out = layers.Dense(classes, activation='softmax')(out)
tuneModel = Model(inputs=base_model.input, outputs=out)
tuneModel.summary()
# for layer in tuneModel.layers[:19]:  # freeze the base model only use it as feature extractors
#     layer.trainable = False
tuneModel.compile(loss="categorical_crossentropy", optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

history = tuneModel.fit_generator(
    generator=train_iter,
    steps_per_epoch=100,
    epochs=100,
    validation_data=val_iter,
    validation_steps=32
)
tuneModel.save_weights("VGG16_fc_model.h5")

acc = history.history['acc']
print(acc.shape)
val_acc = history.history['val_acc']
print(val_acc.shape)
loss = history.history['loss']
print(loss.shape)
val_loss = history.history['val_loss']
print(val_loss.shape)

# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, 101)
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'r', label='Validation acc')
# plt.legend()
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.legend()
# plt.show()
