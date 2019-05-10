# -*- coding:utf-8 -*-
from keras.models import Sequential, Model
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3


def get_vgg16_fc(img_size, classes):
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))

    out = base_model.layers[-1].output
    out = layers.Flatten()(out)
    out = layers.Dense(1024, activation='relu')(out)
    # 因为前面输出的dense feature太多了，我们这里加入dropout layer来防止过拟合
    out = layers.Dropout(0.5)(out)
    out = layers.Dense(512, activation='relu')(out)
    out = layers.Dropout(0.3)(out)
    out = layers.Dense(classes, activation='softmax')(out)
    tune_model = Model(inputs=base_model.input, outputs=out)
    # tune_model.summary()
    return tune_model


def get_inception_v3_fc(img_size, classes):
    base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))

    out = base_model.output
    out = layers.GlobalAveragePooling2D()(out)
    out = layers.Dense(1024, activation='relu')(out)
    # 因为前面输出的dense feature太多了，我们这里加入dropout layer来防止过拟合
    out = layers.Dropout(0.5)(out)
    out = layers.Dense(512, activation='relu')(out)
    out = layers.Dropout(0.3)(out)
    out = layers.Dense(classes, activation='softmax')(out)
    tune_model = Model(inputs=base_model.input, outputs=out)
    # tune_model.summary()
    return tune_model
