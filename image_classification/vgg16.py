# -*- coding:utf-8 -*-
import os

from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

import modeler


def image_reverse(image):
    return 255 - image


def train(all_data_dir, model_id):
    print(model_id)
    data_train = all_data_dir + "train/"
    # data_test = all_data_dir + "test/"
    data_val = all_data_dir + "val/"

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
        preprocessing_function=image_reverse,
    )

    train_iter = data_gen.flow_from_directory(data_train, class_mode='categorical',
                                              target_size=img_shape, batch_size=16)

    val_iter = val_gen.flow_from_directory(data_val, class_mode='categorical',
                                           target_size=img_shape, batch_size=16)

    tune_model = modeler.get_vgg16_fc(img_size, classes)
    # tune_model.summary()

    tune_model.compile(loss="categorical_crossentropy", optimizer=optimizers.RMSprop(lr=1e-4),
                       metrics=['acc'])

    history = tune_model.fit_generator(
        generator=train_iter,
        steps_per_epoch=100,
        epochs=100,
        validation_data=val_iter,
        validation_steps=32
    )
    weights_name = "png_weights/VGG16_fc_model_{}.h5".format(model_id)
    tune_model.save_weights(weights_name)

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    model_log = "model_log/"
    if not os.path.exists(model_log):
        os.mkdir(model_log)

    log_name = "VGG16_fc_model_{}.log".format(model_id)
    log_file = model_log + log_name

    with open(log_file, "w") as f:
        for i in range(len(acc)):
            epoch_info = "epoch_{},{},{},{},{}".format(i + 1, loss[i], acc[i], val_loss[i], val_acc[i])
            print(epoch_info)
            f.write(epoch_info + "\n")


if __name__ == '__main__':

    data_dir = "../png_training_data/"

    for model_i in range(20):
        model__id = "png_{}".format(model_i)
        train(data_dir, model__id)
