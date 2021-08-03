from keras.models import Model
from keras.layers import Input, Conv2D, MaxPool2D, Dropout, UpSampling2D, Concatenate
from keras.optimizers import Adam
from keras import backend as K
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
import tensorflow as tf


def weighted_binary_crossentropy(sample_weight):
    def loss(y_true, y_pred):
        bce = tf.keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE)
        return bce(y_true, y_pred, sample_weight=sample_weight)
    return loss


def unet(input_size=(256, 512, 1), dropout=0.5):
    '''
    Implementation of U-Net using a weighted loss function.
    '''

    inp = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(inp)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv2)
    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv3)
    pool3 = MaxPool2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(dropout)(conv4)
    pool4 = MaxPool2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(dropout)(conv5)

    up6 = Conv2D(
        512,
        2,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(
        UpSampling2D(
            size=(
                2,
                2))(drop5))
    merge6 = Concatenate(axis=-1)([conv4, up6])
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(
        256,
        2,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(
        UpSampling2D(
            size=(
                2,
                2))(conv6))
    merge7 = Concatenate(axis=-1)([conv3, up7])
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(
        128,
        2,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(
        UpSampling2D(
            size=(
                2,
                2))(conv7))
    merge8 = Concatenate(axis=-1)([conv2, up8])
    conv8 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(
        64,
        2,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(
        UpSampling2D(
            size=(
                2,
                2))(conv8))
    merge9 = Concatenate(axis=-1)([conv1, up9])
    conv9 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    #conv9 = Conv2D(128, 3, activation='relu', padding='same',
    #               kernel_initializer='he_normal')(merge9)
    #conv9 = Conv2D(128, 3, activation='relu', padding='same',
    #               kernel_initializer='he_normal')(conv9)
    #conv9 = Conv2D(2, 3, activation='relu', padding='same',
    #               kernel_initializer='he_normal')(conv9)
    #conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    weights_tensor = Input(input_size)
    w_ce = weighted_binary_crossentropy(weights_tensor)
    model = Model(inputs=[inp, weights_tensor], outputs=conv10)
    model.compile(optimizer=Adam(lr=1e-4), loss=w_ce, metrics=['accuracy'])

    return model


def unet_wo_weighting(input_size=(256, 512, 1), dropout=0.5):
    '''
    Implementation of U-Net using NOT a weighted loss function.
    '''

    inp = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(inp)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv2)
    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv3)
    pool3 = MaxPool2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(dropout)(conv4)
    pool4 = MaxPool2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(dropout)(conv5)

    up6 = Conv2D(
        512,
        2,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(
        UpSampling2D(
            size=(
                2,
                2))(drop5))
    merge6 = Concatenate(axis=-1)([conv4, up6])
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(
        256,
        2,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(
        UpSampling2D(
            size=(
                2,
                2))(conv6))
    merge7 = Concatenate(axis=-1)([conv3, up7])
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(
        128,
        2,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(
        UpSampling2D(
            size=(
                2,
                2))(conv7))
    merge8 = Concatenate(axis=-1)([conv2, up8])
    conv8 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(
        64,
        2,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(
        UpSampling2D(
            size=(
                2,
                2))(conv8))
    merge9 = Concatenate(axis=-1)([conv1, up9])
    conv9 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inp, outputs=conv10)
    model.compile(
        optimizer=Adam(
            lr=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy'])

    return model


def unet_reduced(input_size=(256, 512, 1), dropout=0.5):
    '''
    Implementation of a reduced version of the U-Net.
    '''

    inp = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(inp)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)

    conv5 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv5 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(dropout)(conv5)

    up9 = Conv2D(
        64,
        2,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(
        UpSampling2D(
            size=(
                2,
                2))(drop5))
    merge9 = Concatenate(axis=-1)([conv1, up9])
    conv9 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    weights_tensor = Input(input_size)
    w_ce = weighted_binary_crossentropy(weights_tensor)

    model = Model(inputs=[inp, weights_tensor], outputs=conv10)
    model.compile(optimizer=Adam(lr=1e-4), loss=w_ce, metrics=['accuracy'])

    return model


def unet_inference(input_size=(256, 512, 1), dropout=0.5):
    '''
    Implementation of the U-Net for prediction.
    '''

    inp = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(inp)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv2)
    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv3)
    pool3 = MaxPool2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(dropout)(conv4)
    pool4 = MaxPool2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(dropout)(conv5)

    up6 = Conv2D(
        512,
        2,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(
        UpSampling2D(
            size=(
                2,
                2))(drop5))
    merge6 = Concatenate(axis=-1)([conv4, up6])
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(
        256,
        2,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(
        UpSampling2D(
            size=(
                2,
                2))(conv6))
    merge7 = Concatenate(axis=-1)([conv3, up7])
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(
        128,
        2,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(
        UpSampling2D(
            size=(
                2,
                2))(conv7))
    merge8 = Concatenate(axis=-1)([conv2, up8])
    conv8 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(
        64,
        2,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(
        UpSampling2D(
            size=(
                2,
                2))(conv8))
    merge9 = Concatenate(axis=-1)([conv1, up9])
    conv9 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
#     merge9 = Concatenate(axis=-1)([conv1, up9])
#     conv9 = Conv2D(128, 3, activation='relu', padding='same',
#                    kernel_initializer='he_normal')(merge9)
#     conv9 = Conv2D(128, 3, activation='relu', padding='same',
#                    kernel_initializer='he_normal')(conv9)
#     conv9 = Conv2D(2, 3, activation='relu', padding='same',
#                    kernel_initializer='he_normal')(conv9)
#     conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inp, outputs=conv10)
    model.compile(
        optimizer=Adam(
            lr=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy'])

    return model


def unet_reduced_inference(input_size=(256, 512, 1), dropout=0.5):
    '''
    Implementation of a reduced version of the U-Net for the prediction.
    '''

    inp = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(inp)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)

    conv5 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv5 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(dropout)(conv5)

    up9 = Conv2D(
        64,
        2,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(
        UpSampling2D(
            size=(
                2,
                2))(drop5))
    merge9 = Concatenate(axis=-1)([conv1, up9])
    conv9 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inp, outputs=conv10)
    model.compile(
        optimizer=Adam(
            lr=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy'])

    return model
