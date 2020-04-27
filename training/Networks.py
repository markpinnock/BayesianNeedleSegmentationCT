import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


def dnResNetBlock(nc, inputlayer, drop_rate, dropout_flag):
    conv1 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same')(inputlayer)
    BN1 = tf.nn.relu(keras.layers.BatchNormalization()(conv1))
    BN1 = keras.layers.SpatialDropout3D(drop_rate)(BN1, training=dropout_flag)

    conv2 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same')(BN1)
    BN2 = tf.nn.relu(keras.layers.BatchNormalization()(conv2))
    BN2 = keras.layers.SpatialDropout3D(drop_rate)(BN2, training=dropout_flag)

    pool = keras.layers.MaxPool3D((2, 2, 1), strides=(2, 2, 1), padding='same')(BN2)
    return BN1, BN2, pool


def upResNetBlock(nc, inputlayer, skip, tconv_strides, drop_rate, dropout_flag):
    tconv = keras.layers.Conv3DTranspose(nc, (3, 3, 3), strides=tconv_strides, padding='same')(inputlayer)
    BN1 = tf.nn.relu(keras.layers.BatchNormalization()(tconv))
    BN1 = keras.layers.SpatialDropout3D(drop_rate)(BN1, training=dropout_flag)

    concat = keras.layers.concatenate([BN1, skip], axis=4)
    conv1 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same')(concat)
    BN2 = tf.nn.relu(keras.layers.BatchNormalization()(conv1))
    BN2 = keras.layers.SpatialDropout3D(drop_rate)(BN2, training=dropout_flag)

    conv2 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same')(BN2)
    BN3 = tf.nn.relu(keras.layers.BatchNormalization()(conv2))
    BN3 = keras.layers.SpatialDropout3D(drop_rate)(BN3, training=dropout_flag)

    return BN1, BN2, BN3


def UNetGen(input_shape, starting_channels, drop_rate=0.0, dropout_flag=True, return_all=False):
    inputlayer = keras.layers.Input(shape=input_shape)
    nc = starting_channels

    conv1_1, conv2_1, dnres1 = dnResNetBlock(nc, inputlayer, drop_rate, dropout_flag)
    conv1_2, conv2_2, dnres2 = dnResNetBlock(nc * 2, dnres1, drop_rate, dropout_flag)
    conv1_3, conv2_3, dnres3 = dnResNetBlock(nc * 4, dnres2, drop_rate, dropout_flag)
    conv1_4, conv2_4, dnres4 = dnResNetBlock(nc * 8, dnres3, drop_rate, dropout_flag)

    dn5 = keras.layers.Conv3D(nc * 16, (3, 3, 3), strides=(1, 1, 1), padding='same')(dnres4)
    BN5 = tf.nn.relu(keras.layers.BatchNormalization()(dn5))
    BN5 = keras.layers.SpatialDropout3D(drop_rate)(BN5, training=dropout_flag)

    upres6, conv1_6, conv2_6 = upResNetBlock(nc * 8, BN5, conv2_4, (2, 2, 1), drop_rate, dropout_flag)
    upres7, conv1_7, conv2_7 = upResNetBlock(nc * 4, upres6, conv2_3, (2, 2, 1), drop_rate, dropout_flag)
    upres8, conv1_8, conv2_8 = upResNetBlock(nc * 2, upres7, conv2_2, (2, 2, 1), drop_rate, dropout_flag)
    upres9, conv1_9, conv2_9 = upResNetBlock(nc, upres8, conv2_1, (2, 2, 1), drop_rate, dropout_flag)

    outputlayer = keras.layers.Conv3D(1, (1, 1, 1), strides=(1, 1, 1), padding='same', activation='sigmoid')(upres9)

    if return_all:
        output_list = [conv1_1, conv2_1, dnres1, conv1_2, conv2_2, dnres2, conv1_3, conv2_3, dnres3,
                       conv1_4, conv2_4, dnres4, BN5, upres6, conv1_6, conv2_6,
                       upres7, conv1_6, conv2_6, upres8, conv1_8, conv2_8, upres9, conv1_9, conv2_9, outputlayer]
    else:
        output_list = outputlayer

    return keras.Model(inputs=inputlayer, outputs=output_list)

