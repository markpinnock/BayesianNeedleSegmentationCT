import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


def dnResNetBlock(nc, inputlayer, drop_rate, dropout_type, dropout_flag):
    conv1 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same')(inputlayer)
    BN1 = tf.nn.relu(keras.layers.BatchNormalization()(conv1))

    if dropout_type == 'standard':
        BN1 = keras.layers.Dropout(drop_rate)(BN1, training=dropout_flag)
    elif dropout_type == 'spatial':
        BN1 = keras.layers.SpatialDropout3D(drop_rate)(BN1, training=dropout_flag)
    else:
        pass

    conv2 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same')(BN1)
    BN2 = tf.nn.relu(keras.layers.BatchNormalization()(conv2))

    if dropout_type == 'standard':
        BN2 = keras.layers.Dropout(drop_rate)(BN2, training=dropout_flag)
    elif dropout_type == 'spatial':
        BN2 = keras.layers.SpatialDropout3D(drop_rate)(BN2, training=dropout_flag)
    else:
        pass

    pool = keras.layers.MaxPool3D((2, 2, 1), strides=(2, 2, 1), padding='same')(BN2)
    return BN2, pool


def upResNetBlock(nc, inputlayer, skip, tconv_strides, drop_rate, dropout_type, dropout_flag):
    tconv = keras.layers.Conv3DTranspose(nc, (3, 3, 3), strides=tconv_strides, padding='same')(inputlayer)
    BN1 = tf.nn.relu(keras.layers.BatchNormalization()(tconv))

    if dropout_type == 'standard':
        BN1 = keras.layers.Dropout(drop_rate)(BN1, training=dropout_flag)
    elif dropout_type == 'spatial':
        BN1 = keras.layers.SpatialDropout3D(drop_rate)(BN1, training=dropout_flag)
    else:
        pass

    concat = keras.layers.concatenate([BN1, skip], axis=4)
    conv1 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same')(concat)
    BN2 = tf.nn.relu(keras.layers.BatchNormalization()(conv1))

    if dropout_type == 'standard':
        BN2 = keras.layers.Dropout(drop_rate)(BN2, training=dropout_flag)
    elif dropout_type == 'spatial':
        BN2 = keras.layers.SpatialDropout3D(drop_rate)(BN2, training=dropout_flag)
    else:
        pass

    conv2 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same')(BN2)
    BN3 = tf.nn.relu(keras.layers.BatchNormalization()(conv2))

    if dropout_type == 'standard':
        BN3 = keras.layers.Dropout(drop_rate)(BN3, training=dropout_flag)
    elif dropout_type == 'spatial':
        BN3 = keras.layers.SpatialDropout3D(drop_rate)(BN3, training=dropout_flag)
    else:
        pass

    return BN3


def UNetGen(input_shape, starting_channels, drop_rate=0.0, dropout_type=None, dropout_flag=True):
    inputlayer = keras.layers.Input(shape=input_shape)
    nc = starting_channels

    skip1, dnres1 = dnResNetBlock(nc, inputlayer, drop_rate, dropout_type, dropout_flag)
    skip2, dnres2 = dnResNetBlock(nc * 2, dnres1, drop_rate, dropout_type, dropout_flag)
    skip3, dnres3 = dnResNetBlock(nc * 4, dnres2, drop_rate, dropout_type, dropout_flag)
    skip4, dnres4 = dnResNetBlock(nc * 8, dnres3, drop_rate, dropout_type, dropout_flag)
    dn5 = keras.layers.Conv3D(nc * 16, (3, 3, 3), strides=(1, 1, 1), padding='same')(dnres4)
    BN5 = tf.nn.relu(keras.layers.BatchNormalization()(dn5))

    if dropout_type == 'standard':
        BN5 = keras.layers.Dropout(drop_rate)(BN5, training=dropout_flag)
    elif dropout_type == 'spatial':
        BN5 = keras.layers.SpatialDropout3D(drop_rate)(BN5, training=dropout_flag)
    else:
        pass

    upres4 = upResNetBlock(nc * 8, BN5, skip4, (2, 2, 1), drop_rate, dropout_type, dropout_flag)
    upres3 = upResNetBlock(nc * 4, upres4, skip3, (2, 2, 1), drop_rate, dropout_type, dropout_flag)
    upres2 = upResNetBlock(nc * 2, upres3, skip2, (2, 2, 1), drop_rate, dropout_type, dropout_flag)
    upres1 = upResNetBlock(nc, upres2, skip1, (2, 2, 1), drop_rate, dropout_type, dropout_flag)

    outputlayer = keras.layers.Conv3D(1, (1, 1, 1), strides=(1, 1, 1), padding='same', activation='sigmoid')(upres1)

    return keras.Model(inputs=inputlayer, outputs=outputlayer)

