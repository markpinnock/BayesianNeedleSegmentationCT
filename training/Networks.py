import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


def down_block(nc, inputlayer, drop_rate, dropout_flag):

    """ Implements downsampling block for 3D UNet
        - nc: number of channels in block
        - inputlayer: input of type keras.layers.Input
        - drop_rate: dropout rate [0, 1]
        - dropout_flag: add dropout or not (True/False)

        Returns: output before (skip) and after pooling """

    # TODO: REMOVE BATCHNORM GIVEN SMALL MB SIZES AND TRY E.G. INSTANCE/GROUP NORM
    conv1 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same')(inputlayer)
    BN1 = tf.nn.relu(keras.layers.BatchNormalization()(conv1))
    BN1 = keras.layers.Dropout(drop_rate)(BN1, training=dropout_flag)

    conv2 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same')(BN1)
    BN2 = tf.nn.relu(keras.layers.BatchNormalization()(conv2))
    BN2 = keras.layers.Dropout(drop_rate)(BN2, training=dropout_flag)

    # TODO: CONSIDER CHANGING TO CONV STRIDE 2
    pool = keras.layers.MaxPool3D((2, 2, 1), strides=(2, 2, 1), padding='same')(BN2)
    return BN2, pool


def up_block(nc, inputlayer, skip, tconv_strides, drop_rate, dropout_flag):
    
    """ Implements upsampling block for 3D UNet
        - nc: number of channels in block
        - inputlayer: input of type keras.layers.Input
        - skip: skip_layer from encoder
        - tconv_strides: number of strides for transpose convolution (a, b, c)
        - drop_rate: dropout rate [0, 1]
        - dropout_flag: add dropout or not (True/False) """
    
    tconv = keras.layers.Conv3DTranspose(nc, (3, 3, 3), strides=tconv_strides, padding='same')(inputlayer)
    BN1 = tf.nn.relu(keras.layers.BatchNormalization()(tconv))
    BN1 = keras.layers.Dropout(drop_rate)(BN1, training=dropout_flag)

    concat = keras.layers.concatenate([BN1, skip], axis=4)
    conv1 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same')(concat)
    BN2 = tf.nn.relu(keras.layers.BatchNormalization()(conv1))
    BN2 = keras.layers.Dropout(drop_rate)(BN2, training=dropout_flag)

    conv2 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same')(BN2)
    BN3 = tf.nn.relu(keras.layers.BatchNormalization()(conv2))
    BN3 = keras.layers.Dropout(drop_rate)(BN3, training=dropout_flag)

    return BN3


def UNetGen(input_shape, nc, drop_rate=0.0, dropout_flag=True):
    
    """ Implements UNet
        - input_shape: size of input volumes
        - nc: number of channels in first block
        - drop_rate: dropout rate [0, 1]
        - dropout_flag: add dropout or not (True/False) """

    # TODO: SUBCLASS keras.Model
    inputlayer = keras.layers.Input(shape=input_shape)

    # Encoder
    skip1, dnres1 = down_block(nc, inputlayer, drop_rate, dropout_flag)
    skip2, dnres2 = down_block(nc * 2, dnres1, drop_rate, dropout_flag)
    skip3, dnres3 = down_block(nc * 4, dnres2, drop_rate, dropout_flag)
    skip4, dnres4 = down_block(nc * 8, dnres3, drop_rate, dropout_flag)
    dn5 = keras.layers.Conv3D(nc * 16, (3, 3, 3), strides=(1, 1, 1), padding='same')(dnres4)
    BN5 = tf.nn.relu(keras.layers.BatchNormalization()(dn5))
    drop5 = keras.layers.Dropout(drop_rate)(BN5, training=dropout_flag)

    # Decoder
    upres4 = up_block(nc * 8, drop5, skip4, (2, 2, 1), drop_rate, dropout_flag)
    upres3 = up_block(nc * 4, upres4, skip3, (2, 2, 1), drop_rate, dropout_flag)
    upres2 = up_block(nc * 2, upres3, skip2, (2, 2, 1), drop_rate, dropout_flag)
    upres1 = up_block(nc, upres2, skip1, (2, 2, 1), drop_rate, dropout_flag)

    output_layer = keras.layers.Conv3D(1, (1, 1, 1), strides=(1, 1, 1), padding='same', activation='sigmoid')(upres1)
    # Necessary for examining feature maps
    # TODO: CONVERT TO BOOLEAN
    # output_list = [skip1, skip2, skip3, skip4, drop5, upres4, upres3, upres2, upres1, outputlayer]

    return keras.Model(inputs=inputlayer, outputs=output_layer)
