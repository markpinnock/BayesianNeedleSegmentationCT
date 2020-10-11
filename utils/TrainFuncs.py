import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


@tf.function
def dice_loss(pred, mask):

    """ Implements Dice loss
        - pred: predicted segmentation
        - mask: ground truth label """

    numer = tf.reduce_sum(pred * mask, axis=[1, 2, 3, 4]) * 2
    denom = tf.reduce_sum(pred, axis=[1, 2, 3, 4]) + tf.reduce_sum(mask, axis=[1, 2, 3, 4]) + 1e-6
    dice = numer / denom

    return 1 - tf.reduce_mean(dice)


@tf.function
def train_step(imgs, segs, Model, ModelOptimiser):

    """ Implements training step
        - imgs: input images
        - segs: segmentation labels
        - Model: model to be trained (keras.Model)
        - ModelOptimiser: e.g. keras.optimizers.Adam() """

    # TODO: subclass UNet and convert train_step to class method
    with tf.GradientTape() as tape:
        prediction = Model(imgs, training=True)
        loss = dice_loss(prediction, segs)
        gradients = tape.gradient(loss, Model.trainable_variables)
        ModelOptimiser.apply_gradients(zip(gradients, Model.trainable_variables))

        return loss


@tf.function
def val_step(imgs, labels, Model):
    """ Implements validation step """

    # TODO: subclass UNet and convert val_step to class method
    prediction = Model(imgs, training=False)
    loss = dice_loss(prediction, labels)
    
    return loss


# TODO DEACTIVATE BN AND DROPOUT SEPARATELY
# TODO CONSIDER ADDING AS METHOD ONCE UNet CONVERTED TO CLASS
def var_dropout(imgs, Model, T, threshold):

    """ Implement variational dropout (to be run at inference)
        - imgs: input images for model inference
        - Model: model of type keras.Model
        - T: number of forward passes
        - threshold: can threshold predictions if needed [0, 1]
        
        Returns: mean prediction, entropy of predictions, predictions """

    # Create array of forward passes, dims (mb, w, h, d, T)
    arr_dims = imgs.shape
    pred_array = np.zeros((arr_dims[0], arr_dims[1], arr_dims[2], arr_dims[3], T))

    for t in range(T):
        pred_array[:, :, :, :, t] = np.squeeze(Model(imgs, training=False).numpy())

    if threshold == None:
        pass
    else:
        pred_array[pred_array < threshold] = 0
        pred_array[pred_array >= threshold] = 1

    # Mean of predictions and entropy (uncertainty of predictions)
    pred_mean = np.mean(pred_array, axis=4)
    class_1 = np.copy(pred_mean)
    class_2 = np.abs(1 - np.copy(pred_mean))
    p_vec = np.concatenate([class_1[:, :, :, :, np.newaxis], class_2[:, :, :, :, np.newaxis]], axis=4)
    p_vec[p_vec == 0.0] = 1e-8 # Constant for numerical stability
    # TODO: CHANGE TO SUM
    entropies = -np.mean(p_vec * np.log(p_vec), axis=4)

    pred_mean[pred_mean < 1e-7] = 0
    entropies[entropies < 1e-7] = 0
    pred_array[pred_array < 1e-7] = 0

    return pred_mean, entropies, pred_array
