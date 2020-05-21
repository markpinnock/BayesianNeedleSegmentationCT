import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


@tf.function
def diceLoss(pred, mask):
    numer = tf.reduce_sum(pred * mask, axis=[1, 2, 3, 4]) * 2
    denom = tf.reduce_sum(pred, axis=[1, 2, 3, 4]) + tf.reduce_sum(mask, axis=[1, 2, 3, 4]) + 1e-6
    dice = numer / denom

    return 1 - tf.reduce_mean(dice)


def haussdorfDistance(pred, mask, pix_sp):
    def dist(a, b, pix_sp):
        d = (a - b) * pix_sp
        return np.sqrt(d @ d.T)

    X = np.argwhere(pred[0, :, :, 0] == 1)
    Y = np.argwhere(mask[0, :, :, 0] == 1)

    if X.sum() == 0:
        return 1000

    d1 = max(min(dist(X[i, :], Y[j, :], pix_sp) for i in range(X.shape[0])) for j in range(Y.shape[0]))
    d2 = max(min(dist(X[i, :], Y[j, :], pix_sp) for j in range(Y.shape[0])) for i in range(X.shape[0]))

    return max(d1, d2)


def NMSCalc(pred, threshold, test=False):
    pred[pred < threshold] == 0
    pred[pred >= threshold] == 1
    pred = pred
    temp = np.argwhere(pred == 1.0)
    x = temp[:, 1][:, np.newaxis]
    y = temp[:, 0][:, np.newaxis]
    X = np.concatenate([np.ones(x.shape), x], axis=1)

    try:
        Xinv = np.linalg.inv(X.T @ X) @ X.T
    except np.linalg.LinAlgError:
        print(X.T @ X)
        return np.nan

    beta = Xinv @ y
    yhat = X @ beta
    xb = np.linspace(0, 511, 512)
    yb = beta[0] + beta[1] * xb
    err = y - yhat

    unit_vec = np.array([[xb[1] - xb[0]], [yb[1] - yb[0]]])
    unit_vec /= np.sqrt(unit_vec.T @ unit_vec)
    err_vec = np.concatenate([np.zeros(err.T.shape), err.T], axis=0)
    proju = (unit_vec.T @ err_vec) * unit_vec
    projv = err_vec - proju
    dist = np.sqrt(np.sum(projv * projv, axis=0))

    if test:
        test_size = err_vec.shape[1] // 10
        test_err = err_vec[:, 0::test_size]
        test_proju = proju[:, 0::test_size]
        test_projv = projv[:, 0::test_size]
        test_x = x[0::test_size]
        test_y = y[0::test_size]

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(pred, cmap='gray', origin='lower')
        plt.plot(xb, yb, 'r-')

        for i in range(test_x.shape[0]):
            plt.plot(
                np.concatenate([test_x[i] - test_err[0, i], test_x[i]]),
                np.concatenate([test_y[i] - test_err[1, i], test_y[i]]),
                c='y'
            )
        for i in range(test_x.shape[0]):
            plt.plot(
                np.concatenate([test_x[i] - test_projv[0, i], test_x[i]]),
                np.concatenate([test_y[i] - test_projv[1, i], test_y[i]]),
                c='g'
            )
        plt.axis('square')

        plt.subplot(1, 3, 2)
        plt.hist(err, bins=20)
        plt.axis('square')
        plt.title(f"Error: {err.std()}")

        plt.subplot(1, 3, 3)
        plt.hist(dist, bins=20)
        plt.axis('square')
        plt.title(f"Dist: {dist.std()}")

        plt.show()

    return dist.std()


@tf.function
def trainStep(imgs, segs, Model, ModelOptimiser):
    with tf.GradientTape() as tape:
        prediction = Model(imgs, training=True)
        loss = diceLoss(prediction, segs)
        gradients = tape.gradient(loss, Model.trainable_variables)
        ModelOptimiser.apply_gradients(zip(gradients, Model.trainable_variables))

        return loss


@tf.function
def valStep(imgs, labels, Model):
    prediction = Model(imgs, training=False)
    loss = diceLoss(prediction, labels)
    
    return loss


# DEACTIVATE BN AND DROPOUT SEPARATELY
def varDropout(imgs, Model, T, threshold):
    arr_dims = imgs.shape
    pred_array = np.zeros((arr_dims[0], arr_dims[1], arr_dims[2], arr_dims[3], T))

    for t in range(T):
        pred_array[:, :, :, :, t] = np.squeeze(Model(imgs, training=False).numpy())

    if threshold == None:
        pass
    else:
        pred_array[pred_array < threshold] = 0
        pred_array[pred_array >= threshold] = 1

    pred_mean = np.mean(pred_array, axis=4)
    # pred_var = np.var(pred_array, axis=4)
    class_1 = np.copy(pred_mean)
    class_2 = np.abs(1 - np.copy(pred_mean))
    p_vec = np.concatenate([class_1[:, :, :, :, np.newaxis], class_2[:, :, :, :, np.newaxis]], axis=4)
    entropies = -np.mean(p_vec * np.log(p_vec), axis=4)
    entropies[np.isnan(entropies)] = 0

    return pred_mean, entropies, pred_array
