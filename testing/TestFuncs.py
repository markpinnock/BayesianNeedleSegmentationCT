import numpy as np


def thresholdImg(im, threshold):
    im[im < threshold] = 0
    im[im >= threshold] = 1
    return im


def maskImg(im, ma_threshold):
    return np.ma.masked_where(im < ma_threshold, im)


def calcMeanEntropy(img_pred, img_entropy):
    return np.sum(img_entropy) / np.sum(np.logical_or(img_pred > 1e-3, img_entropy > 1e-3))


def diceLoss(pred, mask):
    numer = np.sum(pred * mask, axis=[1, 2, 3, 4]) * 2
    denom = np.sum(pred, axis=[1, 2, 3, 4]) + np.sum(mask, axis=[1, 2, 3, 4]) + 1e-6
    dice = numer / denom

    return 1 - np.mean(dice)


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