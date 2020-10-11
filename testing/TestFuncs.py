import matplotlib.pyplot as plt
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
    numer = np.sum(pred * mask) * 2
    denom = np.sum(pred) + np.sum(mask) + 1e-6
    dice = numer / denom

    return 1 - dice


def haussdorfDistance(pred, mask, pix_sp):

    def dist(a, b, pix_sp):
        d = (a - b) * pix_sp
        return np.sqrt(d @ d.T)

    X = np.argwhere(pred[:, :, 1] == 1)
    Y = np.argwhere(mask[:, :, 1] == 1)

    if X.sum() == 0:
        return np.nan

    d1 = max(min(dist(X[i, :], Y[j, :], pix_sp) for i in range(X.shape[0])) for j in range(Y.shape[0]))
    d2 = max(min(dist(X[i, :], Y[j, :], pix_sp) for j in range(Y.shape[0])) for i in range(X.shape[0]))

    return max(d1, d2)


def NMSCalc(pred, threshold, pix_sp, test=False):
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
    err_vec *= pix_sp
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
        plt.subplot(2, 2, 1)
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

        plt.subplot(2, 2, 2)
        plt.hist(dist, bins=20)
        plt.axis('square')
        plt.title(f" Proj error: {(dist.T @ dist) / dist.shape[0]}")

        plt.subplot(2, 2, 3)
        plt.hist(projv[0, :], bins=20)
        plt.axis('square')
        plt.title(f"x error: {(projv[0, :].T @ projv[0, :]) / projv.shape[1]}")

        plt.subplot(2, 2, 4)
        plt.hist(projv[1, :], bins=20)
        plt.axis('square')
        plt.title(f" y error: {(projv[1, :].T @ projv[1, :]) / projv.shape[1]}")

        plt.show()

    assert np.isclose(((projv[0, :].T @ projv[0, :]) + (projv[1, :].T @ projv[1, :])) / projv.shape[1] - ((dist.T @ dist)) / dist.shape[0], 0.0)

    return (dist.T @ dist) / dist.shape[0]

def trajErrorCalc(pred, mask, threshold, pix_sp, test=False):

    def dist(a, b, pix_sp):
        d = (a - b) * pix_sp
        return np.sqrt(d @ d.T)

    pred[pred < threshold] == 0
    pred[pred >= threshold] == 1

    temp = np.argwhere(pred == 1.0)
    xp = temp[:, 1][:, np.newaxis]
    yp = temp[:, 0][:, np.newaxis]
    Xp = np.concatenate([np.ones(xp.shape), xp], axis=1)

    try:
        Xpinv = np.linalg.inv(Xp.T @ Xp) @ Xp.T
    except np.linalg.LinAlgError:
        print(Xp.T @ Xp)
        return (np.nan, np.nan, np.nan)

    betap = Xpinv @ yp
    xb = np.linspace(0, 511, 512)
    ybp = betap[0] + betap[1] * xb

    temp = np.argwhere(mask == 1.0)
    xg = temp[:, 1][:, np.newaxis]
    yg = temp[:, 0][:, np.newaxis]
    Xg = np.concatenate([np.ones(xg.shape), xg], axis=1)
    Xginv = np.linalg.inv(Xg.T @ Xg) @ Xg.T
    betag = Xginv @ yg
    ybg = betag[0] + betag[1] * xb

    anglep = np.arctan((xb[-1] - xb[0]) / (ybp[-1] - ybp[0])) / np.pi * 180
    angleg = np.arctan((xb[-1] - xb[0]) / (ybg[-1] - ybg[0])) / np.pi * 180

    xbt = xb - 255.5
    ybgt = ybg - 255.5
    ybpt = ybp - 255.5

    dgt = np.sqrt(xbt ** 2 + ybgt ** 2)
    dpt = np.sqrt(xbt ** 2 + ybpt ** 2)
    idxg = np.argmin(dgt)
    idxp = np.argmin(dpt)

    A = np.vstack([xb, ybp]).T
    B = np.vstack([xb, ybg]).T

    d1 = max(min(dist(A[i, :], B[j, :], pix_sp) for i in range(A.shape[0])) for j in range(B.shape[0]))
    d2 = max(min(dist(A[i, :], B[j, :], pix_sp) for j in range(B.shape[0])) for i in range(A.shape[0]))

    if test:
        print(anglep, angleg, anglep - angleg, dpt[idxp] - dgt[idxg], max(d1, d2))
        plt.figure()
        plt.imshow(pred, cmap='gray', origin='lower')
        plt.plot(xb, ybp, 'r-')
        plt.plot(xb, ybg, 'y-')
        plt.plot(xb[idxg], ybg[idxg], 'g+')
        plt.plot(xb[idxp], ybp[idxp], 'g+')
        plt.plot(255.5, 255.5, 'w+')
        plt.show()

    return (anglep - angleg, dpt[idxp] - dgt[idxg], max(d1, d2))