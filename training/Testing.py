import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow.keras as keras

sys.path.append('..')

from Networks import UNetGen
from utils.DataLoader import imgLoader
from utils.TrainFuncs import diceLoss, NMSCalc, varDropout


MB_SIZE = 4
NC = 4
EPOCHS = 500
ETA = 0.001
DROP_RATE = 0.2
T = 20

FILE_PATH = "C:/Users/roybo/OneDrive - University College London/Collaborations/RobotNeedleSeg/Code/003_CNN_Bayes_Traj/"
DATA_PATH = "Z:/Robot_Data/Test/"
# EXPT_NAME = "spatial_dropout"
EXPT_NAME = f"nc{NC}_ep{EPOCHS}_eta{ETA}"
# EXPT_NAME = "nc4_drop0_5"
MODEL_SAVE_PATH = f"{FILE_PATH}models/{EXPT_NAME}/"
img_path = f"{DATA_PATH}Img/"
seg_path = f"{DATA_PATH}Seg/"
imgs = os.listdir(img_path)
segs = os.listdir(seg_path)
imgs.sort()
segs.sort()

N = len(imgs)
assert N == len(segs), "HI/LO IMG PAIRS UNEVEN LENGTHS"

LO_VOL_SIZE = (512, 512, 3, 1, )

test_ds = tf.data.Dataset.from_generator(
    imgLoader, args=[img_path, seg_path, imgs, segs, False, False], output_types=(tf.float32, tf.float32))

UNetStandard = UNetGen(input_shape=LO_VOL_SIZE, starting_channels=NC, drop_rate=0.0, dropout_flag=False)
UNetStandard.load_weights(f"{MODEL_SAVE_PATH}{EXPT_NAME}")
UNetDropout = UNetGen(input_shape=LO_VOL_SIZE, starting_channels=NC, drop_rate=DROP_RATE, dropout_flag=True)
UNetDropout.load_weights(f"{MODEL_SAVE_PATH}{EXPT_NAME}")

test_metric = 0
drop_metric = 0
rgb_pred = np.zeros((MB_SIZE, 512, 512, 3, 3), dtype=np.float32)
rgb_drop = np.zeros((MB_SIZE, 512, 512, 3, 3), dtype=np.float32)
pred_mean = np.zeros((MB_SIZE, 512, 512, 3, 1), dtype=np.float32)
pred_entropy = np.zeros((MB_SIZE, 512, 512, 3, 1), dtype=np.float32)

for img, seg in test_ds.batch(MB_SIZE):
    pred = UNetStandard(img, training=False)
    pred_mean[:, :, :, :, 0], pred_entropy[:, :, :, :, 0], _ = varDropout(img, UNetDropout, T=T)
    mean_entropy = np.mean(pred_entropy, axis=(1, 2, 4))
    temp_metric = diceLoss(pred, seg)
    temp_drop_metric = diceLoss(pred_mean, seg)
    test_metric += temp_metric
    drop_metric += temp_drop_metric # This is probably wrong
    print(pred.shape, seg.shape, pred_mean.shape)
    print(f"Batch Dice score: {1 - temp_metric / MB_SIZE}, Ensemble Dice score: {1 - temp_drop_metric / MB_SIZE}")

    rgb_pred[:, :, :, :, 0] = seg[:, :, :, :, 0]
    rgb_pred[:, :, :, :, 1] = pred[:, :, :, :, 0].numpy()
    rgb_pred[:, :, :, :, 2] = pred[:, :, :, :, 0].numpy()
    rgb_drop[:, :, :, :, 0] = seg[:, :, :, :, 0]
    rgb_drop[:, :, :, :, 1] = pred_mean[:, :, :, :, 0]
    rgb_drop[:, :, :, :, 2] = pred_mean[:, :, :, :, 0]

    for j in range(1, 2):#range(img.shape[3]):
        fig, axs = plt.subplots(MB_SIZE, 7)

        for i in range(img.shape[0]):
            axs[i, 0].imshow(np.fliplr(img[i, :, :, j, 0].numpy().T), cmap='gray', vmin=0.12, vmax=0.18, origin='lower')
            axs[i, 0].axis('off')
            axs[i, 1].imshow(np.fliplr(img[i, :, :, j, 0].numpy().T), cmap='gray', vmin=0.12, vmax=0.18, origin='lower')
            axs[i, 1].axis('off')
            # axs[j, 1].imshow(np.fliplr(np.ma.masked_where(pred[j, :, :, 1, 0].numpy().T == False, pred[j, :, :, 1, 0].numpy().T)), cmap='Set1', origin='lower')
            axs[i, 1].imshow(np.fliplr(pred[i, :, :, j, 0].numpy().T), cmap='hot', alpha=0.3, origin='lower')
            axs[i, 1].axis('off')
            axs[i, 2].imshow(np.fliplr(img[i, :, :, j, 0].numpy().T), cmap='gray', vmin=0.12, vmax=0.18, origin='lower')
            axs[i, 2].axis('off')
            # axs[j, 2].imshow(np.fliplr(np.ma.masked_where(pred_mean[j, :, :, 1, 0].T == False, pred[j, :, :, 1, 0].numpy().T)), cmap='Set1', origin='lower')
            axs[i, 2].imshow(np.fliplr(pred_mean[i, :, :, j, 0].T), cmap='hot', alpha=0.3, origin='lower')
            axs[i, 2].axis('off')
            r_pred = np.fliplr(rgb_pred[i, :, :, j, 0].T)
            g_pred = np.fliplr(rgb_pred[i, :, :, j, 1].T)
            b_pred = np.fliplr(rgb_pred[i, :, :, j, 2].T)
            axs[i, 3].imshow(np.concatenate([r_pred[:, :, np.newaxis], g_pred[:, :, np.newaxis], b_pred[:, :, np.newaxis]], axis=2), origin='lower')
            axs[i, 3].axis('off')
            r_drop = np.fliplr(rgb_drop[i, :, :, j, 0].T)
            g_drop = np.fliplr(rgb_drop[i, :, :, j, 1].T)
            b_drop = np.fliplr(rgb_drop[i, :, :, j, 2].T)
            axs[i, 4].imshow(np.concatenate([r_drop[:, :, np.newaxis], g_drop[:, :, np.newaxis], b_drop[:, :, np.newaxis]], axis=2), origin='lower')
            axs[i, 4].axis('off')
            axs[i, 5].imshow(np.fliplr(pred_mean[i, :, :, j, 0].T), cmap='hot', origin='lower')
            axs[i, 5].axis('off')
            axs[i, 6].imshow(np.fliplr(pred_entropy[i, :, :, j, 0].T), cmap='hot', origin='lower')
            axs[i, 6].axis('off')

            NMS = NMSCalc(pred[i, :, :, 1, 0], 0.5, test=False)
            axs[i, 3].set_title(f"{NMS:.6f}")
            NMS = NMSCalc(pred_mean[i, :, :, 1, 0], 0.5, test=False)
            axs[i, 4].set_title(f"{NMS:.6f}")
            axs[i, 6].set_title(f"{mean_entropy[i, j]:.6f}")

        plt.show()

print(f"Final Dice score: {1 - test_metric / N}, Final ensemble Dice score: {1 - drop_metric / N}")