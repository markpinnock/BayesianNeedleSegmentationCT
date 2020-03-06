import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf

sys.path.append('..')

from Networks import UNetGen
from utils.DataLoader import imgLoader
from utils.TrainFuncs import diceLoss, varDropout


MB_SIZE = 4
NC = 4
EPOCHS = 1
ETA = 0.001

FILE_PATH = "C:/Users/roybo/OneDrive - University College London/Collaborations/RobotNeedleSeg/Code/003_CNN_Bayes_Traj/"
DATA_PATH = "Z:/Robot_Data/Test/"
EXPT_NAME = f"nc{NC}_ep{EPOCHS}_eta{ETA}"
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
UNetDropout = UNetGen(input_shape=LO_VOL_SIZE, starting_channels=NC, drop_rate=0.2, dropout_flag=True)
UNetDropout.load_weights(f"{MODEL_SAVE_PATH}{EXPT_NAME}")

test_metric = 0
drop_metric = 0
rgb_pred = np.zeros((MB_SIZE, 512, 512, 3, 3), dtype=np.float32)
rgb_drop = np.zeros((MB_SIZE, 512, 512, 3, 3), dtype=np.float32)
pred_mean = np.zeros((MB_SIZE, 512, 512, 3, 1), dtype=np.float32)
pred_var = np.zeros((MB_SIZE, 512, 512, 3, 1), dtype=np.float32)

for img, seg in test_ds.batch(MB_SIZE):
    fig, axs = plt.subplots(MB_SIZE, 7)
    pred = UNetStandard(img, training=False)
    pred_mean[:, :, :, :, 0], pred_var[:, :, :, :, 0], _ = varDropout(img, UNetDropout, T=100)
    temp_metric = diceLoss(pred, seg)
    temp_drop_metric = diceLoss(pred_mean, seg)
    test_metric += temp_metric
    drop_metric += temp_drop_metric
    print(f"Batch Dice score: {1 - temp_metric / MB_SIZE}, Ensemble Dice score: {1 - temp_drop_metric / MB_SIZE}")

    rgb_pred[:, :, :, :, 0] = seg[:, :, :, :, 0]
    rgb_pred[:, :, :, :, 1] = pred[:, :, :, :, 0].numpy()
    rgb_pred[:, :, :, :, 2] = pred[:, :, :, :, 0].numpy()
    rgb_drop[:, :, :, :, 0] = seg[:, :, :, :, 0]
    rgb_drop[:, :, :, :, 1] = pred_mean[:, :, :, :, 0]
    rgb_drop[:, :, :, :, 2] = pred_mean[:, :, :, :, 0]

    for j in range(MB_SIZE):
        axs[j, 0].imshow(np.fliplr(img[j, :, :, 1, 0].numpy().T), cmap='gray', vmin=0.12, vmax=0.18, origin='lower')
        axs[j, 0].axis('off')
        axs[j, 1].imshow(np.fliplr(img[j, :, :, 1, 0].numpy().T), cmap='gray', vmin=0.12, vmax=0.18, origin='lower')
        axs[j, 1].axis('off')
        # axs[j, 1].imshow(np.fliplr(np.ma.masked_where(pred[j, :, :, 1, 0].numpy().T == False, pred[j, :, :, 1, 0].numpy().T)), cmap='Set1', origin='lower')
        axs[j, 1].imshow(np.fliplr(pred[j, :, :, 1, 0].numpy().T), cmap='hot', alpha=0.3, origin='lower')
        axs[j, 1].axis('off')
        axs[j, 2].imshow(np.fliplr(img[j, :, :, 1, 0].numpy().T), cmap='gray', vmin=0.12, vmax=0.18, origin='lower')
        axs[j, 2].axis('off')
        # axs[j, 2].imshow(np.fliplr(np.ma.masked_where(pred_mean[j, :, :, 1, 0].T == False, pred[j, :, :, 1, 0].numpy().T)), cmap='Set1', origin='lower')
        axs[j, 2].imshow(np.fliplr(pred_mean[j, :, :, 1, 0].T), cmap='hot', alpha=0.3, origin='lower')
        axs[j, 2].axis('off')
        r_pred = np.fliplr(rgb_pred[j, :, :, 1, 0].T)
        g_pred = np.fliplr(rgb_pred[j, :, :, 1, 1].T)
        b_pred = np.fliplr(rgb_pred[j, :, :, 1, 2].T)
        axs[j, 3].imshow(np.concatenate([r_pred[:, :, np.newaxis], g_pred[:, :, np.newaxis], b_pred[:, :, np.newaxis]], axis=2), origin='lower')
        axs[j, 3].axis('off')
        r_drop = np.fliplr(rgb_drop[j, :, :, 1, 0].T)
        g_drop = np.fliplr(rgb_drop[j, :, :, 1, 1].T)
        b_drop = np.fliplr(rgb_drop[j, :, :, 1, 2].T)
        axs[j, 4].imshow(np.concatenate([r_drop[:, :, np.newaxis], g_drop[:, :, np.newaxis], b_drop[:, :, np.newaxis]], axis=2), origin='lower')
        axs[j, 4].axis('off')
        axs[j, 5].imshow(np.fliplr(pred_mean[j, :, :, 1, 0].T), cmap='hot', origin='lower')
        axs[j, 5].axis('off')
        axs[j, 6].imshow(np.fliplr(pred_var[j, :, :, 1, 0].T), cmap='hot', origin='lower')
        axs[j, 6].axis('off')

    plt.show()

print(f"Final Dice score: {1 - test_metric / N}, Final ensemble Dice score: {1 - drop_metric / N}")