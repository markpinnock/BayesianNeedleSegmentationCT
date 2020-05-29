import numpy as np
import os
import scipy.io as io
import sys
import tensorflow as tf

sys.path.append('..')

from Networks import UNetGen
from utils.DataLoader import imgLoader
from utils.TrainFuncs import diceLoss, haussdorfDistance, NMSCalc, varDropout


MB_SIZE = 1
NC = 4
EPOCHS = 500
ETA = 0.001
T = 20
THRESH = 0.5

FILE_PATH = "C:/Users/roybo/OneDrive - University College London/Collaborations/RobotNeedleSeg/Code/003_CNN_Bayes_Traj/"
DATA_PATH = "Z:/Robot_Data/Test/"
SAVE_PATH = "C:/Users/roybo/OneDrive - University College London/Collaborations/RobotNeedleSeg/Code/003_CNN_Bayes_Traj/mat_files/"
EXPT_NAME = f"nc{NC}_ep{EPOCHS}_eta{ETA}"
# EXPT_NAME = "spatial_dropout"
MODEL_SAVE_PATH = f"{FILE_PATH}models/{EXPT_NAME}/"
SAVE_PATH = f"{FILE_PATH}mat_files/"
STD_MODEL_SAVE_PATH = f"{FILE_PATH}models/nc4_drop0_0/nc4_drop0_0"
START_IDX = 0

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
    imgLoader, args=[img_path, seg_path, imgs[START_IDX:], segs[START_IDX:], False, False], output_types=(tf.float32, tf.float32))

UNetStandard = UNetGen(input_shape=LO_VOL_SIZE, starting_channels=NC, drop_rate=0.0, dropout_flag=False)
UNetStandard.load_weights(f"{STD_MODEL_SAVE_PATH}")
UNetDropout1 = UNetGen(input_shape=LO_VOL_SIZE, starting_channels=NC, drop_rate=0.0, dropout_flag=False)
UNetDropout1.load_weights(f"{MODEL_SAVE_PATH}{EXPT_NAME}")
UNetDropout2 = UNetGen(input_shape=LO_VOL_SIZE, starting_channels=NC, drop_rate=0.2, dropout_flag=True)
UNetDropout2.load_weights(f"{MODEL_SAVE_PATH}{EXPT_NAME}")

pred_drop_1 = np.zeros((MB_SIZE, 512, 512, 3, 1), dtype=np.float32)
pred_drop_2 = np.zeros((MB_SIZE, 512, 512, 3, 1), dtype=np.float32)
pred_var = np.zeros((MB_SIZE, 512, 512, 3, 1), dtype=np.float32)

save_count = 0
idx = 0 + START_IDX
skip_imgs = ["UCLH_08470270_1_006_L.npy",
             "UCLH_08470270_1_013_L.npy",
             "UCLH_08933783_1_007_L.npy",
             "UCLH_08933783_1_017_L.npy",
             "UCLH_09099884_1_014_L.npy",
             "UCLH_09099884_1_023_L.npy",
             "UCLH_11107604_1_009_L.npy",
             "UCLH_11107604_1_017_L.npy",
             "UCLH_11192578_1_008_L.npy",
             "UCLH_11192578_1_016_L.npy",
             "UCLH_11349911_1_019_L.npy",
             "UCLH_11349911_1_027_L.npy"]

dice_std = []
dice_drop_1 = []
dice_drop_2 = []
dist_std = []
dist_drop_1 = []
dist_drop_2 = []
var_drop_2 = []
nms_drop_2 = []


for img, seg in test_ds.batch(MB_SIZE):
    pred_std = UNetStandard(img, training=False).numpy()
    pred_drop_1[:, :, :, :, 0], _, _ = varDropout(img, UNetDropout1, T=1, threshold=None)
    pred_drop_2[:, :, :, :, 0], pred_entropy[:, :, :, :, 0], _ = varDropout(img, UNetDropout2, T=T, threshold=None)
    pred_std[pred_std < THRESH] = 0
    pred_std[pred_std >= THRESH] = 1
    pred_drop_1[pred_drop_1 < THRESH] = 0
    pred_drop_1[pred_drop_1 >= THRESH] = 1
    pred_drop_2[pred_drop_2 < THRESH] = 0
    pred_drop_2[pred_drop_2 >= THRESH] = 1

    if imgs[idx] in skip_imgs:
        print(f"Skipping {imgs[idx]}")
        idx += 1
        continue

    assert MB_SIZE == 1, "INCORRECT MB SIZE"

    dice_std.append(1 - diceLoss(pred_std, seg).numpy())
    dice_drop_1.append(1 - diceLoss(pred_drop_1, seg).numpy())
    dice_drop_2.append(1 - diceLoss(pred_drop_2, seg).numpy())

    if "UCLH_08933783" in imgs[idx]:
        pixel_spacing = 0.976
    else:
        pixel_spacing = 0.781

    dist_std.append(haussdorfDistance(pred_std[:, :, :, 1, :], seg[:, :, :, 1, :], pixel_spacing))
    dist_drop_1.append(haussdorfDistance(pred_drop_1[:, :, :, 1, :], seg[:, :, :, 1, :], pixel_spacing))
    dist_drop_2.append(haussdorfDistance(pred_drop_2[:, :, :, 1, :], seg[:, :, :, 1, :], pixel_spacing))

    mean_entropy = np.sum(pred_entropy, axis=(1, 2, 3, 4)) / np.sum(np.logical_or(pred_drop_2 > 0.0, pred_entropy > 0.0), axis=(1, 2, 3, 4))
    # var_drop_2.append(mean_entropy[0, 1])
    var_drop_2.append(mean_entropy)
    nms_std.append(NMSCalc(pred_std[0, :, :, 1, 0], THRESH))
    nms_drop_1.append(NMSCalc(pred_drop_1[0, :, :, 1, 0], THRESH))
    nms_drop_2.append(NMSCalc(pred_drop_2[0, :, :, 1, 0], THRESH))

    print(
        1 - diceLoss(pred_std, seg).numpy(),
        1 - diceLoss(pred_drop_1, seg).numpy(),
        1 - diceLoss(pred_drop_2, seg).numpy()
    )

    # print(
    #     haussdorfDistance(pred_std[:, :, :, 1, :], seg[:, :, :, 1, :]),
    #     haussdorfDistance(pred_drop_1[:, :, :, 1, :], seg[:, :, :, 1, :]),
    #     haussdorfDistance(pred_drop_2[:, :, :, 1, :], seg[:, :, :, 1, :])
    # )

    idx += 1

list_dict = {
    'dice_std': dice_std,
    'dice_drop_1': dice_drop_1,
    'dice_drop_2': dice_drop_2,
    'dist_std': dist_std,
    'dist_drop_1': dist_drop_1,
    'dist_drop_2': dist_drop_2,
    'var_drop_2': var_drop_2,
    'nms_drop_2': nms_drop_2
}

io.savemat(f"{SAVE_PATH}dice_var_2.mat", list_dict)
