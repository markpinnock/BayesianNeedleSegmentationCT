import json
import numpy as np
import os
import sys

sys.path.append('..')

from TestFuncs import thresholdImg, diceLoss, haussdorfDistance, NMSCalc, trajErrorCalc

DATA_PATH = "Z:/Robot_Data/Test/"
NPY_PATH = "C:/Users/roybo/OneDrive - University College London/Collaborations/RobotNeedleSeg/Code/003_CNN_Bayes_Traj/output_npy/"
FILE_PATH = "C:/Users/roybo/OneDrive - University College London/Collaborations/RobotNeedleSeg/Code/003_CNN_Bayes_Traj/"
SAVE_PATH = f"{FILE_PATH}mat_files/"

SEG_PATH = f"{DATA_PATH}Seg/"
segs = os.listdir(SEG_PATH)
segs.sort()

THRESH = 0.5
T = 20
STD_PATH = f"{NPY_PATH}std/"
DROP1_PATH = f"{NPY_PATH}drop1/"
DROP2_PATH = f"{NPY_PATH}drop2_T{T}/"
DROP2E_PATH = f"{NPY_PATH}drop2E_T{T}/"

std_list = os.listdir(STD_PATH)
std_list.sort()

dice_std = []
dice_drop1 = []
dice_drop2 = []
hdf_std = []
hdf_drop1 = []
hdf_drop2 = []
entropy_drop2 = []
nme_std = []
nme_drop1 = []
nme_drop2 = []
angle_std = []
angle_drop1 = []
angle_drop2 = []
centre_std = []
centre_drop1 = []
centre_drop2 = []
traj_hdf_std = []
traj_hdf_drop1 = []
traj_hdf_drop2 = []


for i in range(len(std_list)):
    img_stem = std_list[i][:-8]
    final_seg_list = [seg for seg in segs if img_stem[:-4] in seg]
    final_seg_list.sort()
    seg_name = final_seg_list[-1]
    seg = np.load(f"{SEG_PATH}{seg_name}")

    std = np.load(f"{STD_PATH}{img_stem}_STD.npy")
    drop1 = np.load(f"{DROP1_PATH}{img_stem}_D1.npy")
    drop2 = np.load(f"{DROP2_PATH}{img_stem}_D2.npy")
    drop2E = np.load(f"{DROP2E_PATH}{img_stem}_D2E.npy")

    mean_entropy = np.sum(drop2E * drop2) / np.sum(drop2)

    if THRESH != None:
        std = thresholdImg(std, THRESH)
        drop1 = thresholdImg(drop1, THRESH)
        drop2 = thresholdImg(drop2, THRESH)

    dice_std.append(1 - diceLoss(std, seg))
    dice_drop1.append(1 - diceLoss(drop1, seg))
    dice_drop2.append(1 - diceLoss(drop2, seg))

    if "UCLH_08933783" in img_stem:
        pixel_spacing = 0.976
    else:
        pixel_spacing = 0.781

    hdf_std.append(haussdorfDistance(std, seg, pixel_spacing))
    hdf_drop1.append(haussdorfDistance(drop1, seg, pixel_spacing))
    hdf_drop2.append(haussdorfDistance(drop2, seg, pixel_spacing))

    entropy_drop2.append(mean_entropy)

    nme_std.append(NMSCalc(std[:, :, 1], THRESH, pixel_spacing))
    nme_drop1.append(NMSCalc(drop1[:, :, 1], THRESH, pixel_spacing))
    nme_drop2.append(NMSCalc(drop2[:, :, 1], THRESH, pixel_spacing))
    temp1 = trajErrorCalc(std[:, :, 1], seg, THRESH, pixel_spacing, False)
    temp2 = trajErrorCalc(drop1[:, :, 1], seg, THRESH, pixel_spacing, False)
    temp3 = trajErrorCalc(drop2[:, :, 1], seg, THRESH, pixel_spacing, False)
    angle_std.append(temp1[0])
    angle_drop1.append(temp2[0])
    angle_drop2.append(temp3[0])
    centre_std.append(temp1[1])
    centre_drop1.append(temp2[1])
    centre_drop2.append(temp3[1])
    traj_hdf_std.append(temp1[2])
    traj_hdf_drop1.append(temp2[2])
    traj_hdf_drop2.append(temp3[2])

    # print(f"{imgs[idx]}, {NMSCalc(drop2[:, :, 1], THRESH)}")

# print(np.sqrt(np.nanmean(np.square(traj_std))), np.sqrt(np.nanmean(np.square(centre_std))), np.nanmean(traj_hdf_std))
# print(np.sqrt(np.mean(np.square(traj_drop1))), np.sqrt(np.mean(np.square(centre_drop1))), np.mean(traj_hdf_drop1))
# print(np.sqrt(np.mean(np.square(traj_drop2))), np.sqrt(np.mean(np.square(centre_drop2))), np.mean(traj_hdf_drop2))
#
# print("========================")
# print(np.mean(dice_std), np.mean(dice_drop1), np.mean(dice_drop2))
# print(np.nanmean(nme_std), np.mean(nme_drop1), np.mean(nme_drop2))
list_dict = {
    'dice_std': dice_std,
    'dice_drop1': dice_drop1,
    'dice_drop2': dice_drop2,
    'hdf_std': hdf_std,
    'hdf_drop1': hdf_drop1,
    'hdf_drop2': hdf_drop2,
    'entropy_drop2': entropy_drop2,
    'nme_std': nme_std,
    'nme_drop1': nme_drop1,
    'nme_drop2': nme_drop2,
    'angle_std': angle_std,
    'angle_drop1': angle_drop1,
    'angle_drop2': angle_drop2,
    'centre_std': centre_std,
    'centre_drop1': centre_drop1,
    'centre_drop2': centre_drop2,
    'traj_hdf_std': traj_hdf_std,
    'traj_hdf_drop1': traj_hdf_drop1,
    'traj_hdf_drop2':traj_hdf_drop2
}

with open(f"{SAVE_PATH}results05.json", 'w') as out_file:
    json.dump(list_dict, out_file, indent=4)
