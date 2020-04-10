import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import tensorflow as tf

sys.path.append('..')
sys.path.append("C:/Users/roybo/OneDrive - University College London/Collaborations/RobotNeedleSeg/Code/003_CNN_Bayes_Traj/scripts/training/")

from utils.TransGen import TransMatGen
from utils.Transformation import affineTransformation


def imgLoader(img_path, seg_path, img_list, seg_list, prior_list, shuffle_flag, prior_flag=False):
    img_path = img_path.decode("utf-8")
    seg_path = seg_path.decode("utf-8")

    if shuffle_flag == True:
        temp_list = list(zip(img_list, seg_list))
        np.random.shuffle(temp_list)
        img_list, seg_list = zip(*temp_list)

    N = len(img_list)
    i = 0 

    while i < N:
        try:
            img_name = img_list[i].decode("utf-8")
            img_vol = np.load(img_path + img_name).astype(np.float32)
            img_vol = img_vol[:, :, :, np.newaxis]

            final_seg_list = [seg.decode("utf-8") for seg in seg_list if img_name[:-9] in seg.decode("utf-8")]
            final_seg_list.sort()
            seg_name = final_seg_list[-1]
            # seg_name = seg_list[i].decode("utf-8")
            # print(img_name, seg_name)
            seg_vol = np.load(seg_path + seg_name).astype(np.float32)
            seg_vol = seg_vol[:, :, :, np.newaxis]

            if prior_flag:
                idx = np.where(prior_list == img_name.encode("utf-8"))
                prior_name = prior_list[idx[0] - 1][0].decode("utf-8")

                if prior_name[:-10] != img_name[:-10]:
                    prior_vol = np.zeros(img_vol.shape, dtype=np.float32)
                    # print(img_name, "ZEROS")
                else:
                    prior_vol = np.load(img_path + prior_name).astype(np.float32)
                    prior_vol = prior_vol[:, :, :, np.newaxis]
                    # print(img_name, prior_name)

        except Exception as e:
            print(f"IMAGE OR MASK LOAD FAILURE: {img_name} ({e})")

        else:
            if prior_flag:
                yield (img_vol, seg_vol, prior_vol)
            else:
                yield (img_vol, seg_vol)
        finally:
            i += 1

# Data aug

if __name__ == "__main__":

    FILE_PATH = "Z:/Robot_Data/Test/"
    img_path = f"{FILE_PATH}Img/"
    seg_path = f"{FILE_PATH}Seg/"

    # PRIOR_NAME = "nc4_ep100_eta0.001"
    # PRIOR_PATH = f"C:/Users/roybo/OneDrive - University College London/Collaborations/RobotNeedleSeg/Code/002_CNN_Bayes_RNS/scripts/training/prior/{PRIOR_NAME}.ckpt"
    # UNetPrior = UNetGen(input_shape=(512, 512, 3, 1, ), starting_channels=4, drop_rate=0.2)
    # UNetPrior.load_weights(PRIOR_PATH)

    imgs = os.listdir(img_path)
    segs = os.listdir(seg_path)
    imgs.sort()
    segs.sort()
    priors = imgs

    N = len(imgs)
    NUM_FOLDS = 5
    FOLD = 0
    MB_SIZE = 8
    random.seed(10)

    for i in range(N):
        # print(imgs[i], segs[i])
        assert imgs[i][:-9] == segs[i][:-9], "HI/LO PAIRS DON'T MATCH"

    temp_list = list(zip(imgs, segs))
    random.shuffle(temp_list)
    imgs, segs = zip(*temp_list)

    for i in range(N):
        # print(imgs[i], segs[i])
        assert imgs[i][:-9] == segs[i][:-9], "HI/LO PAIRS DON'T MATCH"

    num_in_fold = int(N / NUM_FOLDS)
    img_val = imgs[FOLD * num_in_fold:(FOLD + 1) * num_in_fold]
    seg_val = segs[FOLD * num_in_fold:(FOLD + 1) * num_in_fold]
    img_train = imgs[0:FOLD * num_in_fold] + imgs[(FOLD + 1) * num_in_fold:]
    seg_train = segs[0:FOLD * num_in_fold] + segs[(FOLD + 1) * num_in_fold:]

    for i in range(len(img_val)):
        # print(img_val[i], seg_val[i])
        assert img_val[i][:-9] == seg_val[i][:-9], "HI/LO PAIRS DON'T MATCH"
    
    for i in range(len(img_train)):
        # print(img_train[i], seg_train[i])
        assert img_train[i][:-9] == seg_train[i][:-9], "HI/LO PAIRS DON'T MATCH"

    print(f"N: {N}, val: {len(img_val)}, train: {len(img_train)}, val + train: {len(img_val) + len(img_train)}")
    
    train_ds = tf.data.Dataset.from_generator(
        imgLoader, args=[img_path, seg_path, img_train, seg_train, priors, False, False], output_types=(tf.float32, tf.float32))

    val_ds = tf.data.Dataset.from_generator(
        imgLoader, args=[img_path, seg_path, img_val, seg_val, priors, False, False], output_types=(tf.float32, tf.float32))

    DataAug = TransMatGen()

    for img, seg in train_ds.batch(MB_SIZE):
        print(img.shape, seg.shape)
        # trans_mat = DataAug.transMatGen(img.shape[0])
        # img = affineTransformation(img, trans_mat)
        # seg = affineTransformation(seg, trans_mat)

        plt.subplot(1, 2, 1)
        plt.axis('off')
        plt.imshow(np.fliplr(img[0, :, :, 1, 0].numpy().T), cmap='gray', origin='lower', vmin=0.12, vmax=0.18)
        plt.subplot(1, 2, 2)
        plt.axis('off')
        plt.imshow(np.fliplr(seg[0, :, :, 1, 0].numpy().T), cmap='gray', origin='lower', vmin=0.12, vmax=0.18)
        # pred = UNetPrior(data[2])
        # _, pred_var = varDropout(data[2], UNetPrior, T=10)
        # plt.subplot(1, 3, 3)
        # plt.imshow(np.fliplr(pred_var[0, :, :, 1].T), cmap='hot', origin='lower')
        # plt.pause(2)
        plt.show()
    
    for data in val_ds.batch(MB_SIZE):
        print(data[0].shape, data[1].shape)
        # pass
