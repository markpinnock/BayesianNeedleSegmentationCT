import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow.keras as keras

sys.path.append('..')


from training.Networks import UNetGen
from utils.DataLoader import imgLoader
from utils.TrainFuncs import varDropout


MB_SIZE = 4
NC = 4
EPOCHS = 500
ETA = 0.001

FILE_PATH = "C:/Users/roybo/OneDrive - University College London/Collaborations/RobotNeedleSeg/Code/003_CNN_Bayes_Traj/"
DATA_PATH = "Z:/Robot_Data/Test/"
EXPT_NAME = "spatial_dropout"
# EXPT_NAME = f"nc{NC}_ep{EPOCHS}_eta{ETA}"
MODEL_SAVE_PATH = f"{FILE_PATH}models/{EXPT_NAME}/"
img_path = f"{DATA_PATH}Img/"
seg_path = f"{DATA_PATH}Seg/"
imgs = os.listdir(img_path)
segs = os.listdir(seg_path)
imgs.sort()
segs.sort()

test_ds = tf.data.Dataset.from_generator(
    imgLoader, args=[img_path, seg_path, imgs, segs, False, False], output_types=(tf.float32, tf.float32))

UNetStandard = UNetGen(input_shape=(512, 512, 3, 1, ), starting_channels=NC, drop_rate=0.0, dropout_flag=False)
# print(UNetStandard.summary())
UNetStandard.load_weights(f"{MODEL_SAVE_PATH}{EXPT_NAME}")
UNetDropout = UNetGen(input_shape=(512, 512, 3, 1, ), starting_channels=NC, drop_rate=0.2, dropout_flag=True)
UNetDropout.load_weights(f"{MODEL_SAVE_PATH}{EXPT_NAME}")

print(UNetStandard.summary())
print(UNetDropout.summary())

std_vars = [var for var in UNetStandard.trainable_variables if len(var.shape) > 4]
drop_vars = [var for var in UNetDropout.trainable_variables if len(var.shape) > 4]

DN_convs = ['DN1_conv1', 'DN1_conv2', 'DN2_conv1', 'DN2_conv2', 'DN3_conv1', 'DN3_conv2',
            'DN4_conv1', 'DN4_conv2', 'DN5_conv']
UP_convs = ['UP4_tconv', 'UP4_conv1', 'UP4_conv2', 'UP3_tconv', 'UP3_conv1', 'UP3_conv2',
            'UP2_tconv', 'UP2_conv1', 'UP2_conv2', 'UP1_tconv', 'UP1_conv1', 'UP1_conv2',
            'out_conv']

print(len(UNetStandard(np.zeros(1, 512, 512, 3, 1))))

for idx, layer in enumerate(DN_convs + UP_convs):
    plt.hist(std_vars[idx].numpy().ravel())
    plt.title(layer)
    plt.show()
