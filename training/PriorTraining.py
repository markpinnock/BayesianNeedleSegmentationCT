from argparse import ArgumentParser
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import tensorflow.keras as keras
import tensorflow as tf
import time

sys.path.append('..')
sys.path.append('/home/mpinnock/Robot/002_CNN_Bayes_RNS/')

from Networks import UNetGen
from utils.DataLoader import imgLoader
from utils.TrainFuncs import trainStep, valStep, varDropout


# Handle arguments
parser = ArgumentParser()
parser.add_argument('--file_path', '-fp', help="File path", type=str)
parser.add_argument('--data_path', '-dp', help="Data path", type=str)
# parser.add_argument('--data_aug', '-da', help="Data augmentation", action='store_true')
parser.add_argument('--minibatch_size', '-mb', help="Minibatch size", type=int, nargs='?', const=4, default=4)
parser.add_argument('--num_chans', '-nc', help="Starting number of channels", type=int, nargs='?', const=4, default=4)
parser.add_argument('--epochs', '-ep', help="Number of epochs", type=int, nargs='?', const=5, default=5)
parser.add_argument('--gpu', '-g', help="GPU number", type=int, nargs='?', const=0, default=0)
parser.add_argument('--eta', '-e', help="Learning rate", type=float, nargs='?', const=0.001, default=0.001)
arguments = parser.parse_args()

# Generate file path and data path
if arguments.file_path == None:
    FILE_PATH = "C:/Users/roybo/OneDrive - University College London/Collaborations/RobotNeedleSeg/Code/002_CNN_Bayes_RNS/"
else:
    FILE_PATH = arguments.file_path

if arguments.data_path == None:
    DATA_PATH = "Z:/Robot_Data/"
else:
    DATA_PATH = arguments.data_path

# Set hyperparameters
MB_SIZE = arguments.minibatch_size
NC = arguments.num_chans # Number of feature maps in first conv layer
EPOCHS = arguments.epochs
ETA = arguments.eta # Learning rate
DROP_RATE = 0.2
GPU = arguments.gpu

# Generate experiment name and save paths
EXPT_NAME = f"nc{NC}_ep{EPOCHS}_eta{ETA}"
PRIOR_PATH = f"{FILE_PATH}scripts/training/prior/"

if not os.path.exists(PRIOR_PATH):
    os.mkdir(PRIOR_PATH)

IMAGE_SAVE_PATH = f"{FILE_PATH}images/{EXPT_NAME}/"

if not os.path.exists(IMAGE_SAVE_PATH):
    os.mkdir(IMAGE_SAVE_PATH)

# Open log file
if arguments.file_path == None:
    LOG_SAVE_PATH = f"{FILE_PATH}/"
else:
    LOG_SAVE_PATH = f"{FILE_PATH}reports/"

LOG_SAVE_NAME = f"{LOG_SAVE_PATH}{EXPT_NAME}.txt"

if not os.path.exists(LOG_SAVE_PATH):
    os.mkdir(LOG_SAVE_PATH)

log_file = open(LOG_SAVE_NAME, 'w')

# Find data and check img and seg pair numbers match, then shuffle
img_path = f"{DATA_PATH}Img/"
seg_path = f"{DATA_PATH}Seg/"
imgs = os.listdir(img_path)
segs = os.listdir(seg_path)
imgs.sort()
segs.sort()
img_train = imgs
seg_train = segs

N = len(imgs)
assert N == len(segs), "HI/LO IMG PAIRS UNEVEN LENGTHS"

IMG_SIZE = (512, 512, 3, 1, )
RGB_SIZE = (512, 512, 3, 3, )

# Initialise model
UNet = UNetGen(input_shape=IMG_SIZE, starting_channels=NC, drop_rate=DROP_RATE)

# Create dataset
train_ds = tf.data.Dataset.from_generator(
    imgLoader, args=[img_path, seg_path, img_train, seg_train, False, True, False], output_types=(tf.float32, tf.float32))

if arguments.file_path == None:
    print(UNet.summary())

# Create losses
train_metric = 0
train_count = 0
Optimiser = keras.optimizers.Adam(ETA)

# Set start time
start_time = time.time()

# Training
for epoch in range(EPOCHS):
    for img, seg in train_ds.batch(MB_SIZE):
        train_metric += trainStep(img, seg, UNet, Optimiser)
        train_count += 1

    # Print losses every epoch
    print(f"Epoch: {epoch + 1}, Train Loss: {train_metric / train_count}")
    log_file.write(f"Epoch: {epoch + 1}, Train Loss: {train_metric / train_count}\n")
    train_metric = 0

UNet.save_weights(f"{PRIOR_PATH}{EXPT_NAME}.ckpt")

log_file.write(f"Time: {(time.time() - start_time) / 60:.2f} min\n")
log_file.close()
