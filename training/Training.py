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
sys.path.append('/home/mpinnock/Traj/003_CNN_Bayes_Traj/')

from Networks import UNetGen
from utils.DataLoader import imgLoader
from utils.TrainFuncs import trainStep, valStep, varDropout
from utils.TransGen import TransMatGen
from utils.Transformation import affineTransformation


# Handle arguments
parser = ArgumentParser()
parser.add_argument('--file_path', '-fp', help="File path", type=str)
parser.add_argument('--data_path', '-dp', help="Data path", type=str)
parser.add_argument('--data_aug', '-da', help="Data augmentation", action='store_true')
parser.add_argument('--minibatch_size', '-mb', help="Minibatch size", type=int, nargs='?', const=4, default=4)
parser.add_argument('--num_chans', '-nc', help="Starting number of channels", type=int, nargs='?', const=4, default=4)
parser.add_argument('--epochs', '-ep', help="Number of epochs", type=int, nargs='?', const=5, default=5)
parser.add_argument('--folds', '-f', help="Number of cross-validation folds", type=int, nargs='?', const=0, default=0)
parser.add_argument('--crossval', '-c', help="Fold number", type=int, nargs='?', const=0, default=0)
parser.add_argument('--gpu', '-g', help="GPU number", type=int, nargs='?', const=0, default=0)
parser.add_argument('--eta', '-e', help="Learning rate", type=float, nargs='?', const=0.001, default=0.001)
arguments = parser.parse_args()

# Generate file path and data path
if arguments.file_path == None:
    FILE_PATH = "C:/Users/roybo/OneDrive - University College London/Collaborations/RobotNeedleSeg/Code/003_CNN_Bayes_Traj/"
else:
    FILE_PATH = arguments.file_path

if arguments.data_path == None:
    DATA_PATH = "Z:/Robot_Data/Train/"
else:
    DATA_PATH = arguments.data_path

AUG_FLAG = arguments.data_aug

if AUG_FLAG:
    DataAug = TransMatGen()
#     aug_dict = {
#         'flip': 0.5,
#         'rot': 45,
#         'scale': 0.25,
#         'shear': None
#     }

# Set hyperparameters
MB_SIZE = arguments.minibatch_size
NC = arguments.num_chans # Number of feature maps in first conv layer
EPOCHS = arguments.epochs
NUM_FOLDS = arguments.folds
FOLD = arguments.crossval
ETA = arguments.eta # Learning rate
NUM_EX = 4 # Number of example images to display
DROP_RATE = 0.2

if FOLD >= NUM_FOLDS and NUM_FOLDS != 0:
   raise ValueError("Fold number cannot be greater or equal to number of folds")

GPU = arguments.gpu

# Generate experiment name and save paths
EXPT_NAME = "spatial_dropout" #f"nc{NC}_ep{EPOCHS}_eta{ETA}"

if NUM_FOLDS > 0:
    EXPT_NAME += f"_cv{FOLD}"

MODEL_SAVE_PATH = f"{FILE_PATH}models/{EXPT_NAME}/"

if not os.path.exists(MODEL_SAVE_PATH) and NUM_FOLDS == 0:
    os.mkdir(MODEL_SAVE_PATH)

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
priors = imgs

N = len(imgs)
assert N == len(segs), "HI/LO IMG PAIRS UNEVEN LENGTHS"

IMG_SIZE = (512, 512, 3, 1, )
RGB_SIZE = (512, 512, 3, 3, )

random.seed(10)
temp_list = list(zip(imgs, segs))
random.shuffle(temp_list)
imgs, segs = zip(*temp_list)

# Set cross validation folds and example images
if NUM_FOLDS == 0:
    img_train = imgs
    seg_train = segs
    ex_indices = np.random.choice(len(img_train), NUM_EX)
    img_examples = np.array(img_train)[ex_indices]
    seg_examples = np.array(seg_train)[ex_indices]
    img_examples = [s.encode("utf-8") for s in img_examples]
    seg_examples = [s.encode("utf-8") for s in seg_examples]
else:
    num_in_fold = int(N / NUM_FOLDS)
    img_val = imgs[FOLD * num_in_fold:(FOLD + 1) * num_in_fold]
    seg_val = segs[FOLD * num_in_fold:(FOLD + 1) * num_in_fold]
    img_train = imgs[0:FOLD * num_in_fold] + imgs[(FOLD + 1) * num_in_fold:]
    seg_train = segs[0:FOLD * num_in_fold] + segs[(FOLD + 1) * num_in_fold:]
    ex_indices = np.random.choice(len(img_val), NUM_EX)
    img_examples = np.array(img_val)[ex_indices]
    seg_examples = np.array(seg_val)[ex_indices]
    img_examples = [s.encode("utf-8") for s in img_examples]
    seg_examples = [s.encode("utf-8") for s in seg_examples]

# Initialise model
UNet = UNetGen(input_shape=IMG_SIZE, starting_channels=NC, drop_rate=DROP_RATE)

# Create dataset
train_ds = tf.data.Dataset.from_generator(
    imgLoader, args=[img_path, seg_path, img_train, seg_train, priors, True, False], output_types=(tf.float32, tf.float32))

if NUM_FOLDS > 0:
    val_ds = tf.data.Dataset.from_generator(
        imgLoader, args=[img_path, seg_path, img_val, seg_val, priors, False, False], output_types=(tf.float32, tf.float32))

if arguments.file_path == None:
    print(UNet.summary())

# Create losses
train_metric = keras.metrics.BinaryCrossentropy(from_logits=False)
val_metric = keras.metrics.BinaryCrossentropy(from_logits=False)
train_metric = 0
val_metric = 0
train_count = 0
val_count = 0
Optimiser = keras.optimizers.Adam(ETA)

# Set start time
start_time = time.time()

# Training
for epoch in range(EPOCHS):
    for img, seg in train_ds.batch(MB_SIZE):
        if AUG_FLAG:
            trans_mat = DataAug.transMatGen(img.shape[0])
            img = affineTransformation(img, trans_mat)
            seg = affineTransformation(seg, trans_mat)

        train_metric += trainStep(img, seg, UNet, Optimiser)
        train_count += 1

    # Validation step if required
    if NUM_FOLDS > 0:
        for img, seg in val_ds.batch(MB_SIZE):
            val_metric += valStep(img, seg, UNet)
            val_count += 1
    else:
        val_count += 1e-6

    # Print losses every epoch
    print(f"Epoch: {epoch + 1}, Train Loss: {train_metric / train_count}, Val Loss: {val_metric / val_count}")
    log_file.write(f"Epoch: {epoch + 1}, Train Loss: {train_metric / train_count}, Val Loss: {val_metric / val_count}\n")
    train_metric = 0
    val_metric = 0

# CONVERT TO USE THESE AS EXAMPLE IMAGES
drop_imgs = np.zeros((NUM_EX, 512, 512, 3, 1), dtype=np.float32)
rgb_pred = np.zeros((NUM_EX, 512, 512, 3, 3), dtype=np.float32)

prior_bytes = [vol.encode("utf-8") for vol in priors]

for j in range(NUM_EX):
    for data in imgLoader(img_path.encode("utf-8"), seg_path.encode("utf-8"), [img_examples[j]], [seg_examples[j]], np.array(prior_bytes), False, True):
        drop_imgs[j, :, :, :, :] = data[0]
        rgb_pred[j, :, :, :, 0] = np.squeeze(data[1])

pred = UNet(drop_imgs, training=False)
pred_mean, pred_var, _ = varDropout(drop_imgs, UNet, T=100)
rgb_pred[:, :, :, :, 1] = pred[:, :, :, :, 0].numpy()
rgb_pred[:, :, :, :, 2] = pred[:, :, :, :, 0].numpy()

fig, axs = plt.subplots(NUM_EX, 4)

for j in range(NUM_EX):
    axs[j, 0].imshow(np.fliplr(drop_imgs[j, :, :, 1, 0].T), cmap='gray', vmin=0.12, vmax=0.18, origin='lower')
    axs[j, 0].axis('off')
    # axs[j, 0].imshow(np.fliplr(np.ma.masked_where(pred[j, :, :, 1, 0].numpy().T == False, pred[j, :, :, 1, 0].numpy().T)), cmap='hot', origin='lower')
    axs[j, 0].imshow(np.fliplr(pred[j, :, :, 1, 0].numpy().T), cmap='hot', alpha=0.3, origin='lower')
    axs[j, 0].axis('off')
    r_pred = np.fliplr(rgb_pred[j, :, :, 1, 0].T)
    g_pred = np.fliplr(rgb_pred[j, :, :, 1, 1].T)
    b_pred = np.fliplr(rgb_pred[j, :, :, 1, 2].T)
    axs[j, 1].imshow(np.concatenate([r_pred[:, :, np.newaxis], g_pred[:, :, np.newaxis], b_pred[:, :, np.newaxis]], axis=2), origin='lower')
    axs[j, 1].axis('off')
    axs[j, 2].imshow(np.fliplr(pred_mean[j, :, :, 1].T), cmap='hot', origin='lower')
    axs[j, 2].axis('off')
    axs[j, 3].imshow(np.fliplr(pred_var[j, :, :, 1].T), cmap='hot', origin='lower')
    axs[j, 3].axis('off')

fig.subplots_adjust(wspace=0.025, hspace=0.1)
plt.savefig(f"{IMAGE_SAVE_PATH}/Uncertainty.png", dpi=250)
plt.close()

if NUM_FOLDS == 0:
    UNet.save_weights(f"{MODEL_SAVE_PATH}{EXPT_NAME}")

log_file.write(f"Time: {(time.time() - start_time) / 60:.2f} min\n")
log_file.close()
