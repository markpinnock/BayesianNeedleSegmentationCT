from argparse import ArgumentParser
import matplotlib.pyplot as plt
import nrrd
import numpy as np
import os


parser = ArgumentParser()
parser.add_argument('--file_path', '-f', help="File path", type=str)
parser.add_argument('--save_path', '-s', help='Save path', type=str)
parser.add_argument('--review', '-r', help="Review vols y/n", type=str, nargs='?', const='n', default='n')
arguments = parser.parse_args()

if arguments.file_path == None:
    raise ValueError("Must provide file path")
else:
    FILE_PATH = arguments.file_path + '/'

if arguments.save_path == None:
    raise ValueError("Must provide save path")
else:
    SAVE_PATH = arguments.save_path + '/'

if arguments.review == 'y':
    review_flag = True
else:
    review_flag = False

IMG_PATH = f"{FILE_PATH}SparseVolImgs/"
SEG_PATH = f"{FILE_PATH}SparseVolMasks/"
subject_list = os.listdir(SEG_PATH)
subject_list.sort()

TOSHIBA_RANGE = [-2917, 16297]
min_val = TOSHIBA_RANGE[0]
max_val = TOSHIBA_RANGE[1]

for subject in subject_list:
    subject_img_path = f"{IMG_PATH}{subject}/"
    subject_seg_path = f"{SEG_PATH}{subject}/"
    seg_list = os.listdir(subject_seg_path)

    for seg in seg_list:
        seg_vol, _ = nrrd.read(f"{subject_seg_path}/{seg}")
        seg_vol = seg_vol[0, ...]

        img = seg[:-10] + "L.nrrd"
    
        try:
            img_vol, _ = nrrd.read(f"{subject_img_path}/{img}")
        except:
            print(f"NO CORRESPONDING MASK FOUND: {seg} {img}")
        else:
            img_vol = (img_vol - min_val) / (max_val - min_val)

            if img_vol.shape != seg_vol.shape:
                print(f"{img} {seg} INCORRECT DIMENSION")

            if review_flag:
                for i in range(seg_vol.shape[2]):
                    fig, axs = plt.subplots(1, 3)                   
                    axs[0].imshow(np.fliplr(img_vol[:, :, i].T), vmin=0.12, vmax=0.18, cmap='gray', origin='lower')
                    axs[1].imshow(np.fliplr(seg_vol[:, :, i].T), vmin=0.12, vmax=0.18, cmap='gray', origin='lower')
                    axs[2].imshow(np.fliplr(img_vol[:, :, i].T * seg_vol[:, :, i].T), vmin=0.12, vmax=0.18, cmap='gray', origin='lower')

                    plt.show()
                    plt.close()
            
            np.save(f"{SAVE_PATH}Img/{img[:-5]}.npy", img_vol)
            np.save(f"{SAVE_PATH}Seg/{seg[:-9]}.npy", seg_vol)
            print(f"{img[:-5]}.npy, {seg[:-9]}.npy CONVERTED")

if len(os.listdir(f"{SAVE_PATH}Img/")) != len(os.listdir(f"{SAVE_PATH}Seg/")):
    raise ValueError("Unequal number of converted vols")
