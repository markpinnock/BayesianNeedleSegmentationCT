import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time


""" Based on work by Tensorflow authors, found at:
    https://github.com/daviddao/spatial-transformer-tensorflow/blob/master/spatial_transformer.py """


def affineTransformation(input_vol, thetas):
    """ input_vol: 3D img volume (mb, height, width, depth, nc)
        thetas: 2x2 matrix for transform (mb, 2, 2) """
    """ NB: only performs 2D transformations on 3D volume!!! """

    mb_size = input_vol.shape[0]
    height = input_vol.shape[1]
    width = input_vol.shape[2]
    depth = input_vol.shape[3]

    # Generate flattened coordinates and transform
    flat_coords = coordGen(mb_size, height, width, depth)
    trans_mat = tf.concat([thetas, tf.zeros([mb_size, 2, 1])], axis=2)
    trans_mat = tf.concat([trans_mat, tf.tile(tf.constant([[[0.0, 0.0, 1.0]]]), [mb_size, 1, 1])], axis=1)
    new_coords = tf.matmul(trans_mat, flat_coords)

    # Unroll entire coords
    # These are 1D vectors containing consecutive subsections for each img
    # E.g. X_new = [img1_y1...img1_yn, img2_y1...img2_yn, ... imgn_y1...imgn_yn]
    X_new = tf.reshape(new_coords[:, 0, :], [-1])
    Y_new = tf.reshape(new_coords[:, 1, :], [-1])

    # Perform interpolation on input_vol
    output_vol = interpolate(input_vol, X_new, Y_new, mb_size)

    return output_vol


def coordGen(mb_size, height, width, depth):
    """ Generate coordinates (mb, 3, height * width)
        3rd dim consists of height * width rows for X, Y and ones """
    
    height_f = tf.cast(height, tf.float32)
    width_f = tf.cast(width, tf.float32)
    X, Y = tf.meshgrid(tf.linspace(-1.0, 1.0, width), tf.linspace(-1.0, 1.0, height))
    flat_X = tf.reshape(X, (1, -1))
    flat_Y = tf.reshape(Y, (1, -1))

    # Rows are X, Y, Z and row of ones (row length is height * width * depth)
    # Replicate for each minibatch
    flat_coords = tf.concat([flat_X, flat_Y, tf.ones((1, height_f * width_f))], axis=0)
    flat_coords = tf.tile(flat_coords[tf.newaxis, :, :], [mb_size, 1, 1])

    return flat_coords


def interpolate(input_vol, X, Y, mb_size):
    """ Performs interpolation input_vol, using deformation fields X, Y, Z """

    height = input_vol.shape[1]
    width = input_vol.shape[2]
    nc = input_vol.shape[4]
    height_f = tf.cast(height, tf.float32)
    width_f = tf.cast(width, tf.float32)

    # Convert to image coords
    X = (X + 1.0) * width_f / 2.0
    Y = (Y + 1.0) * height_f / 2.0

    # Generate integer coord indices either side of actual value
    x0 = tf.cast(tf.floor(X), tf.int32)
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(Y), tf.int32)
    y1 = y0 + 1

    # Ensure indices don't extend past image height/width
    x0 = tf.clip_by_value(x0, 0, width - 1)
    x1 = tf.clip_by_value(x1, 0, width - 1)
    y0 = tf.clip_by_value(y0, 0, height - 1)
    y1 = tf.clip_by_value(y1, 0, height - 1)

    # Creates a vector of base indices corresponding to each img in mb
    # Allows finding index in unrolled image vector
    base = tf.matmul(tf.reshape(tf.range(mb_size) * height * width, [-1, 1]), tf.cast(tf.ones((1, height * width)), tf.int32))
    base = tf.reshape(base, [-1])

    # Generate 4 vectors of indices corresponding to x0, y0, x1, y1 around base indices
    base_y0 = base + y0 * width
    base_y1 = base + y1 * width
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    # Flatten image vector and look up with indices (image vector arranged z1, z2, z3, z1, z2, z3, etc...)
    # Gather pixel values from flattened img based on 4 index vectors
    input_vol_flat = tf.reshape(input_vol, [-1, nc])
    input_vol_flat = tf.cast(input_vol_flat, tf.float32)

    ImgA1 = tf.gather(input_vol_flat[0::3], idx_a)
    ImgA2 = tf.gather(input_vol_flat[1::3], idx_a)
    ImgA3 = tf.gather(input_vol_flat[2::3], idx_a)
    ImgB1 = tf.gather(input_vol_flat[0::3], idx_b)
    ImgB2 = tf.gather(input_vol_flat[1::3], idx_b)
    ImgB3 = tf.gather(input_vol_flat[2::3], idx_b)
    ImgC1 = tf.gather(input_vol_flat[0::3], idx_c)
    ImgC2 = tf.gather(input_vol_flat[1::3], idx_c)
    ImgC3 = tf.gather(input_vol_flat[2::3], idx_c)
    ImgD1 = tf.gather(input_vol_flat[0::3], idx_d)
    ImgD2 = tf.gather(input_vol_flat[1::3], idx_d)
    ImgD3 = tf.gather(input_vol_flat[2::3], idx_d)


    # Generate vectors of the fractional difference between original indices and rounded indices i.e. weights
    x0_f = tf.cast(x0, 'float32')
    x1_f = tf.cast(x1, 'float32')
    y0_f = tf.cast(y0, 'float32')
    y1_f = tf.cast(y1, 'float32')

    wa = ((x1_f - X) * (y1_f - Y))[:, tf.newaxis]
    wb = ((x1_f - X) * (Y - y0_f))[:, tf.newaxis]
    wc = ((X - x0_f) * (y1_f - Y))[:, tf.newaxis]
    wd = ((X - x0_f) * (Y - y0_f))[:, tf.newaxis]

    # Add weighted imgs from each of four indices and return img_vol
    output_vol_1 = tf.reshape(tf.add_n([wa * ImgA1, wb * ImgB1, wc * ImgC1, wd * ImgD1]), [mb_size, height, width])
    output_vol_2 = tf.reshape(tf.add_n([wa * ImgA2, wb * ImgB2, wc * ImgC2, wd * ImgD2]), [mb_size, height, width])
    output_vol_3 = tf.reshape(tf.add_n([wa * ImgA3, wb * ImgB3, wc * ImgC3, wd * ImgD3]), [mb_size, height, width])

    return tf.stack([output_vol_1, output_vol_2, output_vol_3], axis=3)[:, :, :, :, tf.newaxis]


if __name__ == "__main__":
    from TransGen import TransMatGen

    start_t = time.time()
    img_vol = np.zeros((4, 128, 128, 3, 1))
    img_vol[:, 40:88, 40:88, :, :] = 1

    TestMatGen = TransMatGen()
    new_vol = affineTransformation(img_vol, TestMatGen.transMatGen(4))
    print(time.time() - start_t)

    for i in range(0, 3):
        fig, axs = plt.subplots(2, 4)
        axs[0, 0].imshow(img_vol[0, :, :, i, 0])
        axs[0, 1].imshow(img_vol[1, :, :, i, 0])
        axs[0, 2].imshow(img_vol[2, :, :, i, 0])
        axs[0, 3].imshow(img_vol[3, :, :, i, 0])
        axs[1, 0].imshow(new_vol[0, :, :, i, 0])
        axs[1, 1].imshow(new_vol[1, :, :, i, 0])
        axs[1, 2].imshow(new_vol[2, :, :, i, 0])
        axs[1, 3].imshow(new_vol[3, :, :, i, 0])
        plt.show()
    
    img1 = np.load("test1.npy")
    img2 = np.load("test2.npy")
    img3 = np.load("test3.npy")
    img4 = np.load("test4.npy")
    imgs = np.stack([img1, img2, img3, img4], axis=0)[:, :, :, :, np.newaxis]

    TestMatGen2 = TransMatGen()
    new_imgs = affineTransformation(imgs, TestMatGen.transMatGen(4))
    print(time.time() - start_t)

    for i in range(0, 3):
        fig, axs = plt.subplots(2, 4)
        axs[0, 0].imshow(np.fliplr(imgs[0, :, :, i, 0].T), origin='lower', cmap='gray', vmin=0.12, vmax=0.18)
        axs[0, 1].imshow(np.fliplr(imgs[1, :, :, i, 0].T), origin='lower', cmap='gray', vmin=0.12, vmax=0.18)
        axs[0, 2].imshow(np.fliplr(imgs[2, :, :, i, 0].T), origin='lower', cmap='gray', vmin=0.12, vmax=0.18)
        axs[0, 3].imshow(np.fliplr(imgs[3, :, :, i, 0].T), origin='lower', cmap='gray', vmin=0.12, vmax=0.18)
        axs[1, 0].imshow(np.fliplr(new_imgs[0, :, :, i, 0].numpy().T), origin='lower', cmap='gray', vmin=0.12, vmax=0.18)
        axs[1, 1].imshow(np.fliplr(new_imgs[1, :, :, i, 0].numpy().T), origin='lower', cmap='gray', vmin=0.12, vmax=0.18)
        axs[1, 2].imshow(np.fliplr(new_imgs[2, :, :, i, 0].numpy().T), origin='lower', cmap='gray', vmin=0.12, vmax=0.18)
        axs[1, 3].imshow(np.fliplr(new_imgs[3, :, :, i, 0].numpy().T), origin='lower', cmap='gray', vmin=0.12, vmax=0.18)
        plt.show()