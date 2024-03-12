import tensorflow as tf
import numpy as np
import logging
from skimage.metrics import structural_similarity
import random

DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255

from config import HR_SIZE, DOWNSCALE


def random_crop(lr_img, hr_img, hr_crop_size=HR_SIZE, scale=DOWNSCALE):
    lr_crop_size = hr_crop_size // scale
    lr_img_shape = tf.shape(lr_img)[:2]

    lr_w = tf.random.uniform(shape=(), maxval=lr_img_shape[1] - lr_crop_size + 1, dtype=tf.int32)
    lr_h = tf.random.uniform(shape=(), maxval=lr_img_shape[0] - lr_crop_size + 1, dtype=tf.int32)

    hr_w = lr_w * scale
    hr_h = lr_h * scale

    lr_img_cropped = lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
    hr_img_cropped = hr_img[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]

    return lr_img_cropped, hr_img_cropped


def random_flip(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(rn < 0.5,
                   lambda: (lr_img, hr_img),
                   lambda: (tf.image.flip_left_right(lr_img),
                            tf.image.flip_left_right(hr_img)))


def random_rotate(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lr_img, rn), tf.image.rot90(hr_img, rn)


def resolve_single(model, lr):
    return resolve(model, tf.expand_dims(lr, axis=0))[0]


def resolve(model, lr_batch):
    lr_batch = tf.cast(lr_batch, tf.float32)
    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)
    return sr_batch


def evaluate(model, dataset, metric):
    if metric == 'psnr':
        psnr_values = []
        dataset_num = len(dataset)
        i = 1
        for lr, hr in dataset:
            sr = resolve(model, lr)
            psnr_value = psnr(hr, sr)[0]
            logging.info(f"{i}/{dataset_num}, PSNR: {psnr_value}")
            i += 1
            psnr_values.append(psnr_value)
        return tf.reduce_mean(psnr_values)
    elif metric == 'ssim':
        ssim_values = []
        dataset_num = len(dataset)
        i = 1
        for lr, hr in dataset:
            sr = resolve(model, lr)
            ssim_value = ssim(hr, sr)
            logging.info(f"{i}/{dataset_num}, SSIM: {ssim_value}")
            i += 1
            ssim_values.append(ssim_value)
        return tf.reduce_mean(ssim_values)


def normalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return (x - rgb_mean) / 127.5


def denormalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return x * 127.5 + rgb_mean


def normalize_01(x):
    """Normalizes RGB images to [0, 1]."""
    return x / 255.0


def normalize_m11(x):
    """Normalizes RGB images to [-1, 1]."""
    return x / 127.5 - 1


def denormalize_m11(x):
    """Inverse of normalize_m11."""
    return (x + 1) * 127.5


def psnr(x1, x2):
    return tf.image.psnr(x1, x2, max_val=255)


def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)


def ssim(hr, sr):
    # size = 100
    # h, w = hr.shape[1:3]
    # if h < size or w < size:
    #     print("Error: Image dimensions are smaller than crop size")
    #     return
    # max_x = w - size
    # max_y = h - size
    # start_x = random.randint(0, max_x)
    # start_y = random.randint(0, max_y)
    # results = []
    # for i in range(10):
    #     hr_c, sr_c = hr[:, start_y:start_y+size, start_x:start_x+size, :][0].numpy(), sr[:, start_y:start_y+size, start_x:start_x+size, :][0].numpy()
    #     result = structural_similarity(im1=hr_c, im2=sr_c, win_size=3, multichannel=True, data_range=(hr_c.max() - hr_c.min()))
    #     if result != np.nan:
    #         results.append(result)
    #
    # return np.mean(np.array(results))

    return structural_similarity(im1=hr[0].numpy(), im2=sr[0].numpy(), win_size=3, multichannel=True, data_range=(hr.numpy().max() - hr.numpy().min()))