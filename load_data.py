import os

import requests, zipfile, tqdm
import tensorflow as tf

from tensorflow.python.data.experimental import AUTOTUNE
from data_preprocessing import random_crop, random_flip, random_rotate
from data_preprocessing import resolve_single

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv


class DIV2K:
    def __init__(self,
                 scale=2,
                 subset='train',
                 downgrade='bicubic',
                 images_dir='Dataset/images',
                 caches_dir='Dataset/caches'):

        self._ntire_2018 = False

        _scales = [2, 3, 4, 8]

        if scale in _scales:
            self.scale = scale
        else:
            raise ValueError(f'scale must be in ${_scales}')

        if subset == 'train':
            self.image_ids = range(1, 801)
        elif subset == 'valid':
            self.image_ids = range(801, 901)
        else:
            raise ValueError("subset must be 'train' or 'valid'")

        _downgrades_a = ['bicubic', 'unknown']
        _downgrades_b = ['mild', 'difficult']

        if scale == 8 and downgrade != 'bicubic':
            raise ValueError(f'scale 8 only allowed for bicubic downgrade')

        if downgrade in _downgrades_b and scale != 4:
            raise ValueError(f'{downgrade} downgrade requires scale 4')

        if downgrade == 'bicubic' and scale == 8:
            self.downgrade = 'x8'
        elif downgrade in _downgrades_b:
            self.downgrade = downgrade
        else:
            self.downgrade = downgrade
            self._ntire_2018 = False

        self.subset = subset
        self.images_dir = images_dir
        self.caches_dir = caches_dir

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(caches_dir, exist_ok=True)

    def __len__(self):
        return len(self.image_ids)

    def dataset(self, batch_size=16, repeat_count=None, random_transform=True):
        ds = tf.data.Dataset.zip((self.lr_dataset(), self.hr_dataset()))
        if random_transform:
            ds = ds.map(lambda lr, hr: random_crop(lr, hr, scale=self.scale), num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_flip, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.repeat(repeat_count)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def hr_dataset(self):
        if not os.path.exists(self._hr_images_dir()):
            download_archive(self._hr_images_archive(), self.images_dir, extract=True)

        ds = self._images_dataset(self._hr_image_files()).cache(self._hr_cache_file())

        if not os.path.exists(self._hr_cache_index()):
            self._populate_cache(ds, self._hr_cache_file())

        return ds

    def lr_dataset(self):
        if not os.path.exists(self._lr_images_dir()):
            download_archive(self._lr_images_archive(), self.images_dir, extract=True)

        ds = self._images_dataset(self._lr_image_files()).cache(self._lr_cache_file())

        if not os.path.exists(self._lr_cache_index()):
            self._populate_cache(ds, self._lr_cache_file())

        return ds

    def _hr_cache_file(self):
        return os.path.join(self.caches_dir, f'DIV2K_{self.subset}_HR.cache')

    def _lr_cache_file(self):
        return os.path.join(self.caches_dir, f'DIV2K_{self.subset}_LR_{self.downgrade}_X{self.scale}.cache')

    def _hr_cache_index(self):
        return f'{self._hr_cache_file()}.index'

    def _lr_cache_index(self):
        return f'{self._lr_cache_file()}.index'

    def _hr_image_files(self):
        images_dir = self._hr_images_dir()
        return [os.path.join(images_dir, f'{image_id:04}.png') for image_id in self.image_ids]

    def _lr_image_files(self):
        images_dir = self._lr_images_dir()
        return [os.path.join(images_dir, self._lr_image_file(image_id)) for image_id in self.image_ids]

    def _lr_image_file(self, image_id):
        if not self._ntire_2018 or self.scale == 8:
            return f'{image_id:04}x{self.scale}.png'
        else:
            return f'{image_id:04}x{self.scale}{self.downgrade[0]}.png'

    def _hr_images_dir(self):
        return os.path.join(self.images_dir, f'DIV2K_{self.subset}_HR')

    def _lr_images_dir(self):
        if self._ntire_2018:
            return os.path.join(self.images_dir, f'DIV2K_{self.subset}_LR_{self.downgrade}')
        else:
            return os.path.join(self.images_dir, f'DIV2K_{self.subset}_LR_{self.downgrade}', f'X{self.scale}')

    def _hr_images_archive(self):
        return f'DIV2K_{self.subset}_HR.zip'

    def _lr_images_archive(self):
        if self._ntire_2018:
            return f'DIV2K_{self.subset}_LR_{self.downgrade}.zip'
        else:
            return f'DIV2K_{self.subset}_LR_{self.downgrade}_X{self.scale}.zip'

    @staticmethod
    def _images_dataset(image_files):
        ds = tf.data.Dataset.from_tensor_slices(image_files)
        ds = ds.map(tf.io.read_file)
        ds = ds.map(lambda x: tf.image.decode_png(x, channels=3), num_parallel_calls=AUTOTUNE)
        return ds

    @staticmethod
    def _populate_cache(ds, cache_file):
        print(f'Caching decoded images in {cache_file} ...')
        for _ in ds: pass
        print(f'Cached decoded images in {cache_file}.')



def download_archive(file, target_dir, extract=True):
    source_url = f'http://data.vision.ee.ethz.ch/cvl/DIV2K/{file}'
    target_dir = os.path.abspath(target_dir)
    tf.keras.utils.get_file(file, source_url, cache_subdir=target_dir, extract=extract)
    os.remove(os.path.join(target_dir, file))


def compare_and_plot(downscale: int, downscale_way: str, pre_generator, gan_generator):
    weights_dir = 'models/weights/srgan'
    weights_file = lambda filename: os.path.join(weights_dir, filename)
    lr_image_path = f'Dataset/images/DIV2K_valid_LR_{downscale_way}/X{downscale}'
    hr_image_path = r'Dataset/images/DIV2K_valid_HR'

    num = 897
    lr_img = np.array(Image.open(lr_image_path + '/0' + str(num) + f'x{downscale}.png'))
    hr_img = np.array(Image.open(hr_image_path + '/0' + str(num) + '.png'))

    pre_generator.load_weights(weights_file(f'pre_generator_bicubic_x{downscale}.h5'))
    gan_generator.load_weights(weights_file(f'gan_generator_bicubic_x{downscale}.h5'))

    gan_sr = resolve_single(gan_generator, lr_img)
    pre_sr = resolve_single(pre_generator, lr_img)

    fig = plt.figure(figsize=(12, 8))

    images = [lr_img, pre_sr, gan_sr, hr_img]
    titles = ['LR', 'SR (PRE)', 'SR (GAN)', 'HR']
    positions = [1, 2, 3, 4]


    for i, (img, title, pos) in enumerate(zip(images, titles, positions)):
        ogsize = 30
        resize = 250
        upper = 50
        left = 50
        upper_r = upper
        left_r = int(left + 1.25 * ogsize)

        bottom = upper + ogsize
        right = left + ogsize
        bottom_r = upper_r + resize
        right_r = left_r + resize

        if i > 0:
            resize = int(resize * downscale)
            ogsize = int(ogsize * downscale)
            upper = upper * downscale
            left = left * downscale
            bottom = int(upper + ogsize)
            right = int(left + ogsize)
            upper_r = upper_r * downscale
            left_r = left_r * downscale
            bottom_r = int(upper_r + resize)
            right_r = int(left_r + resize)

        img_np = img.numpy() if isinstance(img,
                                           tf.Tensor) else img  # Convert to NumPy array if it's a TensorFlow tensor
        part = img_np[upper:bottom, left:right]
        part = part.astype(np.float32)  # Convert to float32
        mask = cv.resize(part, (resize, resize), interpolation=cv.INTER_LINEAR)
        img_np[upper_r:bottom_r, left_r:right_r] = mask
        cv.rectangle(img_np, (left, bottom), (right, upper), (0, 255, 0), thickness=2)
        cv.rectangle(img_np, (left_r, bottom_r), (right_r, upper_r), (0, 255, 0), thickness=2)
        img_np = cv.line(img_np, (right, bottom), (left_r, bottom_r), (0, 255, 0), thickness=2)
        img_np = cv.line(img_np, (right, upper), (left_r, upper_r), (0, 255, 0), thickness=2)
        ax = fig.add_subplot(2, 2, pos)
        ax.imshow(img_np)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()

    return fig

    
def download_set14():
    url = 'https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip'
    target_dir = 'Dataset/images'

    # Create target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Download the zip file with progress bar
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm.tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(os.path.join(target_dir, 'temp.zip'), 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

    # Extract the zip file
    with zipfile.ZipFile(os.path.join(target_dir, 'temp.zip'), 'r') as zip_ref:
        zip_ref.extractall(target_dir)

    # Delete the zip file
    os.remove(os.path.join(target_dir, 'temp.zip'))
    print("Downloaded and extracted successfully!")

def load_set14(scale):
    images_dir = f'Dataset/images/Set14/image_SRF_{scale}'
    lr_image_files = [os.path.join(images_dir, f'img_0{str(id).zfill(2)}_SRF_{scale}_LR.png') for id in range(1, 15)]
    hr_image_files = [os.path.join(images_dir, f'img_0{str(id).zfill(2)}_SRF_{scale}_HR.png') for id in range(1, 15)]
    lr_ds = tf.data.Dataset.from_tensor_slices(lr_image_files)
    hr_ds = tf.data.Dataset.from_tensor_slices(hr_image_files)
    lr_ds = lr_ds.map(tf.io.read_file)
    lr_ds = lr_ds.map(lambda x: tf.image.decode_png(x, channels=3), num_parallel_calls=AUTOTUNE)
    hr_ds = hr_ds.map(tf.io.read_file)
    hr_ds = hr_ds.map(lambda x: tf.image.decode_png(x, channels=3), num_parallel_calls=AUTOTUNE)
    ds = tf.data.Dataset.zip((lr_ds, hr_ds))
    ds = ds.batch(1)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds