import load_data
import os
# import numpy as np
from train import SrganGeneratorTrainer, SrganTrainer
from model import generator, discriminator
from data_preprocessing import resolve_single, evaluate
import tensorflow as tf
import logging
from config import DOWNSCALE, DOWNSCALE_WAY, WEIGHT_DIR, COMPARISON_DIR, LOG_DIR
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from load_data import compare_and_plot
import cv2 as cv

# Location of model weights (needed for demo)
weights_dir = WEIGHT_DIR
weights_file = lambda filename: os.path.join(weights_dir, filename)

os.environ['CUDA_VISIBLE_DEVICES']='0'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

os.makedirs(weights_dir, exist_ok=True)

comparison_dir = COMPARISON_DIR
comparison_file = lambda filename: os.path.join(comparison_dir, filename)
os.makedirs(comparison_dir, exist_ok=True)

if __name__ == "__main__":
    # logging.basicConfig(filename=LOG_DIR, level=logging.INFO, format='%(asctime)s - %(message)s')

    bicubic_img = np.array(Image.open('dataset/images/DIV2K_valid_LR_bicubic/X4/0807x4.png'))
    unknown_img = np.array(Image.open('dataset/images/DIV2K_valid_LR_unknown/X4/0807x4.png'))

    fig = plt.figure(figsize=(12,4))
    ax = fig.add_subplot(1,2,1)
    ax.imshow(bicubic_img)
    ax.set_title('LR (x4) downsampled by bicubic')
    ax.set_xticks([])
    ax.set_yticks([])        
    ax = fig.add_subplot(1,2,2)
    ax.imshow(unknown_img)
    ax.set_title('LR (x4) downsampled by unknown')
    ax.set_xticks([])
    ax.set_yticks([])    
    plt.tight_layout()
    fig.savefig('results/comparison/bicubic_unknown.png')






    '''''''''
    plot raw images comparison
    '''''''''
    # hr_img = np.array(Image.open('dataset/images/DIV2K_valid_HR/0802.png'))
    # lr_x2_img = np.array(Image.open('dataset/images/DIV2K_valid_LR_bicubic/X2/0802x2.png'))
    # lr_x3_img = np.array(Image.open('dataset/images/DIV2K_valid_LR_bicubic/X3/0802x3.png'))
    # lr_x4_img = np.array(Image.open('dataset/images/DIV2K_valid_LR_bicubic/X4/0802x4.png'))

    # fig = plt.figure(figsize=(12, 8))

    # images = [hr_img, lr_x2_img, lr_x3_img, lr_x4_img]
    # titles = ['HR', 'LR (x2)', 'LR (x3)', 'LR (x4)']
    # positions = [1, 2, 3, 4]
    # downscale = [1/2, 1/3, 1/4]

    # for i, (img, title, pos) in enumerate(zip(images, titles, positions)):
    #     ogsize = 100
    #     resize = 600
    #     upper = 400
    #     left = 400
    #     upper_r = upper
    #     left_r = int(left + 1.25 * ogsize)

    #     bottom = upper + ogsize
    #     right = left + ogsize
    #     bottom_r = upper_r + resize
    #     right_r = left_r + resize

    #     if i > 0:
    #         resize = int(resize * downscale[i-1])
    #         ogsize = int(ogsize * downscale[i-1])
    #         upper = int(upper * downscale[i-1])
    #         left = int(left * downscale[i-1])
    #         bottom = int(upper + ogsize)
    #         right = int(left + ogsize)
    #         upper_r = int(upper_r * downscale[i-1])
    #         left_r = int(left_r * downscale[i-1])
    #         bottom_r = int(upper_r + resize)
    #         right_r = int(left_r + resize)

    #     img_np = img.numpy() if isinstance(img,
    #                                        tf.Tensor) else img  # Convert to NumPy array if it's a TensorFlow tensor
    #     part = img_np[upper:bottom, left:right]
    #     part = part.astype(np.float32)  # Convert to float32
    #     mask = cv.resize(part, (resize, resize), interpolation=cv.INTER_LINEAR)
    #     img_np[upper_r:bottom_r, left_r:right_r] = mask
    #     cv.rectangle(img_np, (left, bottom), (right, upper), (0, 255, 0), thickness=2)
    #     cv.rectangle(img_np, (left_r, bottom_r), (right_r, upper_r), (0, 255, 0), thickness=2)
    #     img_np = cv.line(img_np, (right, bottom), (left_r, bottom_r), (0, 255, 0), thickness=2)
    #     img_np = cv.line(img_np, (right, upper), (left_r, upper_r), (0, 255, 0), thickness=2)
    #     ax = fig.add_subplot(2, 2, pos)
    #     ax.imshow(img_np)
    #     ax.set_title(title)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    # plt.tight_layout()
    # fig.savefig('results/comparison/hr_lr.png')


    '''''''''
    Evaluation
    '''''''''
    # valid = load_data.DIV2K(scale=DOWNSCALE, downgrade=DOWNSCALE_WAY, subset='valid')
    # valid_dataset = valid.dataset(batch_size=1, random_transform=False, repeat_count=1)
    #
    # gan_generator = generator()
    # gan_generator.load_weights(weights_file(f'gan_generator_x{DOWNSCALE}.h5'))
    #
    # psnr_valid = evaluate(gan_generator, valid_dataset, 'psnr')
    # print(f"PSNR in {DOWNSCALE_WAY}_x{DOWNSCALE}_valid_dataset: {psnr_valid}")
    # logging.info(f"PSNR in {DOWNSCALE_WAY}_x{DOWNSCALE}_valid_dataset: {psnr_valid}")
    #
    # ssim_valid = evaluate(gan_generator, valid_dataset, 'ssim')
    # print(f"SSIM in {DOWNSCALE_WAY}_x{DOWNSCALE}_valid_dataset: {ssim_valid}")
    # logging.info(f"SSIM in {DOWNSCALE_WAY}_x{DOWNSCALE}_valid_dataset: {ssim_valid}")

    '''''''''
    plot SR comparison
    '''''''''
    # try:
    #     fig = compare_and_plot(DOWNSCALE, DOWNSCALE_WAY, generator(), generator())
    #     fig.savefig(comparison_file(f'comparison_{DOWNSCALE_WAY}_x{DOWNSCALE}.png'))
    # except Exception as e:
    #     print(e)


    '''''''''
    training
    '''''''''
    # train = load_data.DIV2K(scale=DOWNSCALE, downgrade=DOWNSCALE_WAY, subset='train')
    # valid = load_data.DIV2K(scale=DOWNSCALE, downgrade=DOWNSCALE_WAY, subset='valid')
    # train_dataset = train.dataset(batch_size=32, random_transform=True)
    # valid_dataset = valid.dataset(batch_size=1, random_transform=False, repeat_count=1)
    
    
    # # generator pre-training
    # logging.info("Start pre-training")
    # pre_trainer = SrganGeneratorTrainer(model=generator(), checkpoint_dir=f'.ckpt/pre_generator_{DOWNSCALE_WAY}_x{DOWNSCALE}')
    
    # pre_trainer.train(train_dataset,
    #               valid_dataset.take(10),
    #               steps=10000,
    #               evaluate_every=1000,
    #               save_best_only=False)
    
    # pre_trainer.model.save_weights(weights_file(f'pre_generator_{DOWNSCALE_WAY}_x{DOWNSCALE}.h5'))
    
    # # generator fine tuning
    # gan_generator = generator()
    # gan_generator.load_weights(weights_file(f'pre_generator_{DOWNSCALE_WAY}_x{DOWNSCALE}.h5'))
    # logging.info("Start training GAN")
    # gan_trainer = SrganTrainer(generator=gan_generator, discriminator=discriminator())
    # loss_result = gan_trainer.train(train_dataset, steps=20000)
    
    # gan_trainer.generator.save_weights(weights_file(f'gan_generator_{DOWNSCALE_WAY}_x{DOWNSCALE}.h5'))
    # gan_trainer.discriminator.save_weights(weights_file(f'gan_discriminator_{DOWNSCALE_WAY}_x{DOWNSCALE}.h5'))
    # logging.info("Training finished")
