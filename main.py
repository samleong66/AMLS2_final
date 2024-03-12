import load_data
import os
# import numpy as np
from train import SrganGeneratorTrainer, SrganTrainer
from model import generator, discriminator
from data_preprocessing import resolve_single, evaluate
import tensorflow as tf
import logging
from config import DOWNSCALE, DOWNSCALE_WAY
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from load_data import compare_and_plot

# Location of model weights (needed for demo)
weights_dir = 'models/weights/srgan'
weights_file = lambda filename: os.path.join(weights_dir, filename)

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

os.makedirs(weights_dir, exist_ok=True)

comparison_dir = 'results/comparison'
comparison_file = lambda filename: os.path.join(comparison_dir, filename)
os.makedirs(comparison_dir, exist_ok=True)

if __name__ == "__main__":
    logging.basicConfig(filename=f'results/training_x{DOWNSCALE}.log', level=logging.INFO,
                        format='%(asctime)s - %(message)s')

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
    plot comparison
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
    # train_dataset = train.dataset(batch_size=8, random_transform=True)
    # valid_dataset = valid.dataset(batch_size=1, random_transform=False, repeat_count=1)
    #
    #
    # # generator pre-training
    # logging.info("Start pre-training")
    # pre_trainer = SrganGeneratorTrainer(model=generator(), checkpoint_dir=f'.ckpt/pre_generator_x{DOWNSCALE}')
    #
    # pre_trainer.train(train_dataset,
    #               valid_dataset.take(10),
    #               steps=10000,
    #               evaluate_every=1000,
    #               save_best_only=False)
    #
    # pre_trainer.model.save_weights(weights_file(f'pre_generator_x{DOWNSCALE}.h5'))
    #
    # # generator fine tuning
    # gan_generator = generator()
    # gan_generator.load_weights(weights_file(f'pre_generator_x{DOWNSCALE}.h5'))
    # logging.info("Start training GAN")
    # gan_trainer = SrganTrainer(generator=gan_generator, discriminator=discriminator())
    # loss_result = gan_trainer.train(train_dataset, steps=20000)
    #
    # gan_trainer.generator.save_weights(weights_file(f'gan_generator_x{DOWNSCALE}.h5'))
    # gan_trainer.discriminator.save_weights(weights_file(f'gan_discriminator_x{DOWNSCALE}.h5'))
    # logging.info("Training finished")
