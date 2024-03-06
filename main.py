import load_data
import os
from train import SrganGeneratorTrainer, SrganTrainer
from model import generator, discriminator
import tensorflow as tf

# Location of model weights (needed for demo)
weights_dir = 'models/weights/srgan'
weights_file = lambda filename: os.path.join(weights_dir, filename)

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

os.makedirs(weights_dir, exist_ok=True)

if __name__ == "__main__":
    train = load_data.DIV2K(scale=2, downgrade='bicubic', subset='train')
    valid = load_data.DIV2K(scale=2, downgrade='bicubic', subset='valid')
    train_dataset = train.dataset(batch_size=32, random_transform=True)
    valid_dataset = valid.dataset(batch_size=1, random_transform=False, repeat_count=1)

    # for i in train_dataset:
    #     print(generator(tf.cast(i[0], tf.float32)))
    #     break
    
    # generator pre-training

    pre_trainer = SrganGeneratorTrainer(model=generator(), checkpoint_dir=f'.ckpt/pre_generator')
    pre_trainer.train(train_dataset,
                  valid_dataset.take(10),
                  steps=10000,
                  evaluate_every=1000, 
                  save_best_only=False)

    pre_trainer.model.save_weights(weights_file('pre_generator.h5'))  

    # generator fine tuning
    gan_generator = generator()
    gan_generator.load_weights(weights_file('pre_generator.h5'))

    gan_trainer = SrganTrainer(generator=gan_generator, discriminator=discriminator())
    loss_result = gan_trainer.train(train_dataset, steps=20000)

    gan_trainer.generator.save_weights(weights_file('gan_generator.h5'))
    gan_trainer.discriminator.save_weights(weights_file('gan_discriminator.h5'))