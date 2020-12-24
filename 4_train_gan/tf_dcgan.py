#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 06:08:23 2020

@author: ladvien
"""

"""
Code borrowed from: https://www.kaggle.com/amyjang/creating-romantic-period-art-w-tensorflow-dcgan

"""

import re
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
from IPython import display
import PIL

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Device:', tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print('Number of replicas:', strategy.num_replicas_in_sync)
    
print(tf.__version__)



#############
# Parameters
#############

dataroot        = "/home/ladvien/deep_arcane/images/data/training/"

OUTPUT_FOLDER   = "/home/ladvien/Desktop/deep_arcane_output/"

IMAGE_SIZE      = [128, 128]
AUTOTUNE        = tf.data.experimental.AUTOTUNE
BATCH_SIZE      = 16

COLOR_CHANNELS  = 1

TRUTH_SMOOTH    = 0.1
FAKE_SMOOTH     = 0.0

EPOCHS          = 25000
D_DROP_PERC     = 0.0
G_INPUT_DIMS    = 100

SAMPLES         = 16

PREVIEW_EVERY   = 10

G_LEARNING_RATE = 0.0002
D_LEARNING_RATE = 0.0002


##########
# DCGAN
##########
filenames = tf.io.gfile.glob(f"{dataroot}*")

IMG_COUNT = len(filenames)
print("Image count for training: " + str(IMG_COUNT))

train_ds = tf.data.Dataset.from_tensor_slices(filenames)

    
# normalizing the images to [-1, 1]
def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image

def decode_img(img, nc):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=nc)
  # Use `convert_image_dtype` to convert to floats in the [-1,1] range.
  img = normalize(img)
  # resize the image to the desired size.
  return tf.image.resize(img, IMAGE_SIZE)

def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = decode_img(img, nc = COLOR_CHANNELS)
    # convert the image to grayscale
    return tf.expand_dims(img[:, :, 0], axis=2)


train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE).cache().batch(256)

image_batch = next(iter(train_ds))
image_batch.shape

def show_batch(image_batch):
    plt.figure(figsize=(10,10))
    for n in range(25):
        plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n, :, :, 0], cmap='gray')
        plt.axis("off")
        
show_batch(image_batch)

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([SAMPLES, G_INPUT_DIMS])

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(8*8*128, use_bias=False, input_shape=(G_INPUT_DIMS,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 8, 128)))
    assert model.output_shape == (None, 8, 8, 128) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 16)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(8, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 64, 64, 8)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    print(model.output_shape)
    assert model.output_shape == (None, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)

    return model

generator = make_generator_model()

noise = tf.random.normal([1, G_INPUT_DIMS])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')

def make_discriminator_model(dropout):
    model = tf.keras.Sequential()
    
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[IMAGE_SIZE[0], IMAGE_SIZE[1], 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dropout))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dropout))
    
    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dropout))
    
    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dropout))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

discriminator = make_discriminator_model(D_DROP_PERC)
decision = discriminator(generated_image)
print (decision)

# helper function
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.fill(real_output.shape, 1 - TRUTH_SMOOTH), real_output)
    fake_loss = cross_entropy(tf.fill(fake_output.shape, 0 + FAKE_SMOOTH) , fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(G_LEARNING_RATE)
discriminator_optimizer = tf.keras.optimizers.Adam(D_LEARNING_RATE)


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, G_INPUT_DIMS])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    if epoch % PREVIEW_EVERY == 0:
        # Produce images for the GIF as we go
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                 epoch + 1,
                                 seed)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)
  
def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig(f"{OUTPUT_FOLDER}image_at_epoch_{epoch}.png")
  plt.show()
  
train(train_ds, EPOCHS)