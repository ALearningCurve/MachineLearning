import tensorflow as tf

import pprint as pp
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import layers
import time



BUFFER_SIZE = 60000
BATCH_SIZE = 256

EPOCHS = 200
noise_dim = 500
num_examples_to_generate = 16

set_num = 3

general_generator = keras.models.load_model("digit_generator")
classifier = keras.models.load_model("digit_classifier")
discriminator = keras.models.load_model("digit_discriminator")

"""
def make_generator_model():
	model = tf.keras.Sequential()
	model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Reshape((7, 7, 256)))
	assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

	model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation=tf.nn.leaky_relu))
	assert model.output_shape == (None, 7, 7, 128)
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation=tf.nn.leaky_relu))
	assert model.output_shape == (None, 14, 14, 64)
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
	assert model.output_shape == (None, 28, 28, 32)

	model.add(layers.Conv2DTranspose(1, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
	assert model.output_shape == (None, 28, 28, 1)

	return model
"""

def make_generator_model():
	model = tf.keras.Sequential()
	model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(noise_dim,)))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Reshape((7, 7, 256)))
	assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

	model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation=tf.nn.leaky_relu))
	assert model.output_shape == (None, 7, 7, 128)
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation=tf.nn.leaky_relu))
	assert model.output_shape == (None, 14, 14, 64)
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
	assert model.output_shape == (None, 28, 28, 32)

	model.add(layers.Conv2DTranspose(1, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
	assert model.output_shape == (None, 28, 28, 1)

	return model




digit_generators = []
for i in range(0, 10):
	digit_generators.append(make_generator_model())

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def generator_loss(fake_output, classifier_guess, generated_digit):
	discriminator_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
	indices = [generated_digit for i in range(BATCH_SIZE)]
	classifier_loss = cross_entropy(tf.one_hot(indices, 10), classifier_guess)
	total_loss = discriminator_loss + classifier_loss
	return total_loss
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
checkpoint_dirs = []
checkpoint_prefixes = []
checkpoints = []
for digit, generator in enumerate(digit_generators):
	checkpoint_dir = './training_checkpoints_{}'.format(digit)
	checkpoint_dirs.append(checkpoint_dir)
	checkpoint_prefixes.append(os.path.join(checkpoint_dir, "ckpt"))
	checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
								 generator=generator)
	checkpoints.append(checkpoint)




@tf.function
def train_step():
	noise = tf.random.normal([BATCH_SIZE, noise_dim])

	for digit, generator in enumerate(digit_generators):
		with tf.GradientTape() as gen_tape:
			generated_images = generator(noise, training=True)

			classifier_guess = classifier(generated_images)
			fake_output = discriminator(generated_images, training=True)

			gen_loss = generator_loss(fake_output, classifier_guess, digit)

		gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
		generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

seed = tf.random.normal([num_examples_to_generate, noise_dim])

def train(epochs):
	for epoch in range(epochs):
		print("Starting epoch {}".format(epoch+1))
		start = time.time()
		train_step()
		for digit, generator in enumerate(digit_generators):
			generate_and_save_images(generator,
								 epoch + 1,
								 seed,
								 digit)
		if (epoch + 1) % 10 == 0:
			for digit in range(10):
				checkpoint = checkpoints[digit]
				checkpoint_prefix = checkpoint_prefixes[digit]
				checkpoint.save(file_prefix = checkpoint_prefix)
		print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
	for digit, generator in enumerate(digit_generators):
		generate_and_save_images(generator,
							 epoch + 1,
							 seed,
							 digit)


def generate_and_save_images(model, epoch, test_input, digit):
	# Notice `training` is set to False.
	# This is so all layers run in inference mode (batchnorm).
	predictions = model(test_input, training=False)

	fig = plt.figure(figsize=(4,4))

	for i in range(predictions.shape[0]):
		plt.subplot(4, 4, i+1)
		plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
		plt.axis('off')

	if not os.path.exists("img_{}".format(set_num)):
		os.mkdir("img_{}".format(set_num))
	plt.savefig('img_{}/{}_image_at_epoch_{:04d}.png'.format(set_num, digit, epoch))
	plt.close(fig)



print("\n\nSTARTING TRAIN")
train(EPOCHS)