import tensorflow as tf

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
from tensorflow import keras
import time
import dotenv
import pprint as pp

from IPython import display


import git
repo = git.Repo(".", search_parent_directories=True)
working_tree_dir = repo.working_tree_dir
models_path = os.path.join(working_tree_dir, "_models")


#GLOBALS
EPOCHS = 90
BATCH_SIZE = 1024
noise_dim = 100
num_examples_to_generate = 16




generator = keras.models.load_model(os.path.join(models_path, "digit_generator"))
classifier = keras.models.load_model(os.path.join(models_path, "digit_classifier"))



noise_inputs = keras.Input(shape=(100,))
digit_input = keras.Input(shape=(1,))
x = layers.Dense(300, activation="relu")(inputs)
x = layers.Dropout(0.2)(x)
x = layers.Dense(300, activation="relu")(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(100)(x)
x = layers.concatenate([x, digit_input])
x = layers.Dense(100)(x)
noise_generator = tf.keras.Model(inputs=[noise_inputs, digit_input], outputs=outputs)



cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

def noise_generator_loss(classifier_output, digit_type):
	indices = [digit_type for i in range(BATCH_SIZE)]
	depth = 10
	one_hots = tf.one_hot(indices, depth)
	return cross_entropy(classifier_output, one_hots)



optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
								model=noise_generator)



#GIF seed
seed = tf.random.normal([num_examples_to_generate, noise_dim])






@tf.function
def train_step(digit_type, noise_generator):
	noise = tf.random.normal([BATCH_SIZE, noise_dim])

	with tf.GradientTape() as tape:
		generated_noise = noise_generator(noise, digit_type, training=True)

		generated_images = generator(generated_noise)
		classifications = classifier(generated_images)

		loss = noise_generator_loss(classifications, digit_type)

	gradients = tape.gradient(loss, noise_generator.trainable_variables)
	optimizer.apply_gradients(zip(gradients, noise_generator.trainable_variables))


def train(epochs):
	for epoch in range(epochs):
		print("Starting epoch {}".format(epoch+1))
		start = time.time()

		for digit in range(10):
			print("Digit {}".format(digit))

			train_step(digit, noise_generator)

			generated_noise = noise_generator(seed)
			display.clear_output(wait=True)
			generate_and_save_images(generator, epoch, generated_noise, digit)

		# Save the model every 15 epochs
		if (epoch + 1) % 15 == 0:
			checkpoint.save(file_prefix = checkpoint_prefix)

		print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))




def generate_and_save_images(model, epoch, test_input, digit):
	# Notice `training` is set to False.
	# This is so all layers run in inference mode (batchnorm).
	predictions = model(test_input, training=False)

	fig = plt.figure(figsize=(4,4))

	for i in range(predictions.shape[0]):
		plt.subplot(4, 4, i+1)
		numpy_predictions = predictions.numpy()
		plt.imshow(numpy_predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
		plt.axis('off')

	plt.savefig('image_{}_at_epoch_{:04d}.png'.format(digit, epoch))
	plt.close("all")




checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
print("\nTRAINING")
train(EPOCHS)




print("\nSAVING MODELS")
noise_generator.save("digit_generator_noise_generator")


# Display a single image using the epoch number
def display_image(epoch_no):
	return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

for digit in range(10):
	anim_file = 'dcgan_{}.gif'.format(digit)

	with imageio.get_writer(anim_file, mode='I') as writer:
		filenames = glob.glob('image_{}*.png'.format(digit))
		filenames = sorted(filenames)
		for filename in filenames:
			image = imageio.imread(filename)
			writer.append_data(image)
		image = imageio.imread(filename)
		writer.append_data(image)
