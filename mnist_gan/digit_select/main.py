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
EPOCHS = 2
BATCH_SIZE = 1024
noise_dim = 100
num_examples_to_generate = 16





generator = keras.models.load_model(os.path.join(models_path, "digit_generator"))
classifier = keras.models.load_model(os.path.join(models_path, "digit_classifier"))


def make_noise_generator_model():
	model = tf.keras.Sequential()
	model.add(layers.Dense(300, input_shape=(100,)))
	model.add(layers.Dropout(0.2))
	model.add(layers.Dense(300))
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(100))
	return model

noise_generators = []
for i in range(10):
	noise_generators.append(make_noise_generator_model())



cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

def noise_generator_loss(classifier_output, digit_type):
	indices = [digit_type for i in range(BATCH_SIZE)]
	depth = 10
	one_hots = tf.one_hot(indices, depth)
	return cross_entropy(classifier_output, one_hots)

optimizer = tf.keras.optimizers.Adam(1e-4)

for i,noise_generator in enumerate(noise_generators):
	checkpoint_dir = './training_checkpoints_{}'.format(i)
	checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
	checkpoint = tf.train.Checkpoint(optimizer=optimizer,
									model=noise_generator)


#GIF seed
seed = tf.random.normal([num_examples_to_generate, noise_dim])












@tf.function
def train_step_0(digit_type, noise_generator):
	noise = tf.random.normal([BATCH_SIZE, noise_dim])

	with tf.GradientTape() as tape:
		generated_noise = noise_generator(noise, training=True)

		generated_images = generator(generated_noise)
		classifications = classifier(generated_images)

		loss = noise_generator_loss(classifications, digit_type)

	gradients = tape.gradient(loss, noise_generator.trainable_variables)
	optimizer.apply_gradients(zip(gradients, noise_generator.trainable_variables))

@tf.function
def train_step_1(digit_type, noise_generator):
	noise = tf.random.normal([BATCH_SIZE, noise_dim])

	with tf.GradientTape() as tape:
		generated_noise = noise_generator(noise, training=True)

		generated_images = generator(generated_noise)
		classifications = classifier(generated_images)

		loss = noise_generator_loss(classifications, digit_type)

	gradients = tape.gradient(loss, noise_generator.trainable_variables)
	optimizer.apply_gradients(zip(gradients, noise_generator.trainable_variables))

@tf.function
def train_step_2(digit_type, noise_generator):
	noise = tf.random.normal([BATCH_SIZE, noise_dim])

	with tf.GradientTape() as tape:
		generated_noise = noise_generator(noise, training=True)

		generated_images = generator(generated_noise)
		classifications = classifier(generated_images)

		loss = noise_generator_loss(classifications, digit_type)

	gradients = tape.gradient(loss, noise_generator.trainable_variables)
	optimizer.apply_gradients(zip(gradients, noise_generator.trainable_variables))

@tf.function
def train_step_3(digit_type, noise_generator):
	noise = tf.random.normal([BATCH_SIZE, noise_dim])

	with tf.GradientTape() as tape:
		generated_noise = noise_generator(noise, training=True)

		generated_images = generator(generated_noise)
		classifications = classifier(generated_images)

		loss = noise_generator_loss(classifications, digit_type)

	gradients = tape.gradient(loss, noise_generator.trainable_variables)
	optimizer.apply_gradients(zip(gradients, noise_generator.trainable_variables))

@tf.function
def train_step_4(digit_type, noise_generator):
	noise = tf.random.normal([BATCH_SIZE, noise_dim])

	with tf.GradientTape() as tape:
		generated_noise = noise_generator(noise, training=True)

		generated_images = generator(generated_noise)
		classifications = classifier(generated_images)

		loss = noise_generator_loss(classifications, digit_type)

	gradients = tape.gradient(loss, noise_generator.trainable_variables)
	optimizer.apply_gradients(zip(gradients, noise_generator.trainable_variables))

@tf.function
def train_step_5(digit_type, noise_generator):
	noise = tf.random.normal([BATCH_SIZE, noise_dim])

	with tf.GradientTape() as tape:
		generated_noise = noise_generator(noise, training=True)

		generated_images = generator(generated_noise)
		classifications = classifier(generated_images)

		loss = noise_generator_loss(classifications, digit_type)

	gradients = tape.gradient(loss, noise_generator.trainable_variables)
	optimizer.apply_gradients(zip(gradients, noise_generator.trainable_variables))

@tf.function
def train_step_6(digit_type, noise_generator):
	noise = tf.random.normal([BATCH_SIZE, noise_dim])

	with tf.GradientTape() as tape:
		generated_noise = noise_generator(noise, training=True)

		generated_images = generator(generated_noise)
		classifications = classifier(generated_images)

		loss = noise_generator_loss(classifications, digit_type)

	gradients = tape.gradient(loss, noise_generator.trainable_variables)
	optimizer.apply_gradients(zip(gradients, noise_generator.trainable_variables))

@tf.function
def train_step_7(digit_type, noise_generator):
	noise = tf.random.normal([BATCH_SIZE, noise_dim])

	with tf.GradientTape() as tape:
		generated_noise = noise_generator(noise, training=True)

		generated_images = generator(generated_noise)
		classifications = classifier(generated_images)

		loss = noise_generator_loss(classifications, digit_type)

	gradients = tape.gradient(loss, noise_generator.trainable_variables)
	optimizer.apply_gradients(zip(gradients, noise_generator.trainable_variables))

@tf.function
def train_step_8(digit_type, noise_generator):
	noise = tf.random.normal([BATCH_SIZE, noise_dim])

	with tf.GradientTape() as tape:
		generated_noise = noise_generator(noise, training=True)

		generated_images = generator(generated_noise)
		classifications = classifier(generated_images)

		loss = noise_generator_loss(classifications, digit_type)

	gradients = tape.gradient(loss, noise_generator.trainable_variables)
	optimizer.apply_gradients(zip(gradients, noise_generator.trainable_variables))

@tf.function
def train_step_9(digit_type, noise_generator):
	noise = tf.random.normal([BATCH_SIZE, noise_dim])

	with tf.GradientTape() as tape:
		generated_noise = noise_generator(noise, training=True)

		generated_images = generator(generated_noise)
		classifications = classifier(generated_images)

		loss = noise_generator_loss(classifications, digit_type)

	gradients = tape.gradient(loss, noise_generator.trainable_variables)
	optimizer.apply_gradients(zip(gradients, noise_generator.trainable_variables))


train_step_fns = [train_step_0, train_step_1, train_step_2, train_step_3, train_step_4, train_step_5, train_step_6, train_step_7, train_step_8, train_step_9]









#all_noise_tensors = [[] for i in range(10)] #tuples (epoch, tensor)



def train(epochs):
	for epoch in range(epochs):
		print("Starting epoch {}".format(epoch+1))
		start = time.time()

		for digit, noise_generator in enumerate(noise_generators):
			print("Digit {}".format(digit))

			train_step = train_step_fns[digit]
			train_step(digit, noise_generator)


			generated_noise = noise_generator(seed)
			#all_noise_tensors[digit].append((epoch, generated_noise))
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
	#plt.show()


checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
print("\nTRAINING")
train(EPOCHS)

"""
print("\nGENERATING IMAGES")
for digit, tupleyDoop in enumerate(all_noise_tensors):
	epoch, noise_tensor = tupleyDoop
	display.clear_output(wait=True)
	generate_and_save_images(noise_generators[digit], epoch, noise_tensor, digit)
"""

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
