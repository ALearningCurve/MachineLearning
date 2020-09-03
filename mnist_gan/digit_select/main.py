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

from IPython import display


models_path = r"C:\Users\cayse\Desktop\WillyWally\models"

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
	return cross_entropy(classifier_output, digit_type)

optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
								model=noise_generator)


EPOCHS = 50
BATCH_SIZE = 1024
noise_dim = 100
num_examples_to_generate = 16

#GIF seed
seed = tf.random.normal([num_examples_to_generate, noise_dim])



@tf.function
def train_step(digit_type, noise_generator):
	noise = tf.random.normal([BATCH_SIZE, noise_dim])

	with tf.GradientTape() as tape:
		generated_noise = noise_generator(noise, training=True)

		generated_images = generator(generated_noise)
		classifications = classifier(generated_images)

		loss = noise_generator_loss(classifications, digit_type)

	gradients = tape.gradient(loss, noise_generator.trainable_variables)
	optimizer.apply_gradients(zip(gradients, noise_generator.trainable_variables))


def train(epochs):
	for epoch in range(epochs):
		start = time.time()

		for i, image_batch in enumerate(dataset, 1):
			print("Batch {}".format(i))
			train_step(image_batch)

		# Produce images for the GIF as we go
		display.clear_output(wait=True)
		generate_and_save_images(generator,
							 epoch + 1,
							 seed)

		# Save the model every 15 epochs
		if (epoch + 1) % 15 == 0:
			checkpoint.save(file_prefix = checkpoint_prefix)

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

	fig = plt.figure(figsize=(4,4))

	for i in range(predictions.shape[0]):
		plt.subplot(4, 4, i+1)
		plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
		plt.axis('off')

	plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
	#plt.show()


checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))



train(train_dataset, 1)

generator.save_model("digit_generator")
discriminator.save_model("digit_discriminator")


# Display a single image using the epoch number
def display_image(epoch_no):
	return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

display_image(EPOCHS)

anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
	filenames = glob.glob('image*.png')
	filenames = sorted(filenames)
	for filename in filenames:
		image = imageio.imread(filename)
		writer.append_data(image)
	image = imageio.imread(filename)
	writer.append_data(image)

import tensorflow_docs.vis.embed as embed
embed.embed_file(anim_file)