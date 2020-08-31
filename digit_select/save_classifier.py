import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

tf.get_logger().setLevel('WARNING')

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0
train_images, test_images = train_images.reshape((train_images.shape[0], 28, 28, 1)), test_images.reshape((test_images.shape[0], 28, 28, 1))


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), name="input/conv_1"))
model.add(layers.Dropout(0.2))
model.add(layers.MaxPooling2D((2, 2), name="pool_1"))
model.add(layers.Conv2D(64, (3, 3), activation='relu', name="conv_2"))
model.add(layers.MaxPooling2D((2, 2), name="pool_2"))
model.add(layers.Conv2D(64, (3, 3), activation='relu', name="conv_3"))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()



model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

