from __future__ import absolute_import, division, print_function
import tensorflow as tf
import tensorflow_datasets as tfds

import math
import numpy as np
import tqdm.auto

tqdm.tqdm = tqdm.auto.tqdm

print(tf.__version__)

ds, md = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_ds, test_ds = ds['train'], ds['test']

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

num_train_examples = md.splits['train'].num_examples
num_test_examples = md.splits['test'].num_examples
print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples:   {}".format(num_test_examples))


# normalize/preprocess data
def normalize(images, labels):
   images = tf.cast(images, tf.float32)
   images /= 255
   return images, labels


train_ds = train_ds.map(normalize)
test_ds = test_ds.map(normalize)

# Flat+2Dense
print("CNN layers")
conv2d = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu,
                                input_shape=(28, 28, 1)),
max_pooling = tf.keras.layers.MaxPooling2D((2, 2), strides=2),
conv2d2 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu),
max_pooling2 = tf.keras.layers.MaxPooling2D((2, 2), strides=2),
flat = tf.keras.layers.Flatten(),
dense = tf.keras.layers.Dense(128, activation=tf.nn.relu),
dense2 = tf.keras.layers.Dense(10, activation=tf.nn.softmax)

# Sequential
print(".Sequential (defining model from layers)")
model = tf.keras.Sequential([
   tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu,
                          input_shape=(28, 28, 1)),
   tf.keras.layers.MaxPooling2D((2, 2), strides=2),
   tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu),
   tf.keras.layers.MaxPooling2D((2, 2), strides=2),
   tf.keras.layers.Flatten(),
   tf.keras.layers.Dense(128, activation=tf.nn.relu),
   tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# compile
print(".compile")
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics='accuracy')

# fit
print(".fit")
BATCH_SIZE = 30
train_ds = train_ds.repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_ds = test_ds.batch(BATCH_SIZE)

EPOCHS = 8
history = model.fit(train_ds,
                    steps_per_epoch=math.ceil(num_train_examples / BATCH_SIZE),
                    epochs=EPOCHS,
                    validation_data=test_ds,
                    validation_steps=math.ceil(num_test_examples / BATCH_SIZE)
                    )

# Evaluate metrics=accu
print(".evaluate")
test_loss, test_accuracy = model.evaluate(test_ds, steps=math.ceil(num_test_examples / BATCH_SIZE))
print("Accuracy on test dataset: ", test_accuracy)

# predict
print("predict")
for test_images, test_labels in test_ds.take(1):
   test_images = test_images.numpy()
   test_labels = test_labels.numpy()
   predictions = model.predict(test_images)

predictions.shape
print(predictions.shape)
print(predictions[0])

print(np.argmax(predictions[0]), "=", test_labels[0], "? if yes then its correct")
